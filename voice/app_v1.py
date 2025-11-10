import os
import asyncio
from pathlib import Path
import traceback
from typing import Optional
from enum import Enum

from dotenv import load_dotenv

# ===== Pipecat (0.0.86) =====
from pipecat.frames.frames import (
    Frame,
    TextFrame,
    InterimTranscriptionFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.processors.frameworks.rtvi import RTVIProcessor, RTVIConfig, RTVIObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.pipeline.runner import PipelineRunner
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import TransportParams

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService

# ===== Parlant client =====
from parlant.client import AsyncParlantClient

# ---------- Env / constants ----------
AGENT_ID_FILE = Path(__file__).parent / ".." / "data" / "agent_id.txt"

load_dotenv(override=True)

PARLANT_BASE_URL = os.getenv("PARLANT_BASE_URL", "http://parlant:8800")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ENV_AGENT_ID = (os.getenv("PARLANT_AGENT_ID") or "").strip() or None

# WebRTC ICE configuration
ICE_SERVERS = os.getenv("ICE_SERVERS", "stun:stun.l.google.com:19302")

# Event-driven configuration (much simpler)
STREAM_TIMEOUT_SECS = int(os.getenv("STREAM_TIMEOUT_SECS", "30"))  # Max time for a single turn


# ---------- Logging helpers ----------
def log_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def log_info(msg: str):
    print(f"[voice] {msg}")


def log_error(msg: str, exc: Exception | None = None):
    print(f"[voice][ERROR] {msg}")
    if exc:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        print(tb.strip())


# ---------- Agent ID helpers ----------
def read_agent_id_from_file() -> str:
    if not AGENT_ID_FILE.exists():
        raise RuntimeError(
            f"{AGENT_ID_FILE} not found. Start the Parlant server first so it writes the agent id."
        )
    return AGENT_ID_FILE.read_text(encoding="utf-8").strip()


def resolve_agent_id() -> str:
    if ENV_AGENT_ID:
        log_info(f"Using agent id from env: {ENV_AGENT_ID}")
        return ENV_AGENT_ID
    aid = read_agent_id_from_file()
    log_info(f"Using agent id from file: {aid}")
    return aid


def parse_ice_servers():
    """Parse ICE_SERVERS env var into aiortc RTCConfiguration format."""
    servers = []
    for server in ICE_SERVERS.split(','):
        server = server.strip()
        if server:
            if server.startswith('stun:'):
                servers.append({"urls": [server]})
            elif server.startswith('turn:'):
                if '@' in server:
                    parts = server.split('@')
                    creds = parts[0].replace('turn:', '').split(':')
                    url = f"turn:{parts[1]}"
                    if len(creds) >= 2:
                        servers.append({
                            "urls": [url],
                            "username": creds[0],
                            "credential": creds[1]
                        })
                else:
                    servers.append({"urls": [server]})
    
    log_info(f"ICE servers configured: {len(servers)} server(s)")
    for srv in servers:
        log_info(f"  - {srv.get('urls', ['unknown'])[0]}")
    
    return servers


# ---------- State machine for turn management ----------
class TurnState(Enum):
    IDLE = "idle"                    # Waiting for user input
    SPEAKING = "speaking"            # Agent is speaking
    PROCESSING = "processing"        # Parlant is processing (tools, thinking)
    

# ---------- Event-driven Parlant bridge ----------
class ParlantBridge(FrameProcessor):
    """
    Event-driven bridge using observer pattern:
    - Subscribes to Parlant event stream (single long-lived connection)
    - State machine tracks conversation state
    - Zero polling, pure event-driven
    """

    def __init__(self, client: AsyncParlantClient, agent_id: str):
        super().__init__()
        self._client = client
        self._agent_id = agent_id
        self._session_id: Optional[str] = None
        self._min_offset: int = 0
        
        # State management
        self._state = TurnState.IDLE
        self._state_lock = asyncio.Lock()
        
        # Event stream observer
        self._stream_task: Optional[asyncio.Task] = None
        self._event_queue = asyncio.Queue()
        
        # Message buffering
        self._pending_messages = []

    async def open_session(self):
        log_header("Opening Parlant session")
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                session = await self._client.sessions.create(agent_id=self._agent_id)
                self._session_id = session.id
                self._min_offset = 0
                log_info(f"Session opened OK: {self._session_id}")
                
                # Start the event stream observer
                self._stream_task = asyncio.create_task(self._event_stream_observer())
                log_info("Event stream observer started")
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    log_info(f"Connection attempt {attempt + 1}/{max_retries} failed, retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                else:
                    log_error("Failed to create Parlant session after all retries", e)
                    raise

    async def _event_stream_observer(self):
        """
        Long-lived task that maintains a persistent connection to Parlant's event stream.
        Pushes events to the queue as they arrive.
        """
        log_info("🔄 Event stream observer running")
        
        while True:
            try:
                # Long-polling: wait up to STREAM_TIMEOUT_SECS for new events
                events = await self._client.sessions.list_events(
                    self._session_id,
                    min_offset=self._min_offset,
                    wait_for_data=STREAM_TIMEOUT_SECS,
                )
                
                if events:
                    for ev in events:
                        self._min_offset = max(self._min_offset, ev.offset + 1)
                        await self._event_queue.put(ev)
                        
            except Exception as e:
                if "504" in str(e) or "timed out" in str(e).lower():
                    # Timeout is normal, just retry
                    continue
                else:
                    log_error("Event stream error, reconnecting...", e)
                    await asyncio.sleep(2)

    async def _transition_state(self, new_state: TurnState):
        """Thread-safe state transition."""
        async with self._state_lock:
            old_state = self._state
            self._state = new_state
            log_info(f"🔀 State: {old_state.value} → {new_state.value}")

    async def _handle_event(self, event):
        """Process a single event from Parlant."""
        kind = getattr(event, "kind", "?")
        src = getattr(event, "source", "?")
        data = getattr(event, "data", {}) or {}
        
        log_info(f"📨 Event: kind={kind} source={src} offset={event.offset}")
        
        # Agent message - buffer it
        if kind == "message" and src in ("ai_agent", "assistant"):
            message = data.get("message", "")
            if message:
                log_info(f"💬 Buffering: {message[:100]}...")
                self._pending_messages.append(message)
        
        # Tool execution started
        elif kind == "tool_call":
            await self._transition_state(TurnState.PROCESSING)
            log_info(f"🔧 Tool called: {data.get('tool_name', 'unknown')}")
        
        # Check if turn is complete (heuristic: agent sent message and we're in PROCESSING)
        # When we receive an agent message, it usually means the turn is done
        if kind == "message" and src in ("ai_agent", "assistant") and self._state != TurnState.SPEAKING:
            # Flush buffered messages
            await self._flush_messages()

    async def _flush_messages(self):
        """Send all buffered messages to TTS."""
        if not self._pending_messages:
            return
        
        await self._transition_state(TurnState.SPEAKING)
        
        log_header("🗣️  Speaking buffered messages")
        for message in self._pending_messages:
            log_info(f"Speaking: {message[:100]}...")
            try:
                await self.push_frame(LLMFullResponseStartFrame())
                await self.push_frame(TextFrame(text=message))
                await self.push_frame(LLMFullResponseEndFrame())
                
                # Brief pause between messages
                await asyncio.sleep(0.3)
            except Exception as e:
                log_error("Failed to speak message", e)
        
        self._pending_messages.clear()
        await self._transition_state(TurnState.IDLE)

    async def _send_user_message(self, user_text: str):
        """Send user message and start processing turn."""
        if not self._session_id:
            log_error("No Parlant session; dropping message.")
            return
        
        log_header("📤 Sending user message")
        log_info(f"User: {user_text}")
        
        await self._transition_state(TurnState.PROCESSING)
        
        try:
            await self._client.sessions.create_event(
                self._session_id, 
                kind="message", 
                source="customer", 
                message=user_text
            )
            log_info("✅ Message sent")
            
            # Start turn processor that watches for events
            asyncio.create_task(self._process_turn())
            
        except Exception as e:
            log_error("Failed to send message", e)
            await self._transition_state(TurnState.IDLE)

    async def _process_turn(self):
        """
        Process events for the current turn.
        Exits when we detect turn completion (no events for a short time after agent response).
        """
        log_info("⏳ Processing turn...")
        turn_complete = False
        idle_time = 0
        check_interval = 0.5  # Check every 500ms
        max_idle = 2.0  # Consider turn done after 2s idle following agent message
        
        while not turn_complete:
            try:
                # Try to get event with short timeout
                event = await asyncio.wait_for(
                    self._event_queue.get(), 
                    timeout=check_interval
                )
                
                # Reset idle counter - we got an event
                idle_time = 0
                await self._handle_event(event)
                
            except asyncio.TimeoutError:
                # No event received
                idle_time += check_interval
                
                # If we have pending messages and we've been idle, flush them
                if self._pending_messages and idle_time >= max_idle:
                    await self._flush_messages()
                    turn_complete = True
                    log_info("✅ Turn complete")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Block all input while agent is speaking
        if isinstance(frame, (InterimTranscriptionFrame, TextFrame)):
            if self._state == TurnState.SPEAKING:
                log_info(f"🚫 Blocked input (agent speaking): {getattr(frame, 'text', '')[:50]}...")
                return  # Drop frame
        
        # Handle interim transcription
        if isinstance(frame, InterimTranscriptionFrame):
            log_info(f"(interim) {frame.text[:50]}...")
            await self.push_frame(frame, direction)
            return

        # Handle final transcription - send to Parlant
        if isinstance(frame, TextFrame):
            log_header("🎤 Final transcription")
            log_info(f"Heard: {frame.text}")
            
            # Mark to skip TTS (this is user input, not agent output)
            try:
                setattr(frame, "skip_tts", True)
            except Exception:
                pass
            
            await self.push_frame(frame, direction)
            
            # Send to Parlant (non-blocking)
            asyncio.create_task(self._send_user_message(frame.text))
            return

        # Pass through all other frames
        await self.push_frame(frame, direction)


# ---------- Bot wiring ----------
async def run_bot(transport, runner_args: RunnerArguments):
    log_header("Voice bridge startup")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing. Set it in .env")

    log_info(f"PARLANT_BASE_URL = {PARLANT_BASE_URL}")
    agent_id = resolve_agent_id()
    log_info(f"PARLANT_AGENT_ID = {agent_id}")

    client = AsyncParlantClient(base_url=PARLANT_BASE_URL)

    stt = OpenAISTTService(api_key=OPENAI_API_KEY, model="gpt-4o-transcribe")
    tts = OpenAITTSService(api_key=OPENAI_API_KEY, model="gpt-4o-mini-tts", voice="alloy")
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    vad = SileroVADAnalyzer(params=VADParams(confidence=0.7, start_secs=0.2, stop_secs=0.8))
    bridge = ParlantBridge(client, agent_id)

    pipeline = Pipeline([
        transport.input(),
        rtvi,
        stt,
        bridge,
        tts,
        transport.output(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        observers=[RTVIObserver(rtvi)],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(tr, client_):
        log_header("Browser connected (WebRTC)")
        try:
            await bridge.open_session()
        except Exception as e:
            log_error("open_session failed", e)

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(tr, client_):
        log_header("Browser disconnected (WebRTC)")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    log_info("Starting pipeline runner… (open the printed /client URL and allow mic)")
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    ice_servers = parse_ice_servers()
    
    transport = await create_transport(
        runner_args,
        {
            "webrtc": lambda: TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_analyzer=SileroVADAnalyzer(
                    params=VADParams(confidence=0.7, start_secs=0.2, stop_secs=0.8)
                ),
                ice_servers=ice_servers if ice_servers else None,
            ),
        },
    )
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    import sys
    
    sys.argv = ["app.py", "--host", "0.0.0.0", "--port", "7860"]
    
    from pipecat.runner.run import main
    main()