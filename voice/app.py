import os
import asyncio
from pathlib import Path
import traceback
from typing import Optional

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
ICE_SERVERS = os.getenv("ICE_SERVERS", "stun:stun.l.google.com:19302")  # comma-separated

# Tuning knobs for the event pump
TURN_TIMEOUT_SECS = int(os.getenv("TURN_TIMEOUT_SECS", "30"))
POLL_WAIT_SECS    = int(os.getenv("POLL_WAIT_SECS", "5"))
MAX_EMPTY_POLLS   = int(os.getenv("MAX_EMPTY_POLLS", "3"))


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
            # Format: stun:host:port or turn:user:pass@host:port
            if server.startswith('stun:'):
                servers.append({"urls": [server]})
            elif server.startswith('turn:'):
                # Expected format: turn:username:credential@host:port
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


# ---------- Parlant bridge ----------
class ParlantBridge(FrameProcessor):
    """
    Turn-level bridge:
      • BEFORE posting user's message, get current tail offset, set baseline = tail+1.
      • Post user's message (customer event).
      • Run an event pump until inactivity or TURN_TIMEOUT_SECS, speaking EVERY agent message.
    """

    def __init__(self, client: AsyncParlantClient, agent_id: str):
        super().__init__()
        self._client = client
        self._agent_id = agent_id
        self._session_id: Optional[str] = None
        self._min_offset: int = 0
        self._processing_lock = asyncio.Lock()

    async def open_session(self):
        log_header("Opening Parlant session")
        try:
            session = await self._client.sessions.create(agent_id=self._agent_id)
            self._session_id = session.id
            self._min_offset = 0
            log_info(f"Session opened OK: {self._session_id}")
        except Exception as e:
            log_error("Failed to create Parlant session", e)
            raise

    async def get_tail_offset(self) -> int:
        """Return highest current offset, or -1 if no events yet."""
        if not self._session_id:
            return -1
        try:
            events = await self._client.sessions.list_events(
                self._session_id, min_offset=0, wait_for_data=0
            )
            if not events:
                return -1
            return max(ev.offset for ev in events)
        except Exception as e:
            log_error("get_tail_offset failed (defaulting to current baseline)", e)
            return max(-1, self._min_offset - 1)

    # async def _event_pump(self, deadline_monotonic: float):
        """
        Keep polling for new events until:
          • we hit TURN_TIMEOUT_SECS, or
          • MAX_EMPTY_POLLS consecutive polls return nothing.
        Speak EVERY agent message as it arrives.
        """
        empty_polls = 0
        while True:
            now = asyncio.get_running_loop().time()
            if now >= deadline_monotonic:
                log_info("Turn timeout reached; ending pump.")
                break

            time_left = max(0.0, deadline_monotonic - now)
            wait = min(POLL_WAIT_SECS, int(time_left)) or 1

            try:
                events = await self._client.sessions.list_events(
                    self._session_id,
                    min_offset=self._min_offset,
                    wait_for_data=wait,
                )
            except Exception as e:
                # Gateway timeout is expected when no more events
                if "504" in str(e) or "timed out" in str(e).lower():
                    log_info("Event stream timed out (turn complete)")
                    break
                else:
                    log_error("list_events failed", e)
                    break

            if not events:
                empty_polls += 1
                log_info(f"[pump] idle (#{empty_polls}/{MAX_EMPTY_POLLS})")
                if empty_polls >= MAX_EMPTY_POLLS:
                    log_info("No new events; ending pump.")
                    break
                continue

            empty_polls = 0
            log_info(f"[pump] got {len(events)} event(s)")
            for ev in events:
                self._min_offset = max(self._min_offset, ev.offset + 1)

                kind = getattr(ev, "kind", "?")
                src  = getattr(ev, "source", "?")
                data = getattr(ev, "data", {}) or {}
                preview = (data.get("message") or "")[:160].replace("\n", " ")
                log_info(f"← event kind={kind} source={src} offset={ev.offset} message_preview={preview!r}")

                if kind == "message" and src in ("ai_agent", "assistant"):
                    reply_text = data.get("message", "")
                    if reply_text:
                        log_header("Streaming agent reply to TTS")
                        log_info(f"agent replies → {reply_text!r}")
                        try:
                            await self.push_frame(LLMFullResponseStartFrame())
                            await self.push_frame(TextFrame(text=reply_text))
                            await self.push_frame(LLMFullResponseEndFrame())
                        except Exception as e:
                            log_error("Failed to push frames for TTS", e)

    async def _event_pump(self, deadline_monotonic: float):
        """
        Keep polling for new events until:
        • we hit TURN_TIMEOUT_SECS, or
        • MAX_EMPTY_POLLS consecutive polls return nothing.
        Speak EVERY agent message as it arrives.
        """
        empty_polls = 0
        while True:
            now = asyncio.get_running_loop().time()
            if now >= deadline_monotonic:
                log_info("Turn timeout reached; ending pump.")
                break

            time_left = max(0.0, deadline_monotonic - now)
            wait = min(POLL_WAIT_SECS, int(time_left)) or 1

            try:
                events = await self._client.sessions.list_events(
                    self._session_id,
                    min_offset=self._min_offset,
                    wait_for_data=wait,
                )
            except Exception as e:
                # Gateway timeout is expected when no more events
                if "504" in str(e) or "timed out" in str(e).lower():
                    log_info("Event stream timed out (turn complete)")
                    break
                else:
                    log_error("list_events failed", e)
                    break

            if not events:
                empty_polls += 1
                log_info(f"[pump] idle (#{empty_polls}/{MAX_EMPTY_POLLS})")
                if empty_polls >= MAX_EMPTY_POLLS:
                    log_info("No new events; ending pump.")
                    break
                continue

            empty_polls = 0
            log_info(f"[pump] got {len(events)} event(s)")
            for ev in events:
                self._min_offset = max(self._min_offset, ev.offset + 1)

                kind = getattr(ev, "kind", "?")
                src  = getattr(ev, "source", "?")
                data = getattr(ev, "data", {}) or {}
                preview = (data.get("message") or "")[:160].replace("\n", " ")
                log_info(f"← event kind={kind} source={src} offset={ev.offset} message_preview={preview!r}")

                if kind == "message" and src in ("ai_agent", "assistant"):
                    reply_text = data.get("message", "")
                    if reply_text:
                        log_header("Streaming agent reply to TTS")
                        log_info(f"agent replies → {reply_text!r}")
                        try:
                            # Send to TTS for voice output
                            await self.push_frame(LLMFullResponseStartFrame())
                            await self.push_frame(TextFrame(text=reply_text))
                            await self.push_frame(LLMFullResponseEndFrame())
                        except Exception as e:
                            log_error("Failed to push frames for TTS", e)

    async def _send_and_stream_reply(self, user_text: str):
        if not self._session_id:
            log_error("No Parlant session id; dropping message.")
            return

        # IMPORTANT: Get tail and set baseline BEFORE posting
        tail = await self.get_tail_offset()
        new_baseline = tail + 1
        self._min_offset = new_baseline  # Set this BEFORE posting!
        
        log_header("Baseline before posting")
        log_info(f"current tail offset = {tail}")
        log_info(f"setting baseline (min_offset) to {new_baseline}")

        # Now post the customer's message
        log_header("Sending customer message to Parlant")
        log_info(f"customer says → {user_text!r}")
        try:
            await self._client.sessions.create_event(
                self._session_id, kind="message", source="customer", message=user_text
            )
            log_info("create_event OK")
        except Exception as e:
            log_error("create_event failed", e)
            return

        log_header("Event pump (speaking all replies for this turn)")
        deadline = asyncio.get_running_loop().time() + TURN_TIMEOUT_SECS
        await self._event_pump(deadline)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InterimTranscriptionFrame):
            log_info(f"(interim) heard: {frame.text!r}")
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, TextFrame):
            log_header("Received STT final from Pipecat")
            log_info(f"heard final: {frame.text!r}")
            try:
                setattr(frame, "skip_tts", True)
            except Exception:
                pass

            await self.push_frame(frame, direction)

            # CHANGE THIS: await instead of create_task
            await self._send_and_stream_reply(frame.text)
            return

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
    # Parse ICE servers from environment
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
                # Pass ICE servers to WebRTC transport
                ice_servers=ice_servers if ice_servers else None,
            ),
        },
    )
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    import sys
    
    # Force host and port for Docker compatibility
    sys.argv = ["app.py", "--host", "0.0.0.0", "--port", "7860"]
    
    from pipecat.runner.run import main
    main()