
import os
import httpx
import asyncio
import parlant.sdk as p


LAKERA_API_KEY = os.environ.get("LAKERA_API_KEY")
LAKERA_API_URL = "https://api.lakera.ai/v1/prompt_injection"


@p.tool
async def check_content(context: p.ToolContext) -> p.ToolResult:
    """
    Checks content for security threats using Lakera AI's prompt injection detection.
    
    This tool scans text content for potential security risks including:
    - Prompt injection attacks
    - Jailbreak attempts
    - Personally Identifiable Information (PII)
    
    Args:
        content (str): The text content to check for security threats
    
    Returns:
        ToolResult with:
        - safe (bool): True if content is safe, False if flagged
        - flagged (bool): Whether the content was flagged by Lakera
        - categories (dict): Which security categories were flagged
        - category_scores (dict): Confidence scores for each category
        - content (str): The original content that was checked
    """
    content = context.inputs.get("content", "")
    
    # Handle empty content
    if not content or content.strip() == "":
        return p.ToolResult(
            {"safe": True, "flagged": False, "content": content},
            metadata={"summary": "Content check: safe (empty)"}
        )
    
    # Check if API key is configured
    if not LAKERA_API_KEY:
        return p.ToolResult(
            {"error": "Lakera API key not configured"},
            metadata={"summary": "Error: Lakera API key not configured"}
        )
    
    headers = {
        "Authorization": f"Bearer {LAKERA_API_KEY}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "input": content
    }
    
    async with httpx.AsyncClient() as client:
        max_retries = 2
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                response = await client.post(
                    LAKERA_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=10.0
                )
                response.raise_for_status()
                
                data = response.json()
                
                # Parse Lakera response
                result = data.get("results", [{}])[0]
                flagged = result.get("flagged", False)
                categories = result.get("categories", {})
                category_scores = result.get("category_scores", {})
                
                return p.ToolResult(
                    {
                        "safe": not flagged,
                        "flagged": flagged,
                        "categories": categories,
                        "category_scores": category_scores,
                        "content": content,
                    },
                    metadata={"summary": f"Content check: {'flagged' if flagged else 'safe'}"}
                )
                
            except httpx.HTTPStatusError as e:
                # Check if it's a rate limit error (429)
                if e.response.status_code == 429 and retry_count < max_retries:
                    retry_count += 1
                    wait_time = 60  # Wait 60 seconds for rate limit
                    await asyncio.sleep(wait_time)
                    continue
                
                return p.ToolResult(
                    {
                        "error": f"Lakera API error: {e.response.status_code}",
                        "details": e.response.text,
                    },
                    metadata={"summary": "Error checking content"}
                )
            except httpx.RequestError as e:
                return p.ToolResult(
                    {"error": f"An error occurred while checking content: {str(e)}"},
                    metadata={"summary": "Error checking content"}
                )
