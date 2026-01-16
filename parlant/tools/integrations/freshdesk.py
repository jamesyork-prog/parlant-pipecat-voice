
import os
import logging
import httpx

import parlant.sdk as p

logger = logging.getLogger(__name__)


FRESHDESK_DOMAIN = os.environ.get("FRESHDESK_DOMAIN")
FRESHDESK_API_KEY = os.environ.get("FRESHDESK_API_KEY")


@p.tool
async def get_ticket(context: p.ToolContext, ticket_id: str) -> p.ToolResult:
    """
    Retrieves basic details of a Freshdesk ticket (metadata only, no description).

    Args:
        ticket_id (str): The ID of the ticket to retrieve.
    """

    if not FRESHDESK_DOMAIN or not FRESHDESK_API_KEY:
        return p.ToolResult(
            {"error": "Freshdesk credentials not configured."},
            metadata={"summary": "Error: Freshdesk credentials not configured."},
        )

    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}"
    auth = (FRESHDESK_API_KEY, "X")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, auth=auth)
            response.raise_for_status()
            ticket_data = response.json()
            
            # Return only metadata, no large text fields
            basic_data = {
                "id": ticket_data.get("id"),
                "subject": ticket_data.get("subject"),
                "status": ticket_data.get("status"),
                "priority": ticket_data.get("priority"),
                "type": ticket_data.get("type"),
                "tags": ticket_data.get("tags", []),
                "custom_fields": ticket_data.get("custom_fields", {}),
                "created_at": ticket_data.get("created_at"),
                "updated_at": ticket_data.get("updated_at"),
            }
            
            return p.ToolResult(
                basic_data, metadata={"summary": f"Fetched basic ticket info for {ticket_id}"}
            )
        except httpx.HTTPStatusError as e:
            return p.ToolResult(
                {
                    "error": f"Failed to fetch ticket: {e.response.status_code}",
                    "details": e.response.text,
                },
                metadata={"summary": f"Error fetching ticket {ticket_id}"},
            )
        except httpx.RequestError as e:
            return p.ToolResult(
                {"error": f"An error occurred while requesting {e.request.url!r}."},
                metadata={"summary": f"Error fetching ticket {ticket_id}"},
            )


@p.tool
async def get_ticket_description(context: p.ToolContext, ticket_id: str) -> p.ToolResult:
    """
    Retrieves the description/content of a Freshdesk ticket.

    Args:
        ticket_id (str): The ID of the ticket to retrieve.
    """

    if not FRESHDESK_DOMAIN or not FRESHDESK_API_KEY:
        return p.ToolResult(
            {"error": "Freshdesk credentials not configured."},
            metadata={"summary": "Error: Freshdesk credentials not configured."},
        )

    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}"
    auth = (FRESHDESK_API_KEY, "X")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, auth=auth)
            response.raise_for_status()
            ticket_data = response.json()
            
            # Return only description fields
            description_data = {
                "ticket_id": ticket_data.get("id"),
                "description": ticket_data.get("description", ""),
                "description_text": ticket_data.get("description_text", ""),
            }
            
            return p.ToolResult(
                description_data, metadata={"summary": f"Fetched ticket description for {ticket_id}"}
            )
        except httpx.HTTPStatusError as e:
            return p.ToolResult(
                {
                    "error": f"Failed to fetch ticket: {e.response.status_code}",
                    "details": e.response.text,
                },
                metadata={"summary": f"Error fetching ticket {ticket_id}"},
            )
        except httpx.RequestError as e:
            return p.ToolResult(
                {"error": f"An error occurred while requesting {e.request.url!r}."},
                metadata={"summary": f"Error fetching ticket {ticket_id}"},
            )


@p.tool
async def get_ticket_conversations(context: p.ToolContext, ticket_id: str) -> p.ToolResult:
    """
    Retrieves the conversation history (notes and replies) for a Freshdesk ticket.

    Args:
        ticket_id (str): The ID of the ticket to retrieve conversations for.
    """

    if not FRESHDESK_DOMAIN or not FRESHDESK_API_KEY:
        return p.ToolResult(
            {"error": "Freshdesk credentials not configured."},
            metadata={"summary": "Error: Freshdesk credentials not configured."},
        )

    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}/conversations"
    auth = (FRESHDESK_API_KEY, "X")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, auth=auth)
            response.raise_for_status()
            conversations = response.json()
            
            # Summarize conversations to keep under size limit
            # Focus on private notes which contain booking info
            conversation_summary = []
            for conv in conversations[:10]:  # Limit to 10 most recent
                # Keep more text for private notes (they have booking info)
                char_limit = 3000 if conv.get("private") else 500
                conversation_summary.append({
                    "id": conv.get("id"),
                    "body_text": conv.get("body_text", "")[:char_limit],
                    "incoming": conv.get("incoming"),
                    "private": conv.get("private"),
                    "created_at": conv.get("created_at"),
                })
            
            return p.ToolResult(
                {"ticket_id": ticket_id, "conversations": conversation_summary},
                metadata={"summary": f"Fetched {len(conversation_summary)} conversations for ticket {ticket_id}"}
            )
        except httpx.HTTPStatusError as e:
            return p.ToolResult(
                {
                    "error": f"Failed to fetch ticket: {e.response.status_code}",
                    "details": e.response.text,
                },
                metadata={"summary": f"Error fetching ticket {ticket_id}"},
            )
        except httpx.RequestError as e:
            return p.ToolResult(
                {"error": f"An error occurred while requesting {e.request.url!r}."},
                metadata={"summary": f"Error fetching ticket {ticket_id}"},
            )

@p.tool
async def add_note(context: p.ToolContext, ticket_id: str, note: str) -> p.ToolResult:
    """
    Adds a private note to a Freshdesk ticket with deduplication.

    Args:
        ticket_id (str): The ID of the ticket to add the note to.
        note (str): The content of the note.
    """

    if not FRESHDESK_DOMAIN or not FRESHDESK_API_KEY:
        return p.ToolResult(
            {"error": "Freshdesk credentials not configured."},
            metadata={"summary": "Error: Freshdesk credentials not configured."},
        )

    # Check for duplicate notes by getting recent conversations
    try:
        # Get existing conversations to check for duplicates
        conversations_url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}/conversations"
        auth = (FRESHDESK_API_KEY, "X")
        
        async with httpx.AsyncClient() as client:
            # Check recent conversations for duplicate content
            conv_response = await client.get(conversations_url, auth=auth)
            if conv_response.status_code == 200:
                conversations = conv_response.json()
                
                # Check last 3 conversations for duplicate content
                for conv in conversations[-3:]:
                    if conv.get("private", False) and conv.get("body_text", "").strip() == note.strip():
                        logger.info(f"Duplicate note detected for ticket {ticket_id}, skipping")
                        return p.ToolResult(
                            {"message": "Duplicate note skipped", "note_id": conv.get("id")},
                            metadata={"summary": f"Skipped duplicate note for ticket {ticket_id}"}
                        )
    except Exception as e:
        logger.warning(f"Could not check for duplicate notes: {e}")
        # Continue with adding note if duplicate check fails

    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}/notes"
    payload = {"body": note, "private": True}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, auth=auth, json=payload)
            response.raise_for_status()
            note_data = response.json()
            logger.info(f"Added new note to ticket {ticket_id}")
            return p.ToolResult(
                note_data, metadata={"summary": f"Added note to ticket {ticket_id}"}
            )
        except httpx.HTTPStatusError as e:
            return p.ToolResult(
                {
                    "error": f"Failed to add note: {e.response.status_code}",
                    "details": e.response.text,
                },
                metadata={"summary": f"Error adding note to ticket {ticket_id}"},
            )
        except httpx.RequestError as e:
            return p.ToolResult(
                {"error": f"An error occurred while requesting {e.request.url!r}."},
                metadata={"summary": f"Error adding note to ticket {ticket_id}"},
            )

@p.tool
async def update_ticket(
    context: p.ToolContext,
    ticket_id: str,
    status: int = None,
    priority: int = None,
    tags: list = None
) -> p.ToolResult:
    """
    Updates the status, priority, and/or tags of a Freshdesk ticket.

    Args:
        ticket_id (str): The ID of the ticket to update.
        status (int, optional): The new status of the ticket.
        priority (int, optional): The new priority of the ticket.
        tags (list, optional): A list of tags to add to the ticket.
    """

    if not FRESHDESK_DOMAIN or not FRESHDESK_API_KEY:
        return p.ToolResult(
            {"error": "Freshdesk credentials not configured."},
            metadata={"summary": "Error: Freshdesk credentials not configured."},
        )

    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}"
    auth = (FRESHDESK_API_KEY, "X")

    payload = {}
    if status is not None:
        payload["status"] = status
    if priority is not None:
        payload["priority"] = priority
    if tags is not None:
        payload["tags"] = tags

    if not payload:
        return p.ToolResult(
            {"error": "No fields to update."},
            metadata={"summary": "No fields provided to update ticket."},
        )

    async with httpx.AsyncClient() as client:
        try:
            response = await client.put(url, auth=auth, json=payload)
            response.raise_for_status()
            ticket_data = response.json()
            return p.ToolResult(
                ticket_data, metadata={"summary": f"Updated ticket {ticket_id}"}
            )
        except httpx.HTTPStatusError as e:
            return p.ToolResult(
                {
                    "error": f"Failed to update ticket: {e.response.status_code}",
                    "details": e.response.text,
                },
                metadata={"summary": f"Error updating ticket {ticket_id}"},
            )
        except httpx.RequestError as e:
            return p.ToolResult(
                {
                    "error": f"An error occurred while requesting {e.request.url!r}."
                },
                metadata={"summary": f"Error updating ticket {ticket_id}"},
            )


# ============================================================================
# Helper functions for direct API calls (without Parlant context)
# These are used by infrastructure components like HatchetWorker
# ============================================================================

async def add_note_direct(ticket_id: str, note: str) -> dict:
    """
    Adds a private note to a Freshdesk ticket (direct API call).
    
    This is a helper function that can be called without a Parlant context.
    Used by infrastructure components like HatchetWorker.
    
    Args:
        ticket_id: The ID of the ticket to add the note to.
        note: The content of the note.
        
    Returns:
        dict with note data on success, or error information on failure
        
    Raises:
        Exception: If the API call fails after all retries
    """
    if not FRESHDESK_DOMAIN or not FRESHDESK_API_KEY:
        raise ValueError("Freshdesk credentials not configured")

    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}/notes"
    auth = (FRESHDESK_API_KEY, "X")
    payload = {"body": note, "private": True}

    async with httpx.AsyncClient() as client:
        response = await client.post(url, auth=auth, json=payload)
        response.raise_for_status()
        return response.json()


async def update_ticket_direct(ticket_id: str, updates: dict) -> dict:
    """
    Updates a Freshdesk ticket (direct API call).
    
    This is a helper function that can be called without a Parlant context.
    Used by infrastructure components like HatchetWorker.
    
    Args:
        ticket_id: The ID of the ticket to update.
        updates: Dictionary with fields to update (status, priority, tags, etc.)
        
    Returns:
        dict with updated ticket data on success
        
    Raises:
        ValueError: If no fields to update or credentials not configured
        Exception: If the API call fails
    """
    if not FRESHDESK_DOMAIN or not FRESHDESK_API_KEY:
        raise ValueError("Freshdesk credentials not configured")

    if not updates:
        raise ValueError("No fields to update")

    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}"
    auth = (FRESHDESK_API_KEY, "X")

    async with httpx.AsyncClient() as client:
        response = await client.put(url, auth=auth, json=updates)
        response.raise_for_status()
        return response.json()
