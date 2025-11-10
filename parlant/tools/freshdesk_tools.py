
import os
import httpx

import parlant.sdk as p


FRESHDESK_DOMAIN = os.environ.get("FRESHDESK_DOMAIN")
FRESHDESK_API_KEY = os.environ.get("FRESHDESK_API_KEY")


@p.tool
async def get_ticket(context: p.ToolContext) -> p.ToolResult:
    """
    Retrieves the details of a specific Freshdesk ticket.

    Args:
        ticket_id (str): The ID of the ticket to retrieve.
    """
    ticket_id = context.inputs.get("ticket_id")

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
            response.raise_for_status()  # Raise an exception for bad status codes
            ticket_data = response.json()
            return p.ToolResult(
                ticket_data, metadata={"summary": f"Fetched ticket details for {ticket_id}"}
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
async def add_note(context: p.ToolContext) -> p.ToolResult:
    """
    Adds a private note to a Freshdesk ticket.

    Args:
        ticket_id (str): The ID of the ticket to add the note to.
        note (str): The content of the note.
    """
    ticket_id = context.inputs.get("ticket_id")
    note = context.inputs.get("note")

    if not FRESHDESK_DOMAIN or not FRESHDESK_API_KEY:
        return p.ToolResult(
            {"error": "Freshdesk credentials not configured."},
            metadata={"summary": "Error: Freshdesk credentials not configured."},
        )

    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}/notes"
    auth = (FRESHDESK_API_KEY, "X")
    payload = {"body": note, "private": True}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, auth=auth, json=payload)
            response.raise_for_status()
            note_data = response.json()
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
async def update_ticket(context: p.ToolContext) -> p.ToolResult:
    """
    Updates the status, priority, and/or tags of a Freshdesk ticket.

    Args:
        ticket_id (str): The ID of the ticket to update.
        status (int, optional): The new status of the ticket.
        priority (int, optional): The new priority of the ticket.
        tags (list, optional): A list of tags to add to the ticket.
    """
    ticket_id = context.inputs.get("ticket_id")
    status = context.inputs.get("status")
    priority = context.inputs.get("priority")
    tags = context.inputs.get("tags")

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
