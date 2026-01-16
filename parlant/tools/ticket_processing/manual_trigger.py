import parlant.sdk as p


@p.tool
async def trigger_ticket_processing(context: p.ToolContext) -> p.ToolResult:
    """
    Manually triggers ticket processing for a specific Freshdesk ticket.
    Use this to test the refund journey with real ticket data.

    Args:
        ticket_id (str): The Freshdesk ticket ID to process (e.g., "1234567")
    """
    ticket_id = context.inputs.get("ticket_id")
    
    if not ticket_id:
        return p.ToolResult(
            {"error": "ticket_id is required"},
            metadata={"summary": "Error: No ticket_id provided"}
        )
    
    return p.ToolResult(
        {
            "ticket_id": ticket_id,
            "status": "triggered",
            "message": f"Processing ticket {ticket_id}"
        },
        metadata={"summary": f"Triggered processing for ticket {ticket_id}"}
    )
