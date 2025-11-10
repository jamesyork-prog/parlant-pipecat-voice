
import os
import httpx
import parlant.sdk as p


PARKWHIZ_API_KEY = os.environ.get("PARKWHIZ_API_KEY")
PARKWHIZ_API_URL = "https://api.parkwhiz.com/v4"


@p.tool
async def get_customer_orders(context: p.ToolContext) -> p.ToolResult:
    """
    Fetches parking reservation orders from ParkWhiz API for a specific customer.
    
    This tool retrieves booking information including booking IDs, amounts, dates,
    and locations. It can search by email, phone, or specific booking ID.
    
    Args:
        customer_email (str, optional): Customer's email address
        customer_phone (str, optional): Customer's phone number
        booking_id (str, optional): Specific booking/order ID to fetch
        start_date (str, optional): Filter orders from this date (YYYY-MM-DD)
        end_date (str, optional): Filter orders until this date (YYYY-MM-DD)
    
    Returns:
        ToolResult with:
        - orders (list): List of order objects with booking details
        - total (int): Total number of orders found
        - customer_identifier (str): The identifier used to search
    """
    customer_email = context.inputs.get("customer_email")
    customer_phone = context.inputs.get("customer_phone")
    booking_id = context.inputs.get("booking_id")
    start_date = context.inputs.get("start_date")
    end_date = context.inputs.get("end_date")
    
    # Check if API key is configured
    if not PARKWHIZ_API_KEY:
        return p.ToolResult(
            {"error": "ParkWhiz API key not configured"},
            metadata={"summary": "Error: ParkWhiz API key not configured"}
        )
    
    # Determine the search identifier
    identifier = customer_email or customer_phone or booking_id
    if not identifier:
        return p.ToolResult(
            {"error": "No customer identifier provided (email, phone, or booking_id required)"},
            metadata={"summary": "Error: No identifier provided"}
        )
    
    headers = {
        "Authorization": f"Bearer {PARKWHIZ_API_KEY}",
        "Content-Type": "application/json",
    }
    
    # Build query parameters
    params = {}
    if customer_email:
        params["email"] = customer_email
    if customer_phone:
        params["phone"] = customer_phone
    if start_date:
        params["start"] = start_date
    if end_date:
        params["end"] = end_date
    
    async with httpx.AsyncClient() as client:
        try:
            # If booking_id is provided, fetch specific order
            if booking_id:
                url = f"{PARKWHIZ_API_URL}/orders/{booking_id}"
                response = await client.get(url, headers=headers, timeout=10.0)
                response.raise_for_status()
                
                order_data = response.json()
                
                return p.ToolResult(
                    {
                        "order": order_data,
                        "booking_id": booking_id,
                    },
                    metadata={"summary": f"Found order {booking_id}"}
                )
            else:
                # Fetch orders list by email/phone
                url = f"{PARKWHIZ_API_URL}/orders/"
                response = await client.get(url, headers=headers, params=params, timeout=10.0)
                response.raise_for_status()
                
                data = response.json()
                orders = data.get("orders", [])
                total = data.get("total", len(orders))
                
                summary_msg = f"Found {total} orders for {identifier}" if total > 0 else f"No orders found for {identifier}"
                
                return p.ToolResult(
                    {
                        "orders": orders,
                        "total": total,
                        "customer_identifier": identifier,
                    },
                    metadata={"summary": summary_msg}
                )
                
        except httpx.HTTPStatusError as e:
            return p.ToolResult(
                {
                    "error": f"ParkWhiz API error: {e.response.status_code}",
                    "details": e.response.text,
                },
                metadata={"summary": "Error fetching orders"}
            )
        except httpx.RequestError as e:
            return p.ToolResult(
                {"error": f"An error occurred while fetching orders: {str(e)}"},
                metadata={"summary": "Error fetching orders"}
            )
