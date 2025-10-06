# main.py

import asyncio, os, pathlib
import parlant.sdk as p
import psycopg2
from typing import Annotated


ROOT = pathlib.Path(__file__).resolve().parents[1]
AGENT_ID_FILE = pathlib.Path("/app/data/agent_id.txt")

#async def get_db_connection(db_name, db_user, db_password, db_host, db_port):
async def get_db_connection():
    """
    Establishes and returns a connection to a PostgreSQL database.

    Args:
        db_name (str): The name of the database.
        db_user (str): The username for the database.
        db_password (str): The password for the user.
        db_host (str): The host address of the database server.
        db_port (str): The port number for the database connection.

    Returns:
        psycopg2.connection: A connection object to the database, or None if the connection fails.
    """
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            database=os.getenv("POSTGRES_DB", "ranjayDB"),
            user=os.getenv("POSTGRES_USER", "ranjay.kumar"),
            password=os.getenv("POSTGRES_PASSWORD", "ranjay"),
            port=os.getenv("POSTGRES_PORT", "5432")
        )
        print("Database connection successful.")
        return conn
    except psycopg2.OperationalError as e:
        print(f"Error: Unable to connect to the database. {e}")
        return None


@p.tool
async def find_products(context: p.ToolContext, query: str) -> p.ToolResult:
    """Fetch products based on a natural-language search query."""

    # Get the database connection
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        database=os.getenv("POSTGRES_DB", "ranjayDB"),
        user=os.getenv("POSTGRES_USER", "ranjay.kumar"),
        password=os.getenv("POSTGRES_PASSWORD", "ranjay"),
        port=os.getenv("POSTGRES_PORT", "5432")
    )
    print("Database connection successful.")

    # Create a cursor object
    cur = conn.cursor()

    # Execute a sample query
    #cur.execute("SELECT version();")
    #products = cur.fetchone()
    cur.execute("SELECT * FROM public.products ORDER BY product_id ASC LIMIT 2")
    products = cur.fetchall()
    print(f"PostgreSQL database version: {products}")

    # Close the cursor and connection
    cur.close()
    conn.close()

    return p.ToolResult(products)

# -----------------------------
# Tools (mock business logic)
# -----------------------------
@p.tool
async def check_refund_eligibility(
    context: p.ToolContext,
    order_number: Annotated[str, p.ToolParameterOptions(
        source="customer",
        description="Order number provided by the customer",
        significance="We need this to locate your purchase."
    )],
    refund_reason: Annotated[str, p.ToolParameterOptions(
        source="customer",
        description="Short description of why you want a refund",
        examples=["defective", "broken", "changed my mind"]
    )],
    days_since_purchase: Annotated[int | None, p.ToolParameterOptions(
        source="customer",
        description="Number of days since you purchased the item",
        examples=["7", "14", "45"]
    )] = None,
    unopened: Annotated[bool | None, p.ToolParameterOptions(
        source="customer",
        description="Whether the item is unopened",
        examples=["yes", "no"]
    )] = None,
) -> p.ToolResult:
    reason = (refund_reason or "").lower()
    dsp = days_since_purchase if isinstance(days_since_purchase, int) else 999
    is_unopened = bool(unopened)

    eligible = False
    policy = ""

    if any(k in reason for k in ("defect", "broken", "damage", "damaged")):
        eligible = True
        policy = "Defective items are always eligible for a refund."
    else:
        if dsp <= 30 and is_unopened:
            eligible = True
            policy = "Non-defective items are eligible if unopened within 30 days."
        else:
            policy = "Non-defective items must be unopened and returned within 30 days."

    return p.ToolResult(
        data={
            "order_number": order_number,
            "refund_reason": refund_reason,
            "days_since_purchase": dsp,
            "unopened": is_unopened,
            "eligible": eligible,
            "policy_explanation": policy,
        }
    )

@p.tool
async def process_refund(
    context: p.ToolContext,
    order_number: Annotated[str, p.ToolParameterOptions(
        source="context",
        description="Order number for which to process the refund"
    )],
) -> p.ToolResult:
    refund_id = f"RFD-{order_number}"
    return p.ToolResult(data={"refund_id": refund_id, "status": "processed"})


async def create_refund_journey(agent: p.Agent):
    """
    Refund journey using conditional transitions (no fork()),
    avoiding the JourneyTransition/transition_to mismatch.
    """
    journey = await agent.create_journey(
        title="Refund Request",
        conditions=["The customer is asking for a refund"],
        description="Guide the customer through the refund process with clear steps and policy clarity.",
    )

    # State 1: collect order number
    s1 = await journey.initial_state.transition_to(
        chat_state=(
            "Politely ask for the order number. "
            "Capture it as vars.order_number. If they don't have it, "
            "offer to look it up by email or phone."
        ),
        canned_responses=[
            await journey.create_canned_response(template="Sure—what’s your order number?"),
            await journey.create_canned_response(template="If you don’t have it handy, I can look it up by your email or phone."),
        ],
    )

    # State 2: collect reason + key facts
    s2 = await s1.target.transition_to(
        chat_state=(
            "Ask for the reason for the refund and capture as vars.refund_reason. "
            "If not defective, ask: Is the item unopened? Capture as vars.unopened (true/false). "
            "Ask how many days since purchase, capture as vars.days_since_purchase (integer). "
            "Confirm the facts with the customer before proceeding."
        )
    )

    # Conditional branches (no fork() used)
    s3_def = await s2.target.transition_to(
        chat_state=(
            "Acknowledge the defective issue empathetically. "
            "Reassure that we’ll check eligibility and help resolve quickly."
        ),
        condition="The item is defective or damaged",
    )

    s3_cxm = await s2.target.transition_to(
        chat_state=(
            "Explain that for change-of-mind refunds, our standard policy requires the item to be unopened "
            "and within 30 days. We'll check eligibility now."
        ),
        condition="The customer changed their mind or the item is not defective",
    )

    # Tool: check eligibility (both branches)
    s4a = await s3_def.target.transition_to(tool_state=check_refund_eligibility)
    s4b = await s3_cxm.target.transition_to(tool_state=check_refund_eligibility)

    # Merge after tool → common chat state to convey result
    s5 = await s4a.target.transition_to(
        chat_state=(
            "Review the result from check_refund_eligibility. "
            "If eligible, say so and outline next steps. If not, explain clearly using policy_explanation "
            "and offer alternatives like exchange or store credit."
        )
    )
    # Point the other branch to the same next state
    await s4b.target.transition_to(state=s5.target)

    # If eligible, process the refund
    s6 = await s5.target.transition_to(
        tool_state=process_refund,
        condition="The refund is eligible"
    )

    # Final notify state (chat state must follow a tool state)
    await s6.target.transition_to(
        chat_state=(
            "If processing succeeded, share the refund_id and expected timeline. "
            "If it failed, apologize and offer to escalate to a human."
        )
    )

    # Optional: journey-scoped escalation guideline
    @p.tool
    async def transfer_to_human_agent(context: p.ToolContext) -> p.ToolResult:
        return p.ToolResult.ok({"status": "queued_for_handoff"})

    await journey.create_guideline(
        condition="the customer is very upset or asks to speak to a manager",
        action="Offer to connect them with a human agent and trigger transfer_to_human_agent.",
        tools=[transfer_to_human_agent],
    )

async def main():
  async with p.Server() as server:
    agent = await server.create_agent(
        name="Refund Assistant",
        description="A helpful customer support agent that handles refunds politely and clearly."
    )

    agent_id = agent.id
    # Persist the id so other processes can read it (voice bridge)
    AGENT_ID_FILE.write_text(agent_id, encoding="utf-8")

    print("\n=== Parlant server is running ===")
    print("UI:   http://127.0.0.1:8800")
    print(f"AGENT_ID: {agent_id}")
    print(f"(Also saved to: {AGENT_ID_FILE})\n")    

    # Add the refund journey
    await create_refund_journey(agent)

    ##############################
    ##    Add the following:    ##
    ##############################
    await agent.create_guideline(
        condition="the customer greets you",
        action="Greet back warmly and ask how you can help today.",
    )


    CONDITION="the customer ask about products"
    await agent.create_guideline(
        # This is when the guideline will be triggered
        condition=CONDITION,
        # This is what the guideline instructs the agent to do
        action="Confirm the inputs with user before running the fetch call. and after confirmation provide the list of products as a list",
        tools=[find_products]
    )    
    

    await agent.attach_tool(condition=CONDITION, tool=find_products)

asyncio.run(main())