# Parlant AI - POC/Pilot

## **Objective**

The Parlant AI pilot’s primary goal is to serve as an intelligent bridge between customer interactions and internal business logic. It will use data from our customer-facing knowledge base, existing IVR, and the ParkWhiz API as input, then consult a rules engine (managed via spreadsheet) to determine outcomes for refund scenarios and provide answers to basic customer questions.

The most critical function of the AI is to determine whether or not a customer is eligible for a refund, and to process it according to defined business rules. These decisions may be non-linear—certain steps can be revisited as new information is gathered. No credit card data will be handled directly; all refund transactions occur through the ParkWhiz API.

A successful pilot will:

1. Provide guidance based on our existing [Knowledge Base and IVR](#knowledge-base-and-ivr-non-refund-rules) response tree.
2. Support [user authentication](#user-authentication) and validation.
3. Use a [spreadsheet-driven rule engine](#simple-refund-rules) to evaluate refund eligibility, covering:
   - Cancellations within the pre-arrival window handled automatically through the [ParkWhiz API](#parkwhiz-api).
   - Third-party bookings, routed to the correct partner channels.
   - On-demand (non-prebooked) reservations, flagged as non-refundable.
   - Confirmed rebookings where duplicate passes exist, refunded through the API.
   - Oversold locations, identified via the admin dashboard and refunded with proper justification.
4. Perform a [warm handoff](#agent-handoff) to live support when customer frustration or uncertainty is detected.
5. Direct callers to the online refund form ([https://go.parkwhiz.com/refund](https://go.parkwhiz.com/refund)) if their scenario cannot be handled automatically.
6. Include a [live analytics dashboard](#reporting) showing conversation logs, recordings, and sentiment trends connected to Looker for BI analysis.
7. Provide a [QA interface](#qa) for manual review and feedback loops to refine model accuracy.
8. Demonstrate measurable case deflection improvements over the current IVR after 45 days of limited production testing.
9. Include a configurable **kill switch** and refund cap logic (via spreadsheet) that halts refund processing after reaching the daily limit.

---

## **Data Sources**

Responses are derived from the knowledge base and current IVR logic.  
The major improvement is automating a subset of refund requests—freeing human agents to focus on complex, high-context cases.

### Knowledge Base and IVR (Non-Refund Rules)

The AI draws answers from the existing knowledge base: [https://help.parkwhiz.com](https://help.parkwhiz.com)

Reference flowchart: [Dialpad IVR Flow](https://lucid.app/lucidchart/5193b55c-5fb0-4e77-bd3d-4bc53591b51f/edit?viewport_loc=-1584%2C-1740%2C6652%2C3084%2CCBES8AjAJ89Q&invitationId=inv_7598581d-5bbd-422f-ac08-ce62dca6dd32)

### Refund Rules Spreadsheet

The AI references a spreadsheet similar to the [Refund Scenario Decision Chart](https://docs.google.com/spreadsheets/d/1DmqS9Ad54z_9mJNbHc4y3-WWvT09WJ44sA7YKwFMib0/edit?usp=sharing), allowing dynamic updates to refund logic without code changes.

### ParkWhiz API

The system integrates with the ParkWhiz API for actions like account lookups, bookings, and refund processing.

Key documentation:

- **Accounts:** [https://developer.parkwhiz.com/v4_internal/#accounts](https://developer.parkwhiz.com/v4_internal/#accounts)
- **Bookings:** [https://developer.parkwhiz.com/v4_internal/#bookings](https://developer.parkwhiz.com/v4_internal/#bookings)

See [Example API Call](#example-api-call) for implementation details.

---

## **User Authentication**

Best practices for authentication should be explored (e.g., OTP via text or email).  
Agents typically use a Parking Pass ID, email, or phone number to locate accounts, cross-referencing with location/time/event data for validation.

---

## **Cancellations, Refunds, & Account Credits**

### Cancellation vs. Refund Definition

A **cancellation** occurs before the parking start time—triggering an automatic refund.  
A **refund** occurs post-event, requiring validation by support or AI logic.

### Refund vs. Account Credit Definition

When possible, **account credits** are preferred to refunds, particularly in borderline cases, to encourage customer retention. Credits cannot stack with refunds.

### Complete Rules Guide

Refer to the [Refund and Credits Guide](https://drive.google.com/file/d/1SNx3Ob5aWzlH7x-jZg3XKORU-l_COTy9/view?usp=sharing) for detailed business logic governing refunds and credits.

---

## **Simple Refund Rules**

These straightforward cases form the foundation of the POC/Pilot scope.  
The spreadsheet ([Refund Scenario Decision Chart](https://docs.google.com/spreadsheets/d/1DmqS9Ad54z_9mJNbHc4y3-WWvT09WJ44sA7YKwFMib0/edit?usp=sharing)) governs refund actions.  
Partial refunds remain out-of-scope and require manual review.

---

## **Complex Refund Rules**

Complex refund cases are excluded from this POC phase but can be revisited during post-pilot expansion.

---

## **Agent Handoff**

The AI initiates a **warm handoff** when escalation is needed.

- Creates a Freshdesk ticket summarizing the call and booking details.
- Routes the call to the Dialpad agent queue with custom “whisper” context (e.g., “Customer requesting refund”).
- Upon agent connection, a browser extension (or API integration) opens the related Freshdesk ticket automatically.

---

## **Reporting and Validation**

### Reporting

The pilot includes a live monitoring dashboard showing key interaction metrics: sentiment, case deflection, transcripts, and trend analysis in Looker.

### QA

A dedicated QA interface enables review of AI decisions, flagging errors, and corrective feedback for continuous improvement—mirroring human agent QA workflows.

---

## **References**

- [Dialpad IVR Flowchart](https://lucid.app/lucidchart/5193b55c-5fb0-4e77-bd3d-4bc53591b51f/edit?viewport_loc=-1584%2C-1740%2C6652%2C3084%2CCBES8AjAJ89Q&invitationId=inv_7598581d-5bbd-422f-ac08-ce62dca6dd32)
- [Customer Knowledge Base](https://help.parkwhiz.com)
- [Refund & Credits Guide](https://drive.google.com/file/d/1SNx3Ob5aWzlH7x-jZg3XKORU-l_COTy9/view?usp=sharing)
- [Refund Decision Chart](https://docs.google.com/spreadsheets/d/1DmqS9Ad54z_9mJNbHc4y3-WWvT09WJ44sA7YKwFMib0/edit?gid=1977437019#gid=1977437019)
- [Refund Form](https://go.parkwhiz.com/refund)
- [ParkWhiz API Docs](https://developer.parkwhiz.com/v4_internal)

---

## **Example API Call**

**Request:**

```bash
curl --location 'https://api-sandbox.parkwhiz.com/v4/bookings/88236950?zoom=pw%3Apartner,pw:parking_pass,pw:customer' \
--header 'Authorization: Bearer <API_KEY>'
```

**Response (truncated):**

```json
{
  "id": 88236950,
  "customer_id": 737314,
  "start_time": "2025-03-06T12:30:00.000-06:00",
  "end_time": "2025-03-06T22:30:00.000-06:00",
  "price_paid": { "USD": "17.22" },
  "full_price": { "USD": "26.5" },
  "type": "transient_booking",
  "on_demand": false,
  "cancellable": false,
  "_embedded": {
    "pw:parking_pass": { "status": "active" },
    "pw:location": { "name": "ABM Parking Services", "city": "Chicago" },
    "pw:customer": { "email": "user@example.com" }
  }
}
```

```

---

Would you like me to version this for **multi-agent collaboration** (e.g., separate "AI Core," "Rule Engine," and "QA/Reporting" sections) so your Parlant implementation can evolve modularly?
```
