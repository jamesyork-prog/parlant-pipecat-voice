import pytest
from unittest.mock import Mock
from app_tools.tools.ticket_processing.helpers import document_decision
import parlant.sdk as p

@pytest.fixture
def mock_context():
    context = Mock(spec=p.ToolContext)
    context.inputs = {}
    return context

@pytest.mark.asyncio
async def test_document_decision_missing_ticket_id(mock_context):
    mock_context.inputs = {"decision_result": {}}
    result = await document_decision(mock_context)
    assert result.data["documented"] is False
    assert "No ticket_id provided" in result.data["error"]
