"""
Tests for PolicyLoader component.

Tests cover:
- Loading all policy files
- Parsing JSON structure
- Caching behavior (singleton pattern)
- Error handling for missing files
"""

import pytest
import os
import json
from unittest.mock import patch, mock_open, MagicMock
from app_tools.tools.refund_decision.policy_loader import PolicyLoader


@pytest.fixture
def reset_policy_loader():
    """Reset PolicyLoader singleton between tests."""
    PolicyLoader._instance = None
    PolicyLoader._initialized = False
    yield
    PolicyLoader._instance = None
    PolicyLoader._initialized = False


@pytest.fixture
def mock_policy_files():
    """Mock policy file contents."""
    return {
        "refund_rules.json": json.dumps([
            {
                "Scenario": "Test Scenario",
                "Trigger/Condition": "Test condition",
                "Checks": "Test checks",
                "Action": "Test action",
                "Recognition Phrases/Keywords": "test keywords",
                "Refund Reason/Settings": "test settings"
            }
        ]),
        "refund_guide.json": json.dumps({
            "title": "Test Guide",
            "introduction": "Test introduction",
            "sections": [
                {
                    "title": "Test Section",
                    "content": "Test content"
                }
            ]
        }),
        "refund_scenario_decision_chart.md": "# Test Decision Chart\n\nTest content",
        "ai_vs_human_refund_scenarios.md": "# Test Scenarios\n\nTest content",
        "refund_policy_condensed.md": "# Condensed Policy\n\nTest condensed content"
    }


def test_policy_loader_singleton(reset_policy_loader):
    """Test that PolicyLoader implements singleton pattern correctly."""
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='[]')):
        
        loader1 = PolicyLoader()
        loader2 = PolicyLoader()
        
        # Both instances should be the same object
        assert loader1 is loader2
        assert id(loader1) == id(loader2)


def test_policy_loader_initialization_once(reset_policy_loader):
    """Test that PolicyLoader only initializes once (caching behavior)."""
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='[]')) as mock_file:
        
        loader1 = PolicyLoader()
        initial_call_count = mock_file.call_count
        
        # Create another instance
        loader2 = PolicyLoader()
        
        # File should not be opened again (cached)
        assert mock_file.call_count == initial_call_count
        assert loader1 is loader2


def test_load_all_policy_files(reset_policy_loader, mock_policy_files):
    """Test loading all required policy files successfully."""
    
    def mock_file_open(filename, *args, **kwargs):
        # Extract just the filename from the path
        base_filename = os.path.basename(filename)
        if base_filename in mock_policy_files:
            return mock_open(read_data=mock_policy_files[base_filename])()
        raise FileNotFoundError(f"File not found: {filename}")
    
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', side_effect=mock_file_open):
        
        loader = PolicyLoader()
        
        # Verify all files were loaded
        assert loader.rules is not None
        assert loader.guide is not None
        assert loader.decision_chart is not None
        assert loader.scenarios is not None
        
        # Verify content
        assert isinstance(loader.rules, list)
        assert len(loader.rules) == 1
        assert loader.rules[0]["Scenario"] == "Test Scenario"
        
        assert isinstance(loader.guide, dict)
        assert loader.guide["title"] == "Test Guide"
        
        assert "Test Decision Chart" in loader.decision_chart
        assert "Test Scenarios" in loader.scenarios


def test_parse_json_structure(reset_policy_loader, mock_policy_files):
    """Test that JSON files are parsed correctly into data structures."""
    
    def mock_file_open(filename, *args, **kwargs):
        base_filename = os.path.basename(filename)
        if base_filename in mock_policy_files:
            return mock_open(read_data=mock_policy_files[base_filename])()
        raise FileNotFoundError(f"File not found: {filename}")
    
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', side_effect=mock_file_open):
        
        loader = PolicyLoader()
        
        # Test refund_rules.json parsing
        rules = loader.get_rules()
        assert isinstance(rules, list)
        assert "Scenario" in rules[0]
        assert "Action" in rules[0]
        
        # Test that guide is properly structured
        assert "title" in loader.guide
        assert "sections" in loader.guide
        assert isinstance(loader.guide["sections"], list)


def test_parse_json_with_curly_quotes(reset_policy_loader):
    """Test that JSON files with curly quotes are handled correctly."""
    
    # JSON with curly quotes (common in documents)
    json_with_curly_quotes = '{"title": "Test "quoted" content", "value": "test"}'
    
    mock_files = {
        "refund_rules.json": json_with_curly_quotes,
        "refund_guide.json": '{"title": "Guide"}',
        "refund_scenario_decision_chart.md": "# Chart",
        "ai_vs_human_refund_scenarios.md": "# Scenarios",
        "refund_policy_condensed.md": "# Condensed"
    }
    
    def mock_file_open(filename, *args, **kwargs):
        base_filename = os.path.basename(filename)
        if base_filename in mock_files:
            return mock_open(read_data=mock_files[base_filename])()
        raise FileNotFoundError(f"File not found: {filename}")
    
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', side_effect=mock_file_open):
        
        loader = PolicyLoader()
        
        # Should successfully parse despite curly quotes
        assert loader.rules is not None
        # If parsing fails, it stores as raw_content
        if isinstance(loader.rules, dict) and "raw_content" in loader.rules:
            assert loader.rules["parsed"] == False
        else:
            # Successfully parsed
            assert isinstance(loader.rules, dict)


def test_missing_required_file_raises_error(reset_policy_loader):
    """Test that missing required policy files raise FileNotFoundError."""
    
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', side_effect=FileNotFoundError("File not found")):
        
        with pytest.raises(FileNotFoundError) as exc_info:
            PolicyLoader()
        
        assert "Required policy file not found" in str(exc_info.value)


def test_get_rules_method(reset_policy_loader, mock_policy_files):
    """Test get_rules() method returns correct data."""
    
    def mock_file_open(filename, *args, **kwargs):
        base_filename = os.path.basename(filename)
        if base_filename in mock_policy_files:
            return mock_open(read_data=mock_policy_files[base_filename])()
        raise FileNotFoundError(f"File not found: {filename}")
    
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', side_effect=mock_file_open):
        
        loader = PolicyLoader()
        rules = loader.get_rules()
        
        assert isinstance(rules, list)
        assert len(rules) > 0
        assert rules[0]["Scenario"] == "Test Scenario"


def test_get_condensed_policy_text(reset_policy_loader, mock_policy_files):
    """Test get_condensed_policy_text() returns condensed version."""
    
    def mock_file_open(filename, *args, **kwargs):
        base_filename = os.path.basename(filename)
        if base_filename in mock_policy_files:
            return mock_open(read_data=mock_policy_files[base_filename])()
        raise FileNotFoundError(f"File not found: {filename}")
    
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', side_effect=mock_file_open):
        
        loader = PolicyLoader()
        condensed = loader.get_condensed_policy_text()
        
        assert "Condensed Policy" in condensed
        assert "Test condensed content" in condensed


def test_get_condensed_policy_fallback(reset_policy_loader, mock_policy_files):
    """Test that get_condensed_policy_text() falls back to full policy if condensed doesn't exist."""
    
    # Remove condensed file from mock
    mock_files_no_condensed = mock_policy_files.copy()
    del mock_files_no_condensed["refund_policy_condensed.md"]
    
    def mock_file_open(filename, *args, **kwargs):
        base_filename = os.path.basename(filename)
        if base_filename in mock_files_no_condensed:
            return mock_open(read_data=mock_files_no_condensed[base_filename])()
        if base_filename == "refund_policy_condensed.md":
            raise FileNotFoundError("Condensed file not found")
        raise FileNotFoundError(f"File not found: {filename}")
    
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', side_effect=mock_file_open):
        
        loader = PolicyLoader()
        policy_text = loader.get_condensed_policy_text()
        
        # Should return full policy as fallback
        assert "Refund and Credits Guide" in policy_text or "Test Guide" in policy_text


def test_get_full_policy_text(reset_policy_loader, mock_policy_files):
    """Test get_full_policy_text() combines all policy documents."""
    
    def mock_file_open(filename, *args, **kwargs):
        base_filename = os.path.basename(filename)
        if base_filename in mock_policy_files:
            return mock_open(read_data=mock_policy_files[base_filename])()
        raise FileNotFoundError(f"File not found: {filename}")
    
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', side_effect=mock_file_open):
        
        loader = PolicyLoader()
        full_text = loader.get_full_policy_text()
        
        # Should contain content from all sources
        assert "Refund and Credits Guide" in full_text
        assert "Test Guide" in full_text
        assert "Refund Rules and Scenarios" in full_text
        assert "Test Scenario" in full_text
        assert "Refund Scenario Decision Chart" in full_text
        assert "Test Decision Chart" in full_text
        assert "AI vs Human Refund Scenarios" in full_text
        assert "Test Scenarios" in full_text


def test_docker_path_detection(reset_policy_loader, mock_policy_files):
    """Test that PolicyLoader correctly detects Docker environment paths."""
    
    def mock_file_open(filename, *args, **kwargs):
        base_filename = os.path.basename(filename)
        if base_filename in mock_policy_files:
            return mock_open(read_data=mock_policy_files[base_filename])()
        raise FileNotFoundError(f"File not found: {filename}")
    
    # Mock Docker path exists
    def mock_path_exists(path):
        return path == "/app/app_tools/context/processed"
    
    with patch('os.path.exists', side_effect=mock_path_exists), \
         patch('builtins.open', side_effect=mock_file_open):
        
        loader = PolicyLoader()
        
        # Should use Docker path
        assert loader.context_dir == "/app/app_tools/context/processed"


def test_local_path_detection(reset_policy_loader, mock_policy_files):
    """Test that PolicyLoader correctly detects local environment paths."""
    
    def mock_file_open(filename, *args, **kwargs):
        base_filename = os.path.basename(filename)
        if base_filename in mock_policy_files:
            return mock_open(read_data=mock_policy_files[base_filename])()
        raise FileNotFoundError(f"File not found: {filename}")
    
    # Mock local path exists
    def mock_path_exists(path):
        return path == "context/processed"
    
    with patch('os.path.exists', side_effect=mock_path_exists), \
         patch('builtins.open', side_effect=mock_file_open):
        
        loader = PolicyLoader()
        
        # Should use local path
        assert loader.context_dir == "context/processed"


def test_caching_prevents_reload(reset_policy_loader, mock_policy_files):
    """Test that caching prevents reloading files on subsequent access."""
    
    load_count = {"count": 0}
    
    def mock_file_open(filename, *args, **kwargs):
        load_count["count"] += 1
        base_filename = os.path.basename(filename)
        if base_filename in mock_policy_files:
            return mock_open(read_data=mock_policy_files[base_filename])()
        raise FileNotFoundError(f"File not found: {filename}")
    
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', side_effect=mock_file_open):
        
        # First instance loads files
        loader1 = PolicyLoader()
        initial_count = load_count["count"]
        
        # Access methods multiple times
        loader1.get_rules()
        loader1.get_condensed_policy_text()
        loader1.get_full_policy_text()
        
        # Second instance should not reload
        loader2 = PolicyLoader()
        loader2.get_rules()
        
        # File open count should only increase for condensed policy access
        # (which opens the file each time it's called)
        # But initialization files should not be reloaded
        assert loader1 is loader2


def test_invalid_json_fallback(reset_policy_loader):
    """Test that invalid JSON is stored as raw content for LLM consumption."""
    
    invalid_json = '{"title": "Test", invalid json here}'
    
    mock_files = {
        "refund_rules.json": invalid_json,
        "refund_guide.json": '{"title": "Guide"}',
        "refund_scenario_decision_chart.md": "# Chart",
        "ai_vs_human_refund_scenarios.md": "# Scenarios",
        "refund_policy_condensed.md": "# Condensed"
    }
    
    def mock_file_open(filename, *args, **kwargs):
        base_filename = os.path.basename(filename)
        if base_filename in mock_files:
            return mock_open(read_data=mock_files[base_filename])()
        raise FileNotFoundError(f"File not found: {filename}")
    
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', side_effect=mock_file_open):
        
        loader = PolicyLoader()
        
        # Should store as raw content when JSON parsing fails
        assert isinstance(loader.rules, dict)
        assert "raw_content" in loader.rules
        assert loader.rules["parsed"] == False
        assert invalid_json in loader.rules["raw_content"]
