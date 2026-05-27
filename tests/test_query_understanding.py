import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.query_understanding import detect_query_type, extract_filters, normalize_query, understand_query


def test_understand_query_empty() -> None:
    understanding = understand_query("   ")
    assert understanding.normalized_query == ""
    assert understanding.expanded_queries == []
    assert understanding.query_type == "unknown"
    assert understanding.filters == {}


def test_detect_query_type_parental_leave() -> None:
    assert detect_query_type(normalize_query("How much parental leave do I get?")) == "hr_policy"


def test_detect_query_type_benefits() -> None:
    assert detect_query_type(normalize_query("What health insurance benefits are available?")) == "benefits"


def test_detect_query_type_security() -> None:
    assert detect_query_type(normalize_query("What are the VPN and encryption requirements?")) == "security"


def test_detect_query_type_software_request() -> None:
    assert detect_query_type(normalize_query("How do I request approved software?")) == "software_request"


def test_detect_query_type_performance() -> None:
    assert detect_query_type(normalize_query("How does performance calibration work?")) == "performance"


def test_detect_query_type_remote_work() -> None:
    assert detect_query_type(normalize_query("How many remote work days are allowed?")) == "remote_work"


def test_detect_query_type_executive_strategy() -> None:
    assert detect_query_type(normalize_query("What is the FY2025 IPO strategy?")) == "executive_strategy"


def test_extract_doc_id_hint_parental_leave() -> None:
    filters = extract_filters(normalize_query("How do I apply for parental leave?"))
    assert filters["doc_id_hint"] == "parental_leave_policy"


def test_extract_filters_security_category() -> None:
    filters = extract_filters(normalize_query("What are the data security guidelines?"))
    assert filters["category"] == "technical_policy"
    assert filters["doc_id_hint"] == "data_security_guidelines"


def test_understand_query_includes_expanded_queries() -> None:
    understanding = understand_query("leave policy")
    assert understanding.normalized_query in understanding.expanded_queries
    assert any(term in understanding.expanded_queries for term in ["vacation", "time off", "parental leave"])
