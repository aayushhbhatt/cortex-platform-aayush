from __future__ import annotations

from dataclasses import dataclass, field

SYNONYM_MAP = {
    "leave": ["vacation", "time off", "parental leave"],
    "policy": ["handbook", "guideline"],
    "expense": ["reimbursement"],
    "remote": ["work from home", "hybrid work"],
    "ticket": ["support request"],
    "escalation": ["incident", "support path"],
    "access": ["permission"],
    "security": ["data security", "acceptable use"],
    "software": ["software request", "it portal"],
    "performance": ["review", "calibration", "rating"],
    "compensation": ["benefits", "merit increase"],
    "strategy": ["executive strategy", "board", "ipo"],
}


@dataclass
class QueryUnderstanding:
    original_query: str
    normalized_query: str
    expanded_queries: list[str]
    query_type: str
    filters: dict = field(default_factory=dict)


def normalize_query(query: str) -> str:
    normalized = " ".join(query.lower().strip().split())
    return normalized


def expand_query(query: str) -> list[str]:
    normalized = normalize_query(query)
    if not normalized:
        return []

    expanded = [normalized]
    for term, synonyms in SYNONYM_MAP.items():
        if term in normalized:
            expanded.extend(synonyms)
            expanded.extend(f"{normalized} {syn}" for syn in synonyms)

    deduped: list[str] = []
    seen = set()
    for item in expanded:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def _contains_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def detect_query_type(normalized_query: str) -> str:
    if _contains_any(normalized_query, ["parental leave", "sick leave", "pto", "vacation", " leave ", " leave", "leave "]):
        return "hr_policy"
    if _contains_any(normalized_query, ["compensation", "benefits", "401k", "wellness", "insurance"]):
        return "benefits"
    if _contains_any(normalized_query, ["performance", "review", "rating", "calibration", "pip"]):
        return "performance"
    if _contains_any(normalized_query, ["remote", "hybrid", "work from home", "office days"]):
        return "remote_work"
    if _contains_any(normalized_query, ["security", "data security", "encryption", "vpn", "incident", "breach"]):
        return "security"
    if _contains_any(normalized_query, ["software", "approved software", "it portal", "license"]):
        return "software_request"
    if _contains_any(normalized_query, ["strategy", "ipo", "board", "arr", "apac", "operating margin", "executive"]):
        return "executive_strategy"
    if _contains_any(normalized_query, ["company", "overview", "mission", "leadership", "product lines"]):
        return "company_overview"
    return "unknown"


def extract_filters(normalized_query: str) -> dict:
    filters: dict[str, str] = {}

    if _contains_any(normalized_query, ["leave", "performance", "remote", "benefits", "hr", "compensation"]):
        filters["category"] = "hr_policy"
    elif _contains_any(normalized_query, ["security", "software", "access", "vpn", "encryption"]):
        filters["category"] = "technical_policy"
    elif _contains_any(normalized_query, ["strategy", "ipo", "board", "arr", "apac"]):
        filters["category"] = "executive_strategy"
    elif _contains_any(normalized_query, ["company", "overview", "mission", "leadership"]):
        filters["category"] = "general"

    if _contains_any(normalized_query, ["executive", "board", "ipo", "restricted", "strategy"]):
        filters["access_level_hint"] = "executive"
    elif _contains_any(normalized_query, ["manager handbook", "compensation conversation", "pip handling"]):
        filters["access_level_hint"] = "manager"

    doc_hints = [
        ("compensation and benefits", "compensation_and_benefits"),
        ("parental leave", "parental_leave_policy"),
        ("remote work", "remote_work_policy"),
        ("data security", "data_security_guidelines"),
        ("acceptable use", "acceptable_use_policy"),
        ("software request", "software_request_process"),
        ("performance review", "performance_review_process"),
        ("company overview", "company_overview"),
        ("manager handbook", "manager_handbook"),
        ("executive strategy", "executive_strategy_fy2025"),
    ]
    for phrase, doc_id in doc_hints:
        if phrase in normalized_query:
            filters["doc_id_hint"] = doc_id
            break

    if "how to apply" in normalized_query:
        filters["section_hint"] = "how to apply"
    elif "apply" in normalized_query:
        filters["section_hint"] = "apply"
    elif "request" in normalized_query:
        filters["section_hint"] = "request"
    elif "eligibility" in normalized_query:
        filters["section_hint"] = "eligibility"
    elif "benefits continuation" in normalized_query:
        filters["section_hint"] = "benefits continuation"
    elif "security requirements" in normalized_query:
        filters["section_hint"] = "security requirements"

    return filters


def understand_query(query: str) -> QueryUnderstanding:
    if not query or not query.strip():
        return QueryUnderstanding(
            original_query=query,
            normalized_query="",
            expanded_queries=[],
            query_type="unknown",
            filters={},
        )

    normalized_query = normalize_query(query)
    return QueryUnderstanding(
        original_query=query,
        normalized_query=normalized_query,
        expanded_queries=expand_query(query),
        query_type=detect_query_type(normalized_query),
        filters=extract_filters(normalized_query),
    )
