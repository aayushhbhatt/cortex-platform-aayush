from __future__ import annotations

TIER_PERMISSIONS = {
    "standard": ["public", "general"],
    "manager": ["public", "general", "manager"],
    "exec": ["public", "general", "manager", "executive"],
}


def normalize_user_tier(user_tier: str | None) -> str:
    if user_tier is None:
        return "standard"

    normalized = user_tier.strip().lower()
    if normalized not in TIER_PERMISSIONS:
        return "standard"
    return normalized


def allowed_access_levels(user_tier: str | None) -> list[str]:
    tier = normalize_user_tier(user_tier)
    return TIER_PERMISSIONS[tier]


def filter_by_tier(chunks: list[dict], user_tier: str | None) -> list[dict]:
    allowed_levels = set(allowed_access_levels(user_tier))
    allowed_chunks: list[dict] = []
    for chunk in chunks:
        access_level = chunk.get("access_level", "public")
        if access_level in allowed_levels:
            allowed_chunks.append(chunk)
    return allowed_chunks


def explain_access_filtering(total_chunks: int, allowed_chunks: int, user_tier: str | None) -> dict:
    tier = normalize_user_tier(user_tier)
    return {
        "user_tier": tier,
        "allowed_access_levels": TIER_PERMISSIONS[tier],
        "total_chunks_before_filter": total_chunks,
        "total_chunks_after_filter": allowed_chunks,
        "filtered_out_count": total_chunks - allowed_chunks,
    }
