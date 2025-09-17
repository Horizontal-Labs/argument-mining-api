def normalize_session_id(session_id: str | None) -> str | None:
    """
    Normalize a provided session_id.

    Returns None when the client omitted the value or sent only whitespace.
    """
    if not session_id:
        return None

    stripped = session_id.strip()
    return stripped or None

