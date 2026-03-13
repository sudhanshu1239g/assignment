from backend.app.ingest import _normalize_text


def test_normalize_text_trims_and_truncates():
    text = "  hello " + ("a" * 10050)
    cleaned = _normalize_text(text)
    assert cleaned.startswith("hello")
    assert len(cleaned) == 10000
