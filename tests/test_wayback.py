from star_history.wayback import parse_star_count


def test_parse_exact_accessible_label() -> None:
    html = '<a href="/org/repo/stargazers" aria-label="5,088 users starred this repository">'
    assert parse_star_count(html) == 5088


def test_parse_embedded_json_fallback() -> None:
    assert parse_star_count('<script>{"stargazerCount":7434}</script>') == 7434


def test_parse_missing_count() -> None:
    assert parse_star_count("<html>Sign in to GitHub</html>") is None
