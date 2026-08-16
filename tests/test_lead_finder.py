from src.fetch.lead_finder import LeadFinder, IGNORED_DOMAINS


def test_domain_extraction():
    f = LeadFinder()
    assert f._domain("https://www.example.com/about") == "example.com"
    assert f._domain("https://Instagram.com/user") == "instagram.com"


def test_ignored_domains_filtered():
    f = LeadFinder()
    assert "instagram.com" in IGNORED_DOMAINS
    assert "google.com" in IGNORED_DOMAINS


def test_clean_search_url():
    f = LeadFinder()
    url = "https://duckduckgo.com/?uddg=https%3A%2F%2Fexample.com%2Fpage"
    assert f._clean_search_url(url) == "https://example.com/page"
    assert f._clean_search_url("https://example.com/direct") == "https://example.com/direct"


def test_clean_name():
    f = LeadFinder()
    assert f._clean_name("Fit Studio | Best Gym", "fitstudio.com") == "Fit Studio"
    assert f._clean_name("", "fitstudio.com") == "Fitstudio"


def test_extract_social_handles():
    f = LeadFinder()
    pages = [
        {"url": "https://fitstudio.in"},
        {"url": "https://www.instagram.com/fitstudio_official/"},
        {"url": "https://www.linkedin.com/company/fitstudio-india"},
    ]
    handles = f._extract_social_handles(pages, "https://fitstudio.in")
    assert handles["instagram"] == "fitstudio_official"
    assert handles["linkedin"] == "fitstudio-india"