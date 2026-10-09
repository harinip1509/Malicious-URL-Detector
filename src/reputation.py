"""Domain reputation: who actually owns the host.

Lexical features alone cannot tell a long, legitimate Google sign-in URL from a
phishing page, because both are long, full of encoded characters and contain
words like "signin". What separates them is the registered domain. This module
supplies that one missing fact, in two directions:

  trusted     the host belongs to a well-known registered domain
              (youtube.com, m.youtube.com and accounts.google.com all qualify;
              youtube.com.evil.xyz does not)
  impersonates  the host mentions a well-known brand, including digit swaps like
              g00gle or amaz0n, on a domain that brand does not own

Matching is on the registered domain, never on a substring, which is exactly the
trick lookalike domains rely on.

Limitation worth stating: trusting a domain also trusts its open redirects, so a
link like youtube.com/redirect?q=<evil> is let through. Real systems follow the
redirect; this prototype does not fetch anything.
"""

from urllib.parse import urlparse

TRUSTED_DOMAINS = {
    # video, search and mail
    "youtube.com", "youtu.be", "youtube-nocookie.com", "google.com", "google.co.in",
    "googleusercontent.com", "gstatic.com", "gmail.com", "bing.com", "yahoo.com",
    # productivity and cloud
    "microsoft.com", "microsoftonline.com", "live.com", "office.com", "outlook.com",
    "apple.com", "icloud.com", "zoom.us", "dropbox.com", "notion.so",
    # social
    "facebook.com", "instagram.com", "whatsapp.com", "twitter.com", "x.com",
    "linkedin.com", "reddit.com", "pinterest.com", "quora.com",
    # shopping, payments, streaming
    "amazon.com", "amazon.in", "flipkart.com", "paypal.com", "netflix.com",
    "spotify.com", "hotstar.com", "myntra.com", "paytm.com", "phonepe.com",
    # developer and reference
    "github.com", "gitlab.com", "stackoverflow.com", "wikipedia.org", "python.org",
    "kaggle.com", "huggingface.co", "colab.research.google.com", "openai.com",
    "anthropic.com", "claude.ai", "medium.com",
    # Indian banks and government, common phishing targets
    "sbi.co.in", "onlinesbi.sbi", "hdfcbank.com", "icicibank.com", "axisbank.com",
    "gov.in", "nic.in", "uidai.gov.in", "incometax.gov.in",
}

# Brand name -> the registered domains that brand genuinely uses.
BRANDS = {
    "youtube": {"youtube.com", "youtu.be", "youtube-nocookie.com"},
    "google": {"google.com", "google.co.in", "googleusercontent.com", "gstatic.com"},
    "gmail": {"gmail.com", "google.com"},
    "microsoft": {"microsoft.com", "microsoftonline.com", "live.com", "office.com"},
    "outlook": {"outlook.com", "live.com", "office.com"},
    "apple": {"apple.com", "icloud.com"},
    "icloud": {"icloud.com", "apple.com"},
    "facebook": {"facebook.com"},
    "instagram": {"instagram.com"},
    "whatsapp": {"whatsapp.com"},
    "linkedin": {"linkedin.com"},
    "amazon": {"amazon.com", "amazon.in"},
    "flipkart": {"flipkart.com"},
    "paypal": {"paypal.com"},
    "netflix": {"netflix.com"},
    "spotify": {"spotify.com"},
    "paytm": {"paytm.com"},
    "phonepe": {"phonepe.com"},
    "github": {"github.com"},
    "hdfc": {"hdfcbank.com"},
    "icici": {"icicibank.com"},
    "sbi": {"sbi.co.in", "onlinesbi.sbi"},
    "bankofamerica": {"bankofamerica.com"},
    "wellsfargo": {"wellsfargo.com"},
    "chase": {"chase.com"},
}

# Digit-for-letter swaps used in lookalike domains: g00gle, amaz0n, paypa1.
_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a"})


def host_of(url: str) -> str:
    url = (url or "").strip()
    try:
        return (urlparse(url if "://" in url else "http://" + url).hostname or "").lower()
    except ValueError:
        return ""


def _owns(host: str, domains) -> str:
    """Return the registered domain in `domains` that owns `host`, or ''."""
    for d in domains:
        if host == d or host.endswith("." + d):
            return d
    return ""


def trusted_domain(url: str) -> str:
    """The trusted registered domain this URL belongs to, or '' if none."""
    return _owns(host_of(url), TRUSTED_DOMAINS)


def impersonated_brand(url: str) -> str:
    """A brand named in the host that the host does not belong to, or ''.

    Short brand names are matched only as a whole hostname label, so that
    'sbi' does not fire on every host that happens to contain those letters.
    """
    host = host_of(url)
    if not host or trusted_domain(url):
        return ""
    normal = host.translate(_LEET)
    labels = set(normal.replace("-", ".").split("."))
    for brand, owned in BRANDS.items():
        hit = brand in labels if len(brand) <= 5 else brand in normal
        if hit and not _owns(host, owned):
            return brand
    return ""


def reputation_signals(url: str) -> list:
    out = []
    d = trusted_domain(url)
    if d:
        out.append(f"Registered domain {d} is on the trusted-domain list")
    b = impersonated_brand(url)
    if b:
        out.append(f"Uses the brand name \u201c{b}\u201d on a domain that brand does not own")
    return out
