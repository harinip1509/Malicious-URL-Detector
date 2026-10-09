"""Lexical and host-based features for a raw URL string.

Kept in one place because the API must compute features exactly the way training
did. If these two ever drift apart, the model still returns confident answers,
they are just wrong. Import this module from both sides, never copy it.
"""

import math
import re
from urllib.parse import urlparse

import numpy as np

SHORTENERS = {
    "bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "bitly.com", "cutt.ly", "rebrand.ly", "shorte.st", "rb.gy",
}

SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "account", "update", "secure", "banking",
    "confirm", "password", "webscr", "ebayisapi", "wp-admin", "admin",
    "free", "bonus", "click", "invoice", "payment",
]

IP_RE = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
HEX_IP_RE = re.compile(r"^0x[0-9a-f]+$", re.I)

FEATURE_NAMES = [
    "url_length", "hostname_length", "path_length", "query_length",
    "path_depth", "digit_count", "letter_count", "digit_letter_ratio",
    "dot_count", "hyphen_count", "underscore_count", "slash_count",
    "question_count", "equal_count", "at_count", "ampersand_count",
    "percent_count", "tilde_count", "plus_count", "special_char_count",
    "subdomain_count", "has_ip", "has_hex_ip", "has_port", "has_https",
    "is_shortened", "url_entropy", "hostname_entropy", "longest_token_length",
    "tld_length", "suspicious_word_count", "has_double_slash_in_path",
    "digit_in_hostname", "vowel_consonant_ratio",
]


def _entropy(s: str) -> float:
    """Shannon entropy. Algorithmically generated domains score high here."""
    if not s:
        return 0.0
    counts = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def extract(url: str) -> np.ndarray:
    """Return the feature vector for one URL, in FEATURE_NAMES order."""
    url = (url or "").strip()
    work = url if "://" in url else "http://" + url
    try:
        p = urlparse(work)
    except ValueError:
        p = urlparse("http://invalid")

    host = (p.hostname or "").lower()
    path = p.path or ""
    query = p.query or ""

    digits = sum(c.isdigit() for c in url)
    letters = sum(c.isalpha() for c in url)
    vowels = sum(c in "aeiou" for c in url.lower())
    consonants = letters - vowels

    host_parts = host.split(".") if host else []
    tokens = re.split(r"[\/\.\-_\?\=\&]+", url)
    longest = max((len(t) for t in tokens), default=0)

    specials = sum(url.count(c) for c in "@-_?=&%~+!*(),$;:")

    f = {
        "url_length": len(url),
        "hostname_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "path_depth": len([x for x in path.split("/") if x]),
        "digit_count": digits,
        "letter_count": letters,
        "digit_letter_ratio": digits / letters if letters else 0.0,
        "dot_count": url.count("."),
        "hyphen_count": url.count("-"),
        "underscore_count": url.count("_"),
        "slash_count": url.count("/"),
        "question_count": url.count("?"),
        "equal_count": url.count("="),
        "at_count": url.count("@"),
        "ampersand_count": url.count("&"),
        "percent_count": url.count("%"),
        "tilde_count": url.count("~"),
        "plus_count": url.count("+"),
        "special_char_count": specials,
        "subdomain_count": max(len(host_parts) - 2, 0),
        "has_ip": int(bool(IP_RE.match(host))),
        "has_hex_ip": int(bool(HEX_IP_RE.match(host))),
        "has_port": int(p.port is not None),
        "has_https": int(url.lower().startswith("https")),
        "is_shortened": int(host in SHORTENERS),
        "url_entropy": _entropy(url),
        "hostname_entropy": _entropy(host),
        "longest_token_length": longest,
        "tld_length": len(host_parts[-1]) if host_parts else 0,
        "suspicious_word_count": sum(w in url.lower() for w in SUSPICIOUS_WORDS),
        "has_double_slash_in_path": int("//" in path),
        "digit_in_hostname": sum(c.isdigit() for c in host),
        "vowel_consonant_ratio": vowels / consonants if consonants else 0.0,
    }
    return np.array([f[k] for k in FEATURE_NAMES], dtype=np.float32)


def extract_batch(urls) -> np.ndarray:
    return np.vstack([extract(u) for u in urls])


def to_sentence(url: str) -> str:
    """Split a URL into pseudo-words so a subword tokenizer has something to chew on.

    URLs are not natural language. Feeding the raw string to BERT wastes most of
    the vocabulary, so we break on the delimiters that carry structure.
    """
    return " ".join(t for t in re.split(r"[\/\.\-_\?\=\&:]+", (url or "").strip()) if t)


def notable_flags(url: str) -> list:
    """Human-readable reasons, for the explanation panel in the UI."""
    v = dict(zip(FEATURE_NAMES, extract(url)))
    out = []
    if v["has_ip"] or v["has_hex_ip"]:
        out.append("Uses a raw IP address instead of a domain name")
    if v["is_shortened"]:
        out.append("Hosted on a URL shortening service, so the destination is hidden")
    if v["url_length"] > 75:
        out.append(f"Unusually long at {int(v['url_length'])} characters")
    if v["subdomain_count"] >= 3:
        out.append(f"{int(v['subdomain_count'])} subdomain levels, often used to bury a brand name")
    if v["at_count"]:
        out.append("Contains an @ symbol, which can hide the real host")
    if v["suspicious_word_count"]:
        out.append("Contains credential-related words such as login, verify or account")
    if v["url_entropy"] > 4.5:
        out.append("High character entropy, typical of generated domains")
    if not v["has_https"]:
        out.append("Not served over HTTPS")
    if v["has_port"]:
        out.append("Specifies a non-standard port")
    return out
