"""Phase 6: adversarial robustness.

Every model here reads the URL string, so an attacker who knows that can rewrite
the string without changing where it points. These are the cheap evasions that
actually get used: swapping letters for lookalike characters, percent-encoding
parts of the path, padding the hostname with subdomains, and hiding the whole
thing behind a shortener.

The point of the test is not that accuracy drops, it will. The point is which
branch drops less. Lexical features collapse when the surface statistics are
rewritten; the semantic branch should hold on longer, and the ensemble should
degrade least of all. That comparison is the strongest chart in the report.
"""

import random
import re
from urllib.parse import urlparse

HOMOGLYPHS = {
    "a": "\u0430",   # Cyrillic a
    "e": "\u0435",   # Cyrillic e
    "o": "\u043e",   # Cyrillic o
    "p": "\u0440",   # Cyrillic er
    "c": "\u0441",   # Cyrillic es
    "i": "\u0456",   # Ukrainian i
    "l": "1",
    "s": "\u0455",
}

SHORTENERS = ["bit.ly", "tinyurl.com", "cutt.ly", "rb.gy", "is.gd"]
PAD_WORDS = ["secure", "account", "verify", "login", "update", "session", "auth"]


def homoglyph(url, rate=0.3, seed=None):
    """Replace some Latin letters with visually identical Unicode characters."""
    rng = random.Random(seed)
    out = []
    for ch in url:
        low = ch.lower()
        if low in HOMOGLYPHS and rng.random() < rate:
            out.append(HOMOGLYPHS[low])
        else:
            out.append(ch)
    return "".join(out)


def percent_encode(url, rate=0.35, seed=None):
    """Percent-encode characters in the path, which browsers still resolve."""
    rng = random.Random(seed)
    p = urlparse(url if "://" in url else "http://" + url)
    if not p.path:
        return url
    path = "".join(f"%{ord(c):02X}" if c.isalnum() and rng.random() < rate else c
                   for c in p.path)
    return url.replace(p.path, path, 1)


def subdomain_pad(url, depth=3, seed=None):
    """Bury the real host under plausible-looking subdomains."""
    rng = random.Random(seed)
    p = urlparse(url if "://" in url else "http://" + url)
    if not p.hostname:
        return url
    pad = ".".join(rng.sample(PAD_WORDS, min(depth, len(PAD_WORDS))))
    return url.replace(p.hostname, f"{pad}.{p.hostname}", 1)


def shorten(url, seed=None):
    """Wrap the URL in a shortener, hiding everything after the domain."""
    rng = random.Random(seed)
    token = "".join(rng.choice("abcdefghijkmnpqrstuvwxyz23456789") for _ in range(7))
    return f"https://{rng.choice(SHORTENERS)}/{token}"


def char_swap(url, rate=0.15, seed=None):
    """Insert hyphens and digits, the way typosquatted domains do."""
    rng = random.Random(seed)
    p = urlparse(url if "://" in url else "http://" + url)
    host = p.hostname or ""
    if not host:
        return url
    parts = host.split(".")
    if len(parts) >= 2:
        core = parts[-2]
        pos = rng.randrange(1, len(core)) if len(core) > 2 else 1
        parts[-2] = core[:pos] + rng.choice("-0123456789") + core[pos:]
    return url.replace(host, ".".join(parts), 1)


ATTACKS = {
    "homoglyph substitution": homoglyph,
    "percent encoding": percent_encode,
    "subdomain padding": subdomain_pad,
    "shortener wrapping": shorten,
    "typosquat characters": char_swap,
}


def perturb(urls, attack, seed=42):
    fn = ATTACKS[attack]
    return [fn(u, seed=seed + i) for i, u in enumerate(urls)]
