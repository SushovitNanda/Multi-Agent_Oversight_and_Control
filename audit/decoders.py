"""Decoders for the second detection level of knowledge closure.

Covers base64, hex, URL encoding, ROT13, reversal, and simple two-step chains.
Swarm Traces is used only to choose which chains are realistic; payloads are never replayed.
"""
import base64
import binascii
import codecs
import re
import urllib.parse

_TOKEN = re.compile(r"[A-Za-z0-9+/=_\-%.]{6,}")


def _b64(s):
    s = s.replace("-", "+").replace("_", "/")
    s = s + "=" * (-len(s) % 4)
    try:
        out = base64.b64decode(s, validate=False)
        return out.decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None


def _hex(s):
    if len(s) % 2 or not re.fullmatch(r"[0-9a-fA-F]+", s):
        return None
    try:
        return bytes.fromhex(s).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None


def _url(s):
    if "%" not in s:
        return None
    out = urllib.parse.unquote(s)
    return out if out != s else None


def _rot13(s):
    return codecs.decode(s, "rot13")


def _rev(s):
    return s[::-1]


SINGLE = {"b64": _b64, "hex": _hex, "url": _url, "rot13": _rot13, "rev": _rev}


def decode_all(text, max_depth=2):
    """Return the set of candidate plaintexts derivable from text, including text itself."""
    if not text:
        return set()
    out = {text}
    frontier = {text}
    for _ in range(max_depth):
        nxt = set()
        for t in frontier:
            # Try whole string and each token separately.
            pieces = {t} | set(_TOKEN.findall(t))
            pieces |= {sub for tok in list(pieces) for sub in re.split(r"[-_./:\s]", tok) if len(sub) >= 6}
            for p in pieces:
                for fn in SINGLE.values():
                    try:
                        d = fn(p)
                    except Exception:  # noqa: BLE001
                        d = None
                    if d and d.isprintable() and d not in out:
                        nxt.add(d)
        out |= nxt
        frontier = nxt
    return out


def encode(text, chain):
    """Apply an encoding chain, e.g. ("b64",), ("rev","hex"). Used by the offline hard-case generator."""
    s = text
    for step in chain:
        if step == "b64":
            s = base64.b64encode(s.encode()).decode()
        elif step == "hex":
            s = s.encode().hex()
        elif step == "url":
            s = urllib.parse.quote(s, safe="")
        elif step == "rot13":
            s = codecs.encode(s, "rot13")
        elif step == "rev":
            s = s[::-1]
        else:
            raise ValueError(step)
    return s
