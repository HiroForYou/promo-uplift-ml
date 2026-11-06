"""Reconfigura stdout a UTF-8 (evita UnicodeEncodeError con emojis/acentos en consolas Windows cp1252)."""
import sys


def reconfigure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
