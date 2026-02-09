#!/usr/bin/env python3
"""
Update elements.<element>.roi {x,y,w,h} in layout.yaml using text + regex only.
Usage: python scripts/set_roi.py <element> <x> <y> <w> <h>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


LAYOUT_PATH = Path("ai/card_verification/registry/templates/national_id_yemen_v1/layout.yaml")


def _usage() -> int:
    print("Usage: python scripts/set_roi.py <element> <x> <y> <w> <h>")
    return 1


def _parse_args(argv: list[str]) -> tuple[str, str, str, str, str]:
    if len(argv) != 6:
        raise ValueError("invalid args")
    _, element, x, y, w, h = argv
    return element, x, y, w, h


def _format_block(indent: str, x: str, y: str, w: str, h: str) -> str:
    return (
        f"{indent}x: {x}\n"
        f"{indent}y: {y}\n"
        f"{indent}w: {w}\n"
        f"{indent}h: {h}\n"
    )


def _update_roi(text: str, element: str, x: str, y: str, w: str, h: str) -> str:
    # Locate the element block with indentation preserved.
    # Example:
    #   emblem:
    #     required: true
    #     roi:
    #       x: ...
    #       y: ...
    #       w: ...
    #       h: ...
    element_re = re.compile(
        r"(?m)^(?P<indent>\s{2})" + re.escape(element) + r":\s*$"
    )
    m = element_re.search(text)
    if not m:
        raise RuntimeError(f"Element not found: {element}")

    base_indent = m.group("indent")
    roi_indent = base_indent + "  "
    kv_indent = roi_indent + "  "

    # Find the roi: line within this element block.
    # Limit search to the element block by scanning until next top-level element.
    start = m.end()
    next_elem = re.search(r"(?m)^\s{2}\S", text[start:])
    end = start + next_elem.start() if next_elem else len(text)
    block = text[start:end]

    roi_re = re.compile(r"(?m)^" + re.escape(roi_indent) + r"roi:\s*$")
    roi_match = roi_re.search(block)
    if not roi_match:
        raise RuntimeError(f"roi block not found for element: {element}")

    # Split block into before roi, roi section, after roi section.
    roi_start = roi_match.end()
    # Remove any existing x/y/w/h lines under roi (including duplicates).
    # Also remove any extra blank lines immediately after the roi block to keep formatting clean.
    remainder = block[roi_start:]
    kv_re = re.compile(
        r"(?m)^" + re.escape(kv_indent) + r"(x|y|w|h)\s*:\s*.*$\n?"
    )
    remainder = kv_re.sub("", remainder)
    # Remove extra blank lines directly after roi:
    remainder = re.sub(r"(?m)^\s*$\n?", "", remainder, count=1)

    new_roi_block = "\n" + _format_block(kv_indent, x, y, w, h)
    new_block = block[:roi_start] + new_roi_block + remainder

    return text[:start] + new_block + text[end:]


def main() -> int:
    try:
        element, x, y, w, h = _parse_args(sys.argv)
    except ValueError:
        return _usage()

    if not LAYOUT_PATH.exists():
        print(f"Layout file not found: {LAYOUT_PATH}")
        return 1

    text = LAYOUT_PATH.read_text(encoding="utf-8")
    updated = _update_roi(text, element, x, y, w, h)
    LAYOUT_PATH.write_text(updated, encoding="utf-8")
    print(f"DONE: {element} roi updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
