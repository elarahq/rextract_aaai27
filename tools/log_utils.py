"""Pretty-printing helpers for workflow log output.

All functions write directly to stdout via print() so no timestamp
prefix is added by the logging formatter.
"""

import logging
import os
import sys

_tee_file = None
_log_handler = None
_orig_stdout = None


class _Tee:
    def __init__(self, stream, file):
        self._stream = stream
        self._file = file

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def open_log_file(path: str) -> None:
    global _tee_file, _log_handler, _orig_stdout
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _tee_file = open(path, "w", encoding="utf-8", buffering=1)
    _orig_stdout = sys.stdout
    sys.stdout = _Tee(_orig_stdout, _tee_file)
    _log_handler = logging.StreamHandler(_tee_file)
    _log_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(_log_handler)


def close_log_file() -> None:
    global _tee_file, _log_handler, _orig_stdout
    if _log_handler:
        logging.getLogger().removeHandler(_log_handler)
        _log_handler.close()
        _log_handler = None
    if _orig_stdout:
        sys.stdout = _orig_stdout
        _orig_stdout = None
    if _tee_file:
        _tee_file.close()
        _tee_file = None

_W = 66  # total display width for all boxes and banners
_BOX_INNER = _W - 4   # content width inside │ … │  (accounts for "│ " and " │")
_STEP_INNER = _W - 4  # same inner width for indented step boxes


# ---------------------------------------------------------------------------
# Run-level banners
# ---------------------------------------------------------------------------

def print_run_header(rera_id: str) -> None:
    print()
    print("═" * _W)
    print(f"  RERA DOCUMENT PARSER  ·  {rera_id}")
    print("═" * _W)
    print()


def print_run_footer(
    rera_id: str,
    elapsed: float,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
) -> None:
    print()
    print("═" * _W)
    print(f"  RUN COMPLETE  ·  {rera_id}")
    print(
        f"  elapsed: {elapsed}s  ·  "
        f"tokens: {input_tokens:,} in / {output_tokens:,} out / {total_tokens:,} total"
    )
    print("═" * _W)
    print()


# ---------------------------------------------------------------------------
# Download section divider
# ---------------------------------------------------------------------------

def print_download_header(n: int) -> None:
    label = f"  DOWNLOAD  ({n} doc{'s' if n != 1 else ''})  "
    left = (_W - len(label)) // 2
    right = _W - len(label) - left
    print()
    print("─" * left + label + "─" * right)


# ---------------------------------------------------------------------------
# Per-document box
# ---------------------------------------------------------------------------

def _box_row(content: str) -> str:
    return f"│ {content:<{_BOX_INNER}} │"


def print_document_box(index: int, total: int, name: str, category: str) -> None:
    print()
    print("┌" + "─" * (_W - 2) + "┐")
    print(_box_row(f"DOCUMENT  {index} / {total}  ·  {category}"))
    remaining = name
    while remaining:
        chunk, remaining = remaining[:_BOX_INNER], remaining[_BOX_INNER:]
        print(_box_row(chunk))
    print("└" + "─" * (_W - 2) + "┘")
    print()


# ---------------------------------------------------------------------------
# Step-level open / close borders
# ---------------------------------------------------------------------------

def print_step_open(label: str) -> None:
    # "  ┌─ LABEL ───────────────────────────────────────────────┐"
    fill = _STEP_INNER - 3 - len(label)  # "─ " + label + " " + fill
    fill = max(fill, 1)
    print(f"\n  ┌─ {label} {'─' * fill}┐")


def print_step_close() -> None:
    print(f"  └{'─' * _STEP_INNER}┘")


# ---------------------------------------------------------------------------
# Outcome one-liners
# ---------------------------------------------------------------------------

def print_saved(rera_id: str) -> None:
    print(f"\n  ✓  SAVED  ·  rera_id={rera_id}\n")


def print_skipped(doc_type: str) -> None:
    print(f"\n  ✗  SKIPPED  ·  {doc_type!r} is not extractable\n")


def print_duplicate(name: str, original_name: str) -> None:
    print(f"\n  ⊘  DUPLICATE  ·  {name!r} matches {original_name!r}, skipping\n")
