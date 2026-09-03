"""Declarative argparse generation from a request dataclass (I3 declare-once).

``build_parser_from_dataclass`` reads each field's ``cli`` metadata (produced by
``src.mhs.contracts.cli_param``) and registers the matching
argparse option; ``request_from_namespace`` reconstructs the request from the
parsed namespace. Adding one MHS execution option therefore requires editing
exactly one request field: its flag, help, choices, and validation all derive
from that field's metadata.
"""

from __future__ import annotations

import argparse
import dataclasses
from typing import Any


def build_parser_from_dataclass(
    parser: argparse.ArgumentParser,
    request_cls: type,
) -> argparse.ArgumentParser:
    """Register one argparse option per request field carrying ``cli`` metadata."""
    for field in dataclasses.fields(request_cls):
        meta = field.metadata
        if not meta.get("flag"):
            continue
        flag = meta["flag"]
        default = field.default if field.default is not dataclasses.MISSING else None
        if not isinstance(default, (type(None), int, float, str, bool)):
            # Non-plain defaults (e.g. the committee_target_gross sentinel)
            # never surface on the CLI: the handler resolves them.
            default = None
        if _is_bool_type(field.type):
            if meta.get("negate_flag"):
                # "Main logic default ON": only the negation flag is exposed, so
                # the handler derives the field from its absence/presence.
                parser.add_argument(
                    meta["negate_flag"], action="store_true", default=False,
                    help=f"Disable {flag}",
                )
            else:
                parser.add_argument(
                    flag, action="store_true", default=bool(default), help=meta["help"],
                )
            continue
        kwargs: dict[str, Any] = {"default": default, "help": meta["help"]}
        if meta.get("choices") is not None:
            kwargs["choices"] = list(meta["choices"])
        elif field.type in ("int", "int | None"):
            kwargs["type"] = int
        elif field.type in ("float", "float | None"):
            kwargs["type"] = float
        parser.add_argument(flag, **kwargs)
        if meta.get("negate_flag"):
            parser.add_argument(
                meta["negate_flag"], action="store_true", default=False,
                help=f"Disable {flag}",
            )
    return parser


def request_from_namespace(request_cls: type, args: argparse.Namespace) -> object:
    """Reconstruct a request from parsed args using each field's ``cli`` metadata."""
    kwargs: dict[str, Any] = {}
    for field in dataclasses.fields(request_cls):
        meta = field.metadata
        if not meta.get("flag"):
            continue
        flag = meta["flag"]
        attr = flag.lstrip("-").replace("-", "_")
        if _is_bool_type(field.type):
            if meta.get("negate_flag"):
                # Negated flags invert a default-ON field; the positive flag
                # itself is never set by the CLI (it only carries a negation).
                kwargs[field.name] = not getattr(args, attr, False)
            else:
                kwargs[field.name] = bool(getattr(args, attr, False))
            continue
        kwargs[field.name] = getattr(args, attr, None)
    return request_cls(**kwargs)


def _is_bool_type(type_repr: Any) -> bool:
    name = getattr(type_repr, "__name__", str(type_repr))
    return name == "bool" or str(type_repr) == "bool"
