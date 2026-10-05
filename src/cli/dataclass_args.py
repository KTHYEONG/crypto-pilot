"""Generic argparse generation from dataclass field metadata (declare-once CLI).

Every option is registered with ``default=argparse.SUPPRESS`` so a parsed
namespace contains exactly the fields the operator stated. Defaults and
cross-field resolution stay with the dataclass and its domain resolver; the
parser never invents a value.
"""

from __future__ import annotations

import argparse
import dataclasses
from typing import Any


def add_dataclass_arguments(parser: argparse.ArgumentParser, request_cls: type[Any]) -> None:
    """Register one option per field carrying ``flag`` metadata.

    Boolean fields (bool default) become ``store_const`` switches whose
    constant is the negated default; value fields become ``store`` options with
    ``type=arg_type`` and ``choices``; a value field's ``negate_flag`` becomes a
    ``store_const`` of ``None`` in a mutually exclusive group with its flag. Every
    option's ``dest`` is the field name.

    Raises:
        ValueError: a boolean field whose ``flag`` polarity contradicts its
            default, or a ``negate_flag`` on a boolean field.
    """
    for f in dataclasses.fields(request_cls):
        meta = f.metadata
        flag = meta.get("flag")
        if not flag:
            continue
        default = f.default if f.default is not dataclasses.MISSING else None
        is_bool = isinstance(default, bool)
        negate_flag = meta.get("negate_flag")
        help_text = meta.get("help", "")
        if is_bool:
            if negate_flag is not None:
                raise ValueError(f"negate_flag on boolean field {f.name}")
            is_neg = flag.startswith("--no-")
            if is_neg != (default is True):
                raise ValueError(f"flag polarity contradicts default for {f.name}: {flag}")
            parser.add_argument(
                flag,
                dest=f.name,
                action="store_const",
                const=not default,
                default=argparse.SUPPRESS,
                help=help_text,
            )
            continue
        arg_type = meta.get("arg_type")
        choices = meta.get("choices")
        if negate_flag is not None:
            group = parser.add_mutually_exclusive_group()
            kwargs: dict[str, Any] = {
                "dest": f.name,
                "action": "store",
                "default": argparse.SUPPRESS,
                "help": help_text,
            }
            if arg_type is not None:
                kwargs["type"] = arg_type
            if choices is not None:
                kwargs["choices"] = list(choices)
            group.add_argument(flag, **kwargs)
            group.add_argument(
                negate_flag,
                dest=f.name,
                action="store_const",
                const=None,
                default=argparse.SUPPRESS,
                help=f"Set {f.name} to None (opt-out of {flag}).",
            )
        else:
            kwargs = {
                "dest": f.name,
                "action": "store",
                "default": argparse.SUPPRESS,
                "help": help_text,
            }
            if arg_type is not None:
                kwargs["type"] = arg_type
            if choices is not None:
                kwargs["choices"] = list(choices)
            parser.add_argument(flag, **kwargs)


def explicit_field_values(request_cls: type[Any], args: argparse.Namespace) -> dict[str, Any]:
    """Field values the operator stated explicitly, keyed by field name; pure (never mutates ``args``)."""
    names = [f.name for f in dataclasses.fields(request_cls)]
    values = vars(args)
    return {name: values[name] for name in names if name in values}
