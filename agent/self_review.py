"""Backward-compatibility shim. Import from agent.quality.self_review instead."""
import importlib as _il, sys as _sys
_mod = _il.import_module("agent.quality.self_review")
_sys.modules[__name__] = _mod
