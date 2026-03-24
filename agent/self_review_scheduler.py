"""Backward-compatibility shim. Import from agent.quality.self_review_scheduler instead."""
import importlib as _il, sys as _sys
_mod = _il.import_module("agent.quality.self_review_scheduler")
_sys.modules[__name__] = _mod
