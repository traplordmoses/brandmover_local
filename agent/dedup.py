"""Backward-compatibility shim. Import from agent.quality.dedup instead."""
import importlib as _il, sys as _sys
_mod = _il.import_module("agent.quality.dedup")
_sys.modules[__name__] = _mod
