"""Backward-compatibility shim. Import from agent.quality.scoring instead."""
import importlib as _il, sys as _sys
_mod = _il.import_module("agent.quality.scoring")
_sys.modules[__name__] = _mod
