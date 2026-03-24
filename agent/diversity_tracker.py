"""Backward-compatibility shim. Import from agent.learning.diversity_tracker instead."""
import importlib as _il, sys as _sys
_mod = _il.import_module("agent.learning.diversity_tracker")
_sys.modules[__name__] = _mod
