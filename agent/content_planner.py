"""Backward-compatibility shim. Import from agent.scheduling.content_planner instead."""
import importlib as _il, sys as _sys
_mod = _il.import_module("agent.scheduling.content_planner")
_sys.modules[__name__] = _mod
