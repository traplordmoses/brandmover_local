"""Backward-compatibility shim. Import from agent.quality.risk_score instead."""
import importlib as _il, sys as _sys
_mod = _il.import_module("agent.quality.risk_score")
_sys.modules[__name__] = _mod
