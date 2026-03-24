"""Backward-compatibility shim. Import from agent.publishing.platform_adapter instead."""
import importlib as _il, sys as _sys
_mod = _il.import_module("agent.publishing.platform_adapter")
_sys.modules[__name__] = _mod
