"""Backward-compatibility shim. Import from agent.publishing.publisher instead."""
import importlib as _il, sys as _sys
_mod = _il.import_module("agent.publishing.publisher")
_sys.modules[__name__] = _mod
