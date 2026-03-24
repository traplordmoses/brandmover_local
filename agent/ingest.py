"""Backward-compatibility shim. Import from agent.brand.ingest instead."""
import importlib as _il, sys as _sys
_mod = _il.import_module("agent.brand.ingest")
_sys.modules[__name__] = _mod
