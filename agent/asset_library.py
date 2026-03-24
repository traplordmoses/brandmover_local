"""Backward-compatibility shim. Import from agent.brand.asset_library instead."""
import importlib as _il, sys as _sys
_mod = _il.import_module("agent.brand.asset_library")
_sys.modules[__name__] = _mod
