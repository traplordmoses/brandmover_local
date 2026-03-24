"""Backward-compatibility shim. Import from agent.templates.template_spec instead."""
import importlib as _il, sys as _sys
_mod = _il.import_module("agent.templates.template_spec")
_sys.modules[__name__] = _mod
