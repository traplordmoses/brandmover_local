"""Backward-compatibility shim. Import from agent.templates.template_memory instead."""
import importlib as _il, sys as _sys
_mod = _il.import_module("agent.templates.template_memory")
_sys.modules[__name__] = _mod
