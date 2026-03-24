"""Backward-compatibility shim. Import from agent.publishing.publish_queue instead."""
import importlib as _il, sys as _sys
_mod = _il.import_module("agent.publishing.publish_queue")
_sys.modules[__name__] = _mod
