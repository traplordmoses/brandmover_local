"""Backward-compatibility shim. Import from agent.publishing.discord_publisher instead."""
import importlib as _il, sys as _sys
_mod = _il.import_module("agent.publishing.discord_publisher")
_sys.modules[__name__] = _mod
