"""
bot.handlers — re-exports all public names from submodules.

This package replaces the former monolithic bot/handlers.py.
All names that were importable via ``from bot.handlers import X``
or ``handlers.X`` continue to work unchanged.
"""

from bot.handlers.core import *        # noqa: F401,F403
from bot.handlers.draft import *       # noqa: F401,F403
from bot.handlers.generation import *  # noqa: F401,F403
from bot.handlers.media import *       # noqa: F401,F403
from bot.handlers.admin import *       # noqa: F401,F403
from bot.handlers.scheduling import *  # noqa: F401,F403
from bot.handlers.debug import *       # noqa: F401,F403

# Private names used by tests and other internal callers.
# (star-imports skip underscore-prefixed names unless __all__ is defined.)
from bot.handlers.core import (       # noqa: F401
    _maybe_compose,
    _can_operate,
    _authorized,
    _rate_limited,
    _cc,
    _PILImage,
    _is_template_from_ref_intent,
    _is_template_region_update,
    _is_direct_photo_intent,
    _bulk_upload_tasks,
)
from bot.handlers.draft import (      # noqa: F401
    _CallbackProxy,
    _do_approve,
    _do_reject,
    refine_command,
)
from bot.handlers.generation import ( # noqa: F401
    _route_intent,
    _handle_pipeline_mode,
    score_command,
    approval_rate_command,
)
from bot.handlers.media import (      # noqa: F401
    _merge_extracted,
    _process_bulk_upload,
    _delayed_bulk_process,
)
from bot.handlers.admin import _run_onboarding_audit   # noqa: F401
from bot.handlers.admin import health_command, digest_command  # noqa: F401
