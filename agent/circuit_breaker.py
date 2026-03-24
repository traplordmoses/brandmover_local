"""
Circuit breaker pattern for external API calls.
Prevents hammering a failing API endpoint every 5 minutes.

States: CLOSED (normal) -> OPEN (failing, reject calls) -> HALF_OPEN (test one call)

Usage:
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=300)

    if breaker.is_open:
        # Use fallback path
        ...
    else:
        try:
            result = await api_call()
            breaker.record_success()
        except Exception:
            breaker.record_failure()
            # Use fallback path
"""

import logging
import time
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Failing — reject calls
    HALF_OPEN = "half_open"  # Testing — allow one call


class CircuitBreaker:
    """Simple circuit breaker for external API calls.

    Parameters:
        name: Human-readable name for logging.
        failure_threshold: Consecutive failures before opening (default 3).
        recovery_timeout: Seconds to wait before trying again (default 300 = 5min).
    """

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 3,
        recovery_timeout: int = 300,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0
        self._success_count = 0
        self._total_failures = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> CircuitState:
        """Return the current state, auto-transitioning OPEN -> HALF_OPEN on timeout."""
        if self._state == CircuitState.OPEN:
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self.recovery_timeout:
                logger.info(
                    "CircuitBreaker[%s]: OPEN -> HALF_OPEN after %.0fs",
                    self.name, elapsed,
                )
                self._state = CircuitState.HALF_OPEN
        return self._state

    @property
    def is_open(self) -> bool:
        """True when the breaker is OPEN (calls should be skipped)."""
        return self.state == CircuitState.OPEN

    @property
    def is_closed(self) -> bool:
        """True when the breaker is CLOSED (normal operation)."""
        return self.state == CircuitState.CLOSED

    @property
    def allow_request(self) -> bool:
        """True when a request should be attempted (CLOSED or HALF_OPEN)."""
        return self.state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)

    def record_success(self) -> None:
        """Record a successful API call. Resets the breaker to CLOSED."""
        if self._state in (CircuitState.HALF_OPEN, CircuitState.OPEN):
            logger.info(
                "CircuitBreaker[%s]: %s -> CLOSED (success)",
                self.name, self._state.value,
            )
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count += 1

    def record_failure(self) -> None:
        """Record a failed API call. May transition to OPEN."""
        self._failure_count += 1
        self._total_failures += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            # Single failure in HALF_OPEN -> back to OPEN
            logger.warning(
                "CircuitBreaker[%s]: HALF_OPEN -> OPEN (test call failed)",
                self.name,
            )
            self._state = CircuitState.OPEN
        elif self._failure_count >= self.failure_threshold:
            logger.warning(
                "CircuitBreaker[%s]: CLOSED -> OPEN after %d consecutive failures",
                self.name, self._failure_count,
            )
            self._state = CircuitState.OPEN

    def reset(self) -> None:
        """Manually reset the breaker to CLOSED."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        logger.info("CircuitBreaker[%s]: manually reset to CLOSED", self.name)

    def get_stats(self) -> dict:
        """Return diagnostic stats."""
        return {
            "name": self.name,
            "state": self.state.value,
            "consecutive_failures": self._failure_count,
            "total_failures": self._total_failures,
            "total_successes": self._success_count,
            "last_failure_time": self._last_failure_time,
            "recovery_timeout": self.recovery_timeout,
        }


# ---------------------------------------------------------------------------
# Module-level singletons for shared breakers
# ---------------------------------------------------------------------------

# Heartbeat reasoning breaker — protects the Haiku call in heartbeat_reason()
heartbeat_breaker = CircuitBreaker(
    name="heartbeat_reasoning",
    failure_threshold=3,
    recovery_timeout=300,
)
