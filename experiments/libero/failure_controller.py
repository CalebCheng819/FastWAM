"""Fail-closed failure/recovery controller for LIBERO evaluation.

This module deliberately contains no model or evaluator dependencies.  A detector
feeds boolean alarm decisions into :meth:`FailureController.observe_alarm`, while
the evaluator explicitly reports recovery start/completion.  In particular, an
alarm never asks the caller to reset learned model context: confirmation only
invalidates pending actions and requests a replan from the existing context.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class FailureState(str, Enum):
    """Externally visible controller states."""

    NORMAL = "Normal"
    SUSPECTED = "Suspected"
    CONFIRMED = "Confirmed"
    RECOVERY = "Recovery"
    RECOVERED = "Recovered"
    ABORT = "Abort"


class ControllerEvent(str, Enum):
    """Side-effect requests emitted by a transition."""

    EPISODE_RESET = "episode_reset"
    CLEAR_PENDING_ACTIONS = "clear_pending_actions"
    REPLAN = "replan"
    RECOVERY_STARTED = "recovery_started"
    RECOVERY_SUCCEEDED = "recovery_succeeded"
    RECOVERY_FAILED = "recovery_failed"
    ABORT = "abort"


@dataclass(frozen=True, slots=True)
class FailureControllerConfig:
    """Configuration for consecutive-alarm confirmation and hysteresis.

    ``confirm_after`` is the number of consecutive detector alarms required to
    confirm a failure.  ``clear_after`` is the number of consecutive healthy
    observations required to clear either a suspicion or a recovered state.
    Recovery attempts are counted over the entire episode and are reset only by
    :meth:`FailureController.reset_episode`.
    """

    confirm_after: int = 2
    clear_after: int = 2
    max_recovery_attempts: int = 2

    def __post_init__(self) -> None:
        for name in ("confirm_after", "clear_after", "max_recovery_attempts"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be an integer >= 1, got {value!r}")


@dataclass(frozen=True, slots=True)
class TransitionReceipt:
    """Immutable record of one controller decision.

    ``events`` is an ordered tuple so consumers can clear stale action chunks
    before replanning.  ``reset_learned_context`` is intentionally always false:
    this controller never converts a detector alarm into a learned-context reset.
    """

    episode_id: Optional[str]
    sequence: int
    previous_state: FailureState
    state: FailureState
    alarm: Optional[bool]
    alarm_streak: int
    healthy_streak: int
    recovery_attempts: int
    events: Tuple[ControllerEvent, ...]
    reason: str
    reset_learned_context: bool = False

    @property
    def transitioned(self) -> bool:
        return self.previous_state is not self.state

    @property
    def clear_pending_actions(self) -> bool:
        return ControllerEvent.CLEAR_PENDING_ACTIONS in self.events

    @property
    def replan(self) -> bool:
        return ControllerEvent.REPLAN in self.events


class FailureController:
    """State machine coordinating failure confirmation and bounded recovery.

    Construction is fail-closed: the controller starts in :attr:`ABORT` and must
    be initialized for every episode with :meth:`reset_episode`.  Invalid runtime
    inputs or illegal transitions also enter ``Abort`` without emitting action
    clearing or replanning requests.  A new explicit episode reset is the only
    way out of ``Abort``.
    """

    def __init__(self, config: FailureControllerConfig) -> None:
        if not isinstance(config, FailureControllerConfig):
            raise TypeError("config must be a FailureControllerConfig")
        self._config = config
        self._episode_id: Optional[str] = None
        self._seen_episode_ids: set[str] = set()
        self._sequence = 0
        self._state = FailureState.ABORT
        self._alarm_streak = 0
        self._healthy_streak = 0
        self._recovery_attempts = 0

    @property
    def state(self) -> FailureState:
        return self._state

    @property
    def episode_id(self) -> Optional[str]:
        return self._episode_id

    @property
    def recovery_attempts(self) -> int:
        return self._recovery_attempts

    def reset_episode(self, episode_id: str) -> TransitionReceipt:
        """Explicitly reset all controller state for one new episode."""

        if type(episode_id) is not str or not episode_id.strip():
            return self._fail_closed("invalid episode_id")
        if episode_id in self._seen_episode_ids:
            return self._fail_closed("episode_id has already been used")

        previous = self._state
        self._episode_id = episode_id
        self._seen_episode_ids.add(episode_id)
        self._sequence = 0
        self._state = FailureState.NORMAL
        self._alarm_streak = 0
        self._healthy_streak = 0
        self._recovery_attempts = 0
        return self._receipt(
            previous,
            alarm=None,
            events=(ControllerEvent.EPISODE_RESET,),
            reason="episode_reset",
        )

    def observe_alarm(self, alarm: bool) -> TransitionReceipt:
        """Consume one detector decision and return an immutable decision receipt."""

        if type(alarm) is not bool:
            return self._fail_closed("alarm must be bool")
        if self._episode_id is None:
            return self._fail_closed("episode not initialized")

        previous = self._state

        if self._state is FailureState.ABORT:
            return self._receipt(previous, alarm=alarm, reason="controller_aborted")

        if self._state is FailureState.NORMAL:
            if not alarm:
                self._alarm_streak = 0
                self._healthy_streak = 0
                return self._receipt(previous, alarm=alarm, reason="healthy")
            return self._start_suspicion(previous, alarm)

        if self._state is FailureState.SUSPECTED:
            if alarm:
                self._healthy_streak = 0
                self._alarm_streak += 1
                if self._alarm_streak >= self._config.confirm_after:
                    return self._confirm(previous, alarm)
                return self._receipt(previous, alarm=alarm, reason="alarm_pending_confirmation")

            # Confirmation requires consecutive alarms.  Hysteresis separately
            # requires consecutive healthy observations before returning Normal.
            self._alarm_streak = 0
            self._healthy_streak += 1
            if self._healthy_streak >= self._config.clear_after:
                self._state = FailureState.NORMAL
                self._healthy_streak = 0
                return self._receipt(previous, alarm=alarm, reason="suspicion_cleared")
            return self._receipt(previous, alarm=alarm, reason="healthy_pending_clear")

        if self._state is FailureState.CONFIRMED:
            return self._receipt(previous, alarm=alarm, reason="failure_latched")

        if self._state is FailureState.RECOVERY:
            return self._receipt(previous, alarm=alarm, reason="recovery_in_progress")

        # A declared recovery is not accepted as stable until clear_after healthy
        # samples.  A renewed alarm starts a fresh, hysteretic confirmation cycle.
        if self._state is FailureState.RECOVERED:
            if alarm:
                self._healthy_streak = 0
                return self._start_suspicion(previous, alarm)
            self._healthy_streak += 1
            if self._healthy_streak >= self._config.clear_after:
                self._state = FailureState.NORMAL
                self._healthy_streak = 0
                return self._receipt(previous, alarm=alarm, reason="recovery_stable")
            return self._receipt(previous, alarm=alarm, reason="recovery_pending_stability")

        return self._fail_closed("unknown controller state")

    def begin_recovery(self) -> TransitionReceipt:
        """Start one bounded recovery attempt from a confirmed failure."""

        if self._episode_id is None:
            return self._fail_closed("episode not initialized")
        previous = self._state
        if self._state is not FailureState.CONFIRMED:
            return self._fail_closed("recovery can only start from Confirmed")
        if self._recovery_attempts >= self._config.max_recovery_attempts:
            return self._fail_closed("recovery attempt limit exhausted")

        self._recovery_attempts += 1
        self._state = FailureState.RECOVERY
        self._alarm_streak = 0
        self._healthy_streak = 0
        return self._receipt(
            previous,
            alarm=None,
            events=(ControllerEvent.RECOVERY_STARTED,),
            reason="recovery_started",
        )

    def finish_recovery(self, success: bool) -> TransitionReceipt:
        """Report the result of the active recovery attempt."""

        if type(success) is not bool:
            return self._fail_closed("success must be bool")
        if self._episode_id is None:
            return self._fail_closed("episode not initialized")
        previous = self._state
        if self._state is not FailureState.RECOVERY:
            return self._fail_closed("recovery result requires Recovery state")

        self._alarm_streak = 0
        self._healthy_streak = 0
        if success:
            self._state = FailureState.RECOVERED
            return self._receipt(
                previous,
                alarm=None,
                events=(ControllerEvent.RECOVERY_SUCCEEDED,),
                reason="recovery_succeeded",
            )

        if self._recovery_attempts >= self._config.max_recovery_attempts:
            self._state = FailureState.ABORT
            return self._receipt(
                previous,
                alarm=None,
                events=(ControllerEvent.RECOVERY_FAILED, ControllerEvent.ABORT),
                reason="recovery_failed_attempt_limit",
            )

        # Retry from Confirmed without re-emitting clear/replan.  Those requests
        # were already emitted exactly once when this failure was confirmed.
        self._state = FailureState.CONFIRMED
        return self._receipt(
            previous,
            alarm=None,
            events=(ControllerEvent.RECOVERY_FAILED,),
            reason="recovery_failed_retry_available",
        )

    def _start_suspicion(
        self,
        previous: FailureState,
        alarm: bool,
    ) -> TransitionReceipt:
        self._alarm_streak = 1
        self._healthy_streak = 0
        if self._config.confirm_after == 1:
            return self._confirm(previous, alarm)
        self._state = FailureState.SUSPECTED
        return self._receipt(previous, alarm=alarm, reason="failure_suspected")

    def _confirm(
        self,
        previous: FailureState,
        alarm: bool,
    ) -> TransitionReceipt:
        self._state = FailureState.CONFIRMED
        self._healthy_streak = 0
        return self._receipt(
            previous,
            alarm=alarm,
            events=(ControllerEvent.CLEAR_PENDING_ACTIONS, ControllerEvent.REPLAN),
            reason="failure_confirmed",
        )

    def _fail_closed(self, reason: str) -> TransitionReceipt:
        previous = self._state
        self._state = FailureState.ABORT
        self._alarm_streak = 0
        self._healthy_streak = 0
        events: Tuple[ControllerEvent, ...]
        if previous is FailureState.ABORT:
            events = ()
        else:
            events = (ControllerEvent.ABORT,)
        return self._receipt(previous, alarm=None, events=events, reason=reason)

    def _receipt(
        self,
        previous: FailureState,
        *,
        alarm: Optional[bool],
        events: Tuple[ControllerEvent, ...] = (),
        reason: str,
    ) -> TransitionReceipt:
        self._sequence += 1
        return TransitionReceipt(
            episode_id=self._episode_id,
            sequence=self._sequence,
            previous_state=previous,
            state=self._state,
            alarm=alarm,
            alarm_streak=self._alarm_streak,
            healthy_streak=self._healthy_streak,
            recovery_attempts=self._recovery_attempts,
            events=events,
            reason=reason,
        )
