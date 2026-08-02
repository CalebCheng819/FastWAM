import unittest
from dataclasses import FrozenInstanceError

from experiments.libero.failure_controller import (
    ControllerEvent,
    FailureController,
    FailureControllerConfig,
    FailureState,
)


class FailureControllerTest(unittest.TestCase):
    def make_controller(self, **overrides):
        config = FailureControllerConfig(**overrides)
        controller = FailureController(config)
        reset = controller.reset_episode("episode-0001")
        self.assertEqual(reset.events, (ControllerEvent.EPISODE_RESET,))
        return controller

    def confirm_failure(self, controller):
        first = controller.observe_alarm(True)
        second = controller.observe_alarm(True)
        self.assertEqual(first.state, FailureState.SUSPECTED)
        self.assertEqual(second.state, FailureState.CONFIRMED)
        return first, second

    def test_nominal_path_never_clears_actions(self):
        controller = self.make_controller()

        receipts = [controller.observe_alarm(False) for _ in range(5)]

        self.assertTrue(all(receipt.state is FailureState.NORMAL for receipt in receipts))
        self.assertTrue(all(not receipt.clear_pending_actions for receipt in receipts))
        self.assertTrue(all(not receipt.replan for receipt in receipts))
        self.assertTrue(all(not receipt.reset_learned_context for receipt in receipts))

    def test_single_alarm_uses_consecutive_confirmation_and_clear_hysteresis(self):
        controller = self.make_controller(confirm_after=2, clear_after=2)

        suspected = controller.observe_alarm(True)
        first_healthy = controller.observe_alarm(False)
        renewed_alarm = controller.observe_alarm(True)
        healthy_again = controller.observe_alarm(False)
        cleared = controller.observe_alarm(False)

        self.assertEqual(suspected.state, FailureState.SUSPECTED)
        self.assertEqual(first_healthy.state, FailureState.SUSPECTED)
        self.assertEqual(first_healthy.healthy_streak, 1)
        self.assertEqual(renewed_alarm.state, FailureState.SUSPECTED)
        self.assertEqual(renewed_alarm.alarm_streak, 1)
        self.assertEqual(healthy_again.state, FailureState.SUSPECTED)
        self.assertEqual(cleared.state, FailureState.NORMAL)
        self.assertFalse(any(
            receipt.clear_pending_actions
            for receipt in (suspected, first_healthy, renewed_alarm, healthy_again, cleared)
        ))

    def test_confirmation_emits_exactly_one_ordered_clear_and_replan(self):
        controller = self.make_controller()

        suspected, confirmed = self.confirm_failure(controller)
        latched_one = controller.observe_alarm(True)
        latched_two = controller.observe_alarm(False)

        self.assertEqual(suspected.events, ())
        self.assertEqual(
            confirmed.events,
            (ControllerEvent.CLEAR_PENDING_ACTIONS, ControllerEvent.REPLAN),
        )
        self.assertTrue(confirmed.clear_pending_actions)
        self.assertTrue(confirmed.replan)
        self.assertFalse(confirmed.reset_learned_context)
        all_receipts = (suspected, confirmed, latched_one, latched_two)
        self.assertEqual(
            sum(
                receipt.events.count(ControllerEvent.CLEAR_PENDING_ACTIONS)
                for receipt in all_receipts
            ),
            1,
        )
        self.assertEqual(
            sum(receipt.events.count(ControllerEvent.REPLAN) for receipt in all_receipts),
            1,
        )

    def test_recovery_success_requires_stable_healthy_hysteresis(self):
        controller = self.make_controller(clear_after=2)
        self.confirm_failure(controller)

        started = controller.begin_recovery()
        recovered = controller.finish_recovery(True)
        pending_stability = controller.observe_alarm(False)
        stable = controller.observe_alarm(False)

        self.assertEqual(started.state, FailureState.RECOVERY)
        self.assertEqual(started.recovery_attempts, 1)
        self.assertEqual(started.events, (ControllerEvent.RECOVERY_STARTED,))
        self.assertEqual(recovered.state, FailureState.RECOVERED)
        self.assertEqual(recovered.events, (ControllerEvent.RECOVERY_SUCCEEDED,))
        self.assertEqual(pending_stability.state, FailureState.RECOVERED)
        self.assertEqual(stable.state, FailureState.NORMAL)
        self.assertTrue(all(
            not receipt.reset_learned_context
            for receipt in (started, recovered, pending_stability, stable)
        ))

    def test_recovery_retries_are_bounded_and_abort(self):
        controller = self.make_controller(max_recovery_attempts=2)
        _, confirmed = self.confirm_failure(controller)

        first_start = controller.begin_recovery()
        first_failure = controller.finish_recovery(False)
        second_start = controller.begin_recovery()
        second_failure = controller.finish_recovery(False)

        self.assertEqual(first_start.recovery_attempts, 1)
        self.assertEqual(first_failure.state, FailureState.CONFIRMED)
        self.assertEqual(first_failure.events, (ControllerEvent.RECOVERY_FAILED,))
        self.assertEqual(second_start.recovery_attempts, 2)
        self.assertEqual(second_failure.state, FailureState.ABORT)
        self.assertEqual(
            second_failure.events,
            (ControllerEvent.RECOVERY_FAILED, ControllerEvent.ABORT),
        )
        receipts = (confirmed, first_start, first_failure, second_start, second_failure)
        self.assertEqual(
            sum(
                receipt.events.count(ControllerEvent.CLEAR_PENDING_ACTIONS)
                for receipt in receipts
            ),
            1,
        )

    def test_explicit_episode_reset_clears_all_counters_and_abort(self):
        controller = self.make_controller(max_recovery_attempts=1)
        self.confirm_failure(controller)
        controller.begin_recovery()
        controller.finish_recovery(False)
        self.assertEqual(controller.state, FailureState.ABORT)

        reset = controller.reset_episode("episode-0002")

        self.assertEqual(reset.episode_id, "episode-0002")
        self.assertEqual(reset.previous_state, FailureState.ABORT)
        self.assertEqual(reset.state, FailureState.NORMAL)
        self.assertEqual(reset.sequence, 1)
        self.assertEqual(reset.alarm_streak, 0)
        self.assertEqual(reset.healthy_streak, 0)
        self.assertEqual(reset.recovery_attempts, 0)

    def test_same_episode_id_cannot_reset_attempt_budget(self):
        controller = self.make_controller(max_recovery_attempts=1)
        self.confirm_failure(controller)
        controller.begin_recovery()
        controller.finish_recovery(False)
        self.assertEqual(controller.state, FailureState.ABORT)

        duplicate_reset = controller.reset_episode("episode-0001")

        self.assertEqual(duplicate_reset.state, FailureState.ABORT)
        self.assertEqual(duplicate_reset.reason, "episode_id has already been used")
        self.assertEqual(controller.recovery_attempts, 1)

        new_episode = controller.reset_episode("episode-0002")
        self.assertEqual(new_episode.state, FailureState.NORMAL)
        self.assertEqual(new_episode.recovery_attempts, 0)

    def test_invalid_configuration_and_inputs_fail_closed(self):
        for field in ("confirm_after", "clear_after", "max_recovery_attempts"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    FailureControllerConfig(**{field: 0})
                with self.assertRaises(ValueError):
                    FailureControllerConfig(**{field: True})

        controller = self.make_controller()
        invalid_alarm = controller.observe_alarm(1)
        self.assertEqual(invalid_alarm.state, FailureState.ABORT)
        self.assertEqual(invalid_alarm.events, (ControllerEvent.ABORT,))
        self.assertFalse(invalid_alarm.clear_pending_actions)
        self.assertFalse(invalid_alarm.replan)

        invalid_reset = controller.reset_episode("   ")
        self.assertEqual(invalid_reset.state, FailureState.ABORT)
        self.assertEqual(controller.state, FailureState.ABORT)

        clean_controller = self.make_controller()
        illegal_transition = clean_controller.begin_recovery()
        self.assertEqual(illegal_transition.state, FailureState.ABORT)
        self.assertEqual(illegal_transition.events, (ControllerEvent.ABORT,))

    def test_receipts_capture_transitions_and_are_immutable(self):
        controller = self.make_controller(confirm_after=1)

        confirmed = controller.observe_alarm(True)

        self.assertTrue(confirmed.transitioned)
        self.assertEqual(confirmed.previous_state, FailureState.NORMAL)
        self.assertEqual(confirmed.state, FailureState.CONFIRMED)
        self.assertEqual(confirmed.alarm_streak, 1)
        self.assertIsInstance(confirmed.events, tuple)
        with self.assertRaises(FrozenInstanceError):
            confirmed.state = FailureState.NORMAL


if __name__ == "__main__":
    unittest.main()
