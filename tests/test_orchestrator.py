from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from orchestrator import (
    Feedback, FeedbackMismatch, OrchestrationError, Orchestrator, ReviewRequired, Target,
)


class OrchestratorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "state.json"
        self.now = 0.0
        self.inputs = {"mapping.json": b'{"W1":"H1"}', "mock-action.json": b'{"version":1}'}
        self.flow = Orchestrator(self.path, timeout_seconds=5, clock=lambda: self.now)

    def freeze(self, frame="f1", wire="W1", hole="H1", grasp="synthetic-g1"):
        return self.flow.freeze(frame, "scene-1", [Target(wire, hole, grasp)], "reviewed",
                                input_contents=self.inputs)

    def dispatch(self, snapshot=None):
        snapshot = snapshot or self.freeze()
        return self.flow.dispatch(snapshot, snapshot.targets[0].wire_id,
                                  snapshot_sha256=snapshot.sha256,
                                  current_frame_id=snapshot.frame_id,
                                  current_scene_revision=snapshot.scene_revision,
                                  current_input_contents=self.inputs)

    def test_snapshot_is_immutable_and_detached_from_input_list(self):
        targets = [Target("W1", "H1", "synthetic-g1")]
        snapshot = self.flow.freeze("f1", "scene", targets, "reviewed", input_contents=self.inputs)
        digest = snapshot.sha256
        targets.append(Target("W2", "H2", "synthetic-g2"))
        self.assertEqual(len(snapshot.targets), 1)
        self.assertEqual(snapshot.sha256, digest)
        with self.assertRaises(FrozenInstanceError):
            snapshot.frame_id = "changed"
        self.assertNotEqual(replace(snapshot, scene_revision="changed").sha256, digest)

    def test_digest_and_current_scene_are_checked(self):
        snapshot = self.freeze()
        with self.assertRaises(ReviewRequired):
            self.flow.dispatch(snapshot, "W1", snapshot_sha256="0" * 64,
                               current_frame_id="f1", current_scene_revision="scene-1",
                               current_input_contents=self.inputs)
        with self.assertRaises(ReviewRequired):
            self.flow.dispatch(snapshot, "W1", snapshot_sha256=snapshot.sha256,
                               current_frame_id="f1", current_scene_revision="moved",
                               current_input_contents=self.inputs)
        with self.assertRaises(ReviewRequired):
            self.dispatch(snapshot)

    def test_replaced_review_is_rejected(self):
        previous = self.freeze()
        self.freeze("f2")
        with self.assertRaises(ReviewRequired):
            self.dispatch(previous)

    def test_one_in_flight_and_completed_identities_cannot_repeat(self):
        dispatch = self.dispatch()
        with self.assertRaises(OrchestrationError):
            self.freeze("f2", "W2", "H2")
        self.assertEqual(self.flow.feedback(Feedback(dispatch.binding, "COMPLETE")), "APPLIED")
        for wire, hole in (("W1", "H9"), ("W9", "H1")):
            with self.assertRaises(ReviewRequired):
                self.freeze("f2", wire, hole)
        with self.assertRaises(ReviewRequired):
            self.freeze("f1", "W2", "H2")

    def test_all_feedback_binding_fields_match_or_stop(self):
        for field in ("task_id", "review_id", "frame_id", "scene_revision",
                      "wire_id", "hole_id", "snapshot_sha256"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                flow = Orchestrator(Path(directory) / "state.json")
                snapshot = flow.freeze("f", "r", [Target("w", "h", "g")], "reviewed",
                                       input_contents=self.inputs)
                dispatch = flow.dispatch(snapshot, "w", snapshot_sha256=snapshot.sha256,
                                         current_frame_id="f", current_scene_revision="r",
                                         current_input_contents=self.inputs)
                changed = replace(dispatch.binding, **{
                    field: "0" * 64 if field == "snapshot_sha256" else "wrong"})
                with self.assertRaises(FeedbackMismatch):
                    flow.feedback(Feedback(changed, "COMPLETE"))
                self.assertTrue(flow.blocked)
                self.assertFalse(flow.completed)
                with self.assertRaises(ReviewRequired):
                    flow.feedback(Feedback(dispatch.binding, "COMPLETE"))

    def test_success_is_idempotent_even_after_restart(self):
        dispatch = self.dispatch()
        self.flow.feedback(Feedback(dispatch.binding, "COMPLETE"))
        self.assertEqual(self.flow.feedback(Feedback(dispatch.binding, "COMPLETE")), "DUPLICATE")
        reopened = Orchestrator(self.path)
        self.assertEqual(reopened.feedback(Feedback(dispatch.binding, "COMPLETE")), "DUPLICATE")
        self.assertEqual(reopened.completed, (("W1", "H1"),))

    def test_failed_feedback_blocks_late_success(self):
        dispatch = self.dispatch()
        self.flow.feedback(Feedback(dispatch.binding, "FAILED"))
        with self.assertRaises(ReviewRequired):
            self.flow.feedback(Feedback(dispatch.binding, "COMPLETE"))
        with self.assertRaises(ReviewRequired):
            self.freeze("f2", "W2", "H2")
        self.flow.resolve_after_review(task_id=dispatch.binding.task_id,
                                       confirmed_completed=False, note="确认本次未完成")
        with self.assertRaises(ReviewRequired):
            self.flow.feedback(Feedback(dispatch.binding, "COMPLETE"))
        self.dispatch(self.freeze("f2"))

    def test_feedback_at_deadline_times_out_before_success(self):
        dispatch = self.dispatch()
        self.now = 5.0
        with self.assertRaises(ReviewRequired):
            self.flow.feedback(Feedback(dispatch.binding, "COMPLETE"))
        self.assertTrue(self.flow.blocked)
        self.assertFalse(self.flow.completed)
        self.assertFalse(self.flow.check_timeout())

    def test_explicit_timeout_and_late_success_stay_terminal(self):
        dispatch = self.dispatch()
        self.flow.feedback(Feedback(dispatch.binding, "TIMEOUT"))
        with self.assertRaises(ReviewRequired):
            self.flow.feedback(Feedback(dispatch.binding, "COMPLETE"))

    def test_restart_does_not_restore_pending_grasp_and_needs_resolution(self):
        old_snapshot = self.freeze(grasp="NEVER-PERSIST-THIS-GRASP")
        dispatch = self.dispatch(old_snapshot)
        self.assertNotIn("NEVER-PERSIST-THIS-GRASP", self.path.read_text(encoding="utf-8"))
        reopened = Orchestrator(self.path)
        self.assertIsNone(reopened.pending)
        self.assertTrue(reopened.blocked)
        with self.assertRaises(ReviewRequired):
            reopened.feedback(Feedback(dispatch.binding, "COMPLETE"))
        reopened.resolve_after_review(task_id=dispatch.binding.task_id,
                                      confirmed_completed=True, note="人工确认已完成")
        self.assertEqual(reopened.completed, (("W1", "H1"),))
        with self.assertRaises(ReviewRequired):
            reopened.dispatch(old_snapshot, "W1", snapshot_sha256=old_snapshot.sha256,
                              current_frame_id="f1", current_scene_revision="scene-1",
                              current_input_contents=self.inputs)

    def test_unknown_feedback_and_invalid_inputs(self):
        for timeout in (0, -1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                Orchestrator(self.path, timeout_seconds=timeout)
        with self.assertRaises(ValueError):
            self.flow.freeze("f", "scene", [Target("W", "H", "g"), Target("W", "J", "x")],
                             "reviewed", input_contents=self.inputs)
        with self.assertRaises(ValueError):
            Target("", "h", "g")
        dispatch = self.dispatch()
        with self.assertRaises(ValueError):
            Feedback(dispatch.binding, "RUNNING")

    def test_persistence_error_is_visible_and_does_not_issue_dispatch(self):
        snapshot = self.freeze()
        with patch("orchestrator.os.replace", side_effect=OSError("synthetic disk failure")):
            with self.assertRaises(OSError):
                self.dispatch(snapshot)
        self.assertIsNone(self.flow.pending)
        self.assertFalse(self.path.exists())

    def test_response_deadline_starts_after_slow_persistence(self):
        commit = self.flow._commit

        def slow_commit(state):
            commit(state)
            self.now += 20.0

        with patch.object(self.flow, "_commit", side_effect=slow_commit):
            self.dispatch()
        self.assertEqual(self.now, 20.0)
        self.assertFalse(self.flow.check_timeout())
        self.now = 24.9
        self.assertFalse(self.flow.check_timeout())
        self.now = 25.0
        self.assertTrue(self.flow.check_timeout())
        self.assertTrue(self.flow.blocked)

    def test_invalid_clock_after_commit_does_not_issue_or_allow_second_dispatch(self):
        snapshot = self.freeze()
        self.now = float("inf")
        with self.assertRaises(ValueError):
            self.dispatch(snapshot)
        self.assertIsNone(self.flow.pending)
        self.assertIsNotNone(json.loads(self.path.read_text(encoding="utf-8"))["active"])
        with self.assertRaises(ReviewRequired):
            self.freeze("f2", "W2", "H2")
        reopened = Orchestrator(self.path)
        self.assertTrue(reopened.blocked)
        self.assertIsNone(reopened.pending)

    def test_corrupt_state_is_not_silently_reset(self):
        self.path.write_text('{"schema": 999}', encoding="utf-8")
        with self.assertRaises(ValueError):
            Orchestrator(self.path)

    def test_mapping_and_action_content_digests_are_in_frozen_snapshot(self):
        snapshot = self.freeze()
        changed_inputs = {**self.inputs, "mock-action.json": b'{"version":2}'}
        changed_snapshot = self.flow.freeze(
            "f1", "scene-1", [Target("W1", "H1", "synthetic-g1")], "reviewed",
            input_contents=changed_inputs)
        self.assertNotEqual(snapshot.inputs, changed_snapshot.inputs)
        self.assertNotEqual(snapshot.sha256, changed_snapshot.sha256)
        with self.assertRaises(ReviewRequired):
            self.flow.dispatch(changed_snapshot, "W1", snapshot_sha256=changed_snapshot.sha256,
                               current_frame_id="f1", current_scene_revision="scene-1",
                               current_input_contents=self.inputs)
        self.assertIsNone(self.flow.pending)

    def test_mutating_caller_input_mapping_cannot_mutate_snapshot(self):
        inputs = dict(self.inputs)
        snapshot = self.flow.freeze("f1", "scene-1", [Target("W1", "H1", "g")],
                                    "reviewed", input_contents=inputs)
        digest = snapshot.sha256
        inputs["mapping.json"] = b'{"W1":"H999"}'
        self.assertEqual(snapshot.sha256, digest)
        with self.assertRaises(ReviewRequired):
            self.flow.dispatch(snapshot, "W1", snapshot_sha256=digest,
                               current_frame_id="f1", current_scene_revision="scene-1",
                               current_input_contents=inputs)


if __name__ == "__main__":
    unittest.main()
