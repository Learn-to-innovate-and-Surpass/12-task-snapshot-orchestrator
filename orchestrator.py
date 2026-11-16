"""Offline, single-writer orchestration with immutable, one-use review snapshots."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Callable
from uuid import uuid4


class OrchestrationError(RuntimeError):
    pass


class ReviewRequired(OrchestrationError):
    pass


class FeedbackMismatch(OrchestrationError):
    pass


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a nonempty, trimmed string")
    return value


def _canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class Target:
    wire_id: str
    hole_id: str
    grasp_reference: str

    def __post_init__(self):
        for name in ("wire_id", "hole_id", "grasp_reference"):
            _text(getattr(self, name), name)


@dataclass(frozen=True)
class InputDigest:
    name: str
    sha256: str

    def __post_init__(self):
        _text(self.name, "input name")
        if (not isinstance(self.sha256, str) or len(self.sha256) != 64
                or any(char not in "0123456789abcdef" for char in self.sha256)):
            raise ValueError("input digest must be a lowercase SHA-256 value")


def digest_inputs(contents: dict[str, bytes]) -> tuple[InputDigest, ...]:
    """Hash the actual reviewed bytes, not just a filename or reference string."""
    if not isinstance(contents, dict) or not contents:
        raise ValueError("provide the reviewed mapping/action input bytes")
    for name, data in contents.items():
        _text(name, "input name")
        if not isinstance(data, bytes):
            raise ValueError("input contents must be immutable bytes")
    return tuple(InputDigest(name, hashlib.sha256(contents[name]).hexdigest())
                 for name in sorted(contents))


@dataclass(frozen=True)
class Snapshot:
    session_id: str
    review_id: str
    frame_id: str
    scene_revision: str
    targets: tuple[Target, ...]
    review_note: str
    inputs: tuple[InputDigest, ...]

    def __post_init__(self):
        for name in ("session_id", "review_id", "frame_id", "scene_revision", "review_note"):
            _text(getattr(self, name), name)
        # Do not retain a caller-owned list inside a frozen dataclass.
        object.__setattr__(self, "targets", tuple(self.targets))
        object.__setattr__(self, "inputs", tuple(self.inputs))
        if not self.targets or any(not isinstance(item, Target) for item in self.targets):
            raise ValueError("targets must contain Target values")
        for field in ("wire_id", "hole_id"):
            values = [getattr(item, field) for item in self.targets]
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {field} in reviewed selection")
        if not self.inputs or any(not isinstance(item, InputDigest) for item in self.inputs):
            raise ValueError("snapshot requires input content digests")
        if len({item.name for item in self.inputs}) != len(self.inputs):
            raise ValueError("duplicate input names")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical(asdict(self))).hexdigest()


@dataclass(frozen=True)
class Binding:
    task_id: str
    review_id: str
    frame_id: str
    scene_revision: str
    wire_id: str
    hole_id: str
    snapshot_sha256: str

    def __post_init__(self):
        for name, value in asdict(self).items():
            _text(value, name)
        if len(self.snapshot_sha256) != 64 or any(
                char not in "0123456789abcdef" for char in self.snapshot_sha256):
            raise ValueError("snapshot_sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class Dispatch:
    binding: Binding
    grasp_reference: str


@dataclass(frozen=True)
class Feedback:
    binding: Binding
    status: str

    def __post_init__(self):
        if not isinstance(self.binding, Binding):
            raise ValueError("feedback needs a complete Binding")
        if self.status not in {"COMPLETE", "FAILED", "TIMEOUT"}:
            raise ValueError("unsupported terminal feedback status")


class Orchestrator:
    """One writer owns this state file; no device or transport is opened.

    Persistence records identities and outcomes, never a grasp reference or pose.
    A restarted instance requires explicit review of an interrupted task.
    """

    def __init__(self, state_path: str | Path, timeout_seconds: float = 10.0,
                 clock: Callable[[], float] = time.monotonic):
        if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
                or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
            raise ValueError("timeout_seconds must be finite and positive")
        self.path = Path(state_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = float(timeout_seconds)
        self.clock = clock
        self.session_id = uuid4().hex
        self._lock = threading.RLock()
        self._snapshot = None
        self._pending = None
        self._deadline = None
        self._state = self._load()
        if self._state["active"] is not None:
            state = deepcopy(self._state)
            binding = state["active"]
            state["active"] = None
            state["receipts"][binding["task_id"]] = {
                "binding": binding, "status": "INTERRUPTED"}
            state["block"] = {"binding": binding, "reason": "restart interrupted a task"}
            self._commit(state)

    def _load(self):
        if not self.path.exists():
            return {"schema": 1, "used_frames": [], "completed": [], "receipts": {},
                    "active": None, "block": None, "resolutions": []}
        state = json.loads(self.path.read_text(encoding="utf-8"))
        required = {"schema", "used_frames", "completed", "receipts", "active", "block",
                    "resolutions"}
        if not isinstance(state, dict) or set(state) != required or state["schema"] != 1:
            raise ValueError("unsupported state schema")
        for name in ("used_frames", "completed", "resolutions"):
            if not isinstance(state[name], list):
                raise ValueError(f"invalid {name} state")
        for frame in state["used_frames"]:
            _text(frame, "used frame")
        if len(state["used_frames"]) != len(set(state["used_frames"])):
            raise ValueError("duplicate persisted frame")
        for row in state["completed"]:
            if not isinstance(row, dict) or set(row) != {"wire_id", "hole_id"}:
                raise ValueError("invalid completed identity")
            _text(row["wire_id"], "wire_id")
            _text(row["hole_id"], "hole_id")
        if not isinstance(state["receipts"], dict):
            raise ValueError("invalid receipts")
        for task_id, receipt in state["receipts"].items():
            binding = Binding(**receipt["binding"])
            if binding.task_id != task_id or receipt["status"] not in {
                    "COMPLETE", "FAILED", "TIMEOUT", "REVIEW", "INTERRUPTED"}:
                raise ValueError("invalid receipt binding/status")
        if state["active"] is not None:
            Binding(**state["active"])
        if state["block"] is not None:
            Binding(**state["block"]["binding"])
            _text(state["block"]["reason"], "block reason")
        return state

    def _commit(self, state):
        data = _canonical(state)
        fd, temp_name = tempfile.mkstemp(prefix=".state-", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, self.path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        self._state = state

    @property
    def pending(self) -> Binding | None:
        return self._pending.binding if self._pending else None

    @property
    def blocked(self) -> bool:
        return self._state["block"] is not None

    @property
    def completed(self) -> tuple[tuple[str, str], ...]:
        return tuple((row["wire_id"], row["hole_id"]) for row in self._state["completed"])

    def _ready(self):
        if self.blocked:
            raise ReviewRequired(self._state["block"]["reason"])
        if self._pending is not None:
            raise OrchestrationError("a task is already in flight")
        if self._state["active"] is not None:
            raise ReviewRequired("durable intent has no live timer; restart and review")

    def freeze(self, frame_id: str, scene_revision: str, targets,
               review_note: str, *, input_contents: dict[str, bytes]) -> Snapshot:
        """Called after external review. A subsequent freeze replaces the old review."""
        with self._lock:
            self._ready()
            snapshot = Snapshot(self.session_id, uuid4().hex, frame_id, scene_revision,
                                tuple(targets), review_note, digest_inputs(input_contents))
            if frame_id in self._state["used_frames"]:
                raise ReviewRequired("a dispatched frame cannot supply another grasp")
            completed_wires = {wire for wire, _ in self.completed}
            completed_holes = {hole for _, hole in self.completed}
            if any(target.wire_id in completed_wires or target.hole_id in completed_holes
                   for target in snapshot.targets):
                raise ReviewRequired("completed wire or destination cannot be scheduled again")
            self._snapshot = snapshot
            return snapshot

    def dispatch(self, snapshot: Snapshot, wire_id: str, *, snapshot_sha256: str,
                 current_frame_id: str, current_scene_revision: str,
                 current_input_contents: dict[str, bytes]) -> Dispatch:
        with self._lock:
            self._ready()
            if snapshot is not self._snapshot or snapshot.session_id != self.session_id:
                raise ReviewRequired("review was replaced, consumed, or belongs to an old run")
            if snapshot.sha256 != snapshot_sha256:
                raise ReviewRequired("review digest does not match")
            if digest_inputs(current_input_contents) != snapshot.inputs:
                self._snapshot = None
                raise ReviewRequired("mapping/action input contents changed; review again")
            if (snapshot.frame_id, snapshot.scene_revision) != (
                    current_frame_id, current_scene_revision):
                self._snapshot = None
                raise ReviewRequired("frame or scene changed; review a fresh result")
            target = next((item for item in snapshot.targets if item.wire_id == wire_id), None)
            if target is None:
                raise ValueError("wire was not included in the frozen review")
            binding = Binding(uuid4().hex, snapshot.review_id, snapshot.frame_id,
                              snapshot.scene_revision, target.wire_id, target.hole_id,
                              snapshot.sha256)
            state = deepcopy(self._state)
            state["active"] = asdict(binding)
            state["used_frames"].append(snapshot.frame_id)
            self._commit(state)  # Durable intent precedes returning a dispatch to a caller.
            self._snapshot = None
            # Storage latency must not consume the receiver's response budget.
            # An invalid clock after durable intent leaves the task unresolved,
            # rather than issuing an untimed dispatch or permitting a second one.
            now = self.clock()
            if not isinstance(now, (int, float)) or not math.isfinite(now):
                raise ValueError("clock must return a finite monotonic value")
            deadline = now + self.timeout_seconds
            if not math.isfinite(deadline):
                raise ValueError("response deadline must be finite")
            self._pending = Dispatch(binding, target.grasp_reference)
            self._deadline = deadline
            return self._pending

    def _finish(self, status: str, reason: str = ""):
        binding = self._pending.binding
        state = deepcopy(self._state)
        state["active"] = None
        state["receipts"][binding.task_id] = {"binding": asdict(binding), "status": status}
        if status == "COMPLETE":
            state["completed"].append({"wire_id": binding.wire_id, "hole_id": binding.hole_id})
        else:
            state["block"] = {"binding": asdict(binding), "reason": reason or status}
        self._commit(state)
        self._pending = None
        self._deadline = None

    def check_timeout(self) -> bool:
        with self._lock:
            if self._pending is None:
                return False
            now = self.clock()
            if not isinstance(now, (int, float)) or not math.isfinite(now):
                raise ValueError("clock must return a finite monotonic value")
            if now >= self._deadline:
                self._finish("TIMEOUT", "task deadline elapsed; verify the actual outcome")
                return True
            return False

    def feedback(self, feedback: Feedback) -> str:
        """Return APPLIED or DUPLICATE; uncertain, late and mismatched events raise."""
        with self._lock:
            if not isinstance(feedback, Feedback):
                raise ValueError("a complete Feedback value is required")
            receipt = self._state["receipts"].get(feedback.binding.task_id)
            if receipt is not None:
                if receipt["binding"] != asdict(feedback.binding):
                    if self._pending is not None:
                        self._finish("REVIEW", "feedback identity mismatch")
                    raise FeedbackMismatch("terminal receipt has different binding")
                if receipt["status"] == "COMPLETE" and feedback.status == "COMPLETE":
                    return "DUPLICATE"
                raise ReviewRequired("task is terminal; late feedback cannot change its outcome")
            if self._pending is None:
                raise FeedbackMismatch("no matching task is in flight")
            if self.check_timeout():
                raise ReviewRequired("feedback arrived at or after the deadline")
            if feedback.binding != self._pending.binding:
                self._finish("REVIEW", "feedback task, review, frame, scene, wire, hole or digest differs")
                raise FeedbackMismatch("feedback does not match the full active binding")
            self._finish(feedback.status)
            return "APPLIED"

    def resolve_after_review(self, *, task_id: str, confirmed_completed: bool, note: str):
        """An external operator resolves uncertainty; this never restores a grasp."""
        with self._lock:
            _text(note, "review note")
            if type(confirmed_completed) is not bool:
                raise ValueError("confirmed_completed must be a boolean")
            block = self._state["block"]
            if block is None or block["binding"]["task_id"] != task_id:
                raise ReviewRequired("no matching interrupted or blocked task")
            state = deepcopy(self._state)
            binding = block["binding"]
            if confirmed_completed:
                pair = {"wire_id": binding["wire_id"], "hole_id": binding["hole_id"]}
                if pair not in state["completed"]:
                    state["completed"].append(pair)
            state["resolutions"].append({"task_id": task_id,
                                         "confirmed_completed": confirmed_completed, "note": note})
            state["block"] = None
            self._commit(state)
            self._snapshot = None
