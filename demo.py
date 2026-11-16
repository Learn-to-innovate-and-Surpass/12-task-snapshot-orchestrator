"""Synthetic data only. No device, camera, network or motion command."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from orchestrator import Feedback, FeedbackMismatch, Orchestrator, Target


def main():
    with TemporaryDirectory() as directory:
        clock = [0.0]
        path = Path(directory) / "state.json"
        flow = Orchestrator(path, clock=lambda: clock[0])
        inputs = {"mapping.json": b'{"W001":"H001","W002":"H002"}',
                  "mock-action.json": b'{"mode":"synthetic-only","version":1}'}
        snapshot = flow.freeze("synthetic-frame-1", "scene-1",
                               [Target("W001", "H001", "synthetic-grasp-1")],
                               "人工审核示例：目标和孔位一致", input_contents=inputs)
        dispatch = flow.dispatch(snapshot, "W001", snapshot_sha256=snapshot.sha256,
                                 current_frame_id="synthetic-frame-1",
                                 current_scene_revision="scene-1", current_input_contents=inputs)
        print("成功核销：", flow.feedback(Feedback(dispatch.binding, "COMPLETE")))
        print("重复成功：", flow.feedback(Feedback(dispatch.binding, "COMPLETE")))

        snapshot = flow.freeze("synthetic-frame-2", "scene-2",
                               [Target("W002", "H002", "synthetic-grasp-2")],
                               "第二帧重新计算并审核", input_contents=inputs)
        dispatch = flow.dispatch(snapshot, "W002", snapshot_sha256=snapshot.sha256,
                                 current_frame_id="synthetic-frame-2",
                                 current_scene_revision="scene-2", current_input_contents=inputs)
        try:
            flow.feedback(Feedback(replace(dispatch.binding, hole_id="H999"), "COMPLETE"))
        except FeedbackMismatch:
            print("错孔反馈被拒绝，进入复核：", flow.blocked)

        # A new run reads durable identities/outcomes, never the old grasp.
        reopened = Orchestrator(path, clock=lambda: clock[0])
        print("重启后已完成身份：", reopened.completed)
        print("重启后待发抓姿：", reopened.pending)
        print("需要现场确认结果：", reopened.blocked)


if __name__ == "__main__":
    main()
