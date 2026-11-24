"""Event stream tests: the JSONL contract mirrored in packages/core/src/events.ts."""

from __future__ import annotations

import io
import json

from supertuner_engine.events import EVENT_TYPES, Emitter, NullEmitter, parse_event_line
from supertuner_engine.schema import PlanStep


class RecordingStream(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.flushes = 0

    def flush(self) -> None:
        self.flushes += 1
        super().flush()


def _lines(stream: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


def _step() -> PlanStep:
    return PlanStep(id="train", kind="train", title="训练", description="demo", command=None)


def test_every_event_is_one_json_object_per_line():
    stream = RecordingStream()
    emit = Emitter(stream, "run-1", clock=lambda: 1700000000000)
    emit.run_start(None)
    emit.phase_start(_step())
    emit.phase_end(_step(), "completed", exit_code=0)
    emit.log("info", "hello")
    emit.step(3, loss=1.5, lr=1e-4, grad_norm=0.5, tokens_per_s=120.0, vram_peak_mib=7500.0, elapsed_s=0.4)
    emit.eval(3, {"loss": 1.4})
    emit.checkpoint(3, "/tmp/adapter_model.safetensors", 1024)
    emit.error("boom", "detail")
    emit.run_end("completed", 1)

    raw = [line for line in stream.getvalue().splitlines() if line]
    assert len(raw) == 9
    events = _lines(stream)
    assert {event["ts"] for event in events} == {1700000000000}
    assert [event["type"] for event in events] == [
        "run_start",
        "phase_start",
        "phase_end",
        "log",
        "step",
        "eval",
        "checkpoint",
        "error",
        "run_end",
    ]
    assert len(stream.getvalue().splitlines()) == len(raw)


def test_every_line_is_flushed():
    stream = RecordingStream()
    emit = Emitter(stream, "run-1")
    emit.log("info", "a")
    emit.log("info", "b")
    assert stream.flushes >= 2


def test_run_start_matches_wire_schema():
    stream = io.StringIO()
    emit = Emitter(stream, "run-42")
    emit.run_start(None, resumed_from_step=12)
    event = _lines(stream)[0]
    assert event["type"] == "run_start"
    assert event["run_id"] == "run-42"
    assert event["plan"] is None
    assert event["resumed_from_step"] == 12


def test_phase_events_match_wire_schema():
    stream = io.StringIO()
    emit = Emitter(stream, "run")
    emit.phase_start(_step())
    emit.phase_end(_step(), "failed", exit_code=3, reason="step failed")
    started, ended = _lines(stream)
    assert set(started) == {"ts", "type", "step_id", "phase", "title"}
    assert started["phase"] == "train"
    assert set(ended) == {"ts", "type", "step_id", "phase", "status", "exit_code", "reason"}
    assert ended["status"] == "failed"
    assert ended["exit_code"] == 3


def test_step_event_carries_grad_norm_and_vram():
    stream = io.StringIO()
    emit = Emitter(stream, "run")
    emit.step(7, loss=2.0, lr=0.01, grad_norm=1.25, tokens_per_s=99.5, vram_peak_mib=8123.0, elapsed_s=0.3)
    event = _lines(stream)[0]
    assert event["step"] == 7
    assert event["grad_norm"] == 1.25
    assert event["vram_peak_mib"] == 8123.0
    assert event["tokens_per_s"] == 99.5


def test_step_count_tracks_highest_step():
    stream = io.StringIO()
    emit = Emitter(stream, "run")
    emit.step(2)
    emit.step(9)
    emit.step(5)
    assert emit.step_count == 9
    emit.count_step(11)
    assert emit.step_count == 11


def test_run_end_defaults_to_step_count():
    stream = io.StringIO()
    emit = Emitter(stream, "run")
    emit.step(4)
    emit.run_end("completed")
    event = _lines(stream)[-1]
    assert event["type"] == "run_end"
    assert event["status"] == "completed"
    assert event["steps"] == 4
    assert event["reason"] is None


def test_parse_event_line_roundtrip():
    stream = io.StringIO()
    emit = Emitter(stream, "run")
    emit.log("warn", "磁盘空间不足")
    line = stream.getvalue().strip()
    parsed = parse_event_line(line)
    assert parsed is not None
    assert parsed["type"] == "log"
    assert parsed["message"] == "磁盘空间不足"


def test_parse_event_line_rejects_non_events():
    assert parse_event_line("") is None
    assert parse_event_line("not json") is None
    assert parse_event_line('{"type": "mystery", "ts": 1}') is None
    assert parse_event_line('{"type": "log", "message": "no ts"}') is None
    assert parse_event_line('{"ts": "1700", "type": "log", "level": "info", "message": "bad ts"}') is None


def test_forwarded_events_keep_their_own_timestamp():
    stream = io.StringIO()
    emit = Emitter(stream, "run", clock=lambda: 1)
    emit.emit({"ts": 999, "type": "log", "level": "info", "message": "child"})
    event = _lines(stream)[0]
    assert event["ts"] == 999


def test_event_types_constant_lists_required_v01_types():
    assert {
        "run_start",
        "phase_start",
        "step",
        "eval",
        "checkpoint",
        "log",
        "error",
        "phase_end",
        "run_end",
        "memory_report",
        "eval_report",
    } <= EVENT_TYPES


def test_memory_report_event_round_trips():
    stream = io.StringIO()
    emit = Emitter(stream, "run", clock=lambda: 1234)
    emit.memory_report(
        {
            "planned_mib": 1751.5,
            "measured_peak_mib": 1787.0,
            "delta_mib": 35.5,
            "delta_pct": 2.03,
            "device": "cuda",
            "family": "qwen3-0.6b",
            "stage": "sft",
            "adapter": "lora",
        }
    )
    event = _lines(stream)[0]
    assert event["type"] == "memory_report"
    assert event["ts"] == 1234
    assert event["planned_mib"] == 1751.5
    assert event["measured_peak_mib"] == 1787.0
    assert parse_event_line(stream.getvalue().strip()) is not None


def test_null_emitter_accepts_everything():
    emit = NullEmitter()
    emit.run_start(None)
    emit.step(1, loss=1.0)
    emit.phase_start(_step())
    emit.phase_end(_step(), "completed")
    emit.log("info", "x")
    emit.error("x")
    emit.run_end("failed")
    emit.memory_report({"planned_mib": None, "measured_peak_mib": None})
    emit.eval_report({"status": "ok", "path": None, "report": {}})
    assert emit.step_count == 0


def test_eval_report_event_round_trips():
    stream = io.StringIO()
    emit = Emitter(stream, "run", clock=lambda: 1234)
    emit.eval_report({"status": "ok", "path": "/tmp/eval_report.json", "report": {"summary": {"train": {"steps": 10}}}})
    event = _lines(stream)[0]
    assert event["type"] == "eval_report"
    assert event["ts"] == 1234
    assert event["status"] == "ok"
    assert event["path"] == "/tmp/eval_report.json"
    assert event["report"]["summary"]["train"]["steps"] == 10
    assert parse_event_line(stream.getvalue().strip()) is not None


def test_unicode_is_not_escaped():
    stream = io.StringIO()
    emit = Emitter(stream, "run")
    emit.log("info", "显存不足：正在交换块")
    assert "显存不足" in stream.getvalue()