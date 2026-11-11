import json

import pytest

from sarship.benchmark import (
    accuracy_from_robustness,
    benchmark_markdown,
    equivalence,
    plot_accuracy_vs_speed,
    summarize_timings,
)


def test_summarize_timings_median_p90_and_total():
    speeds = [{"preprocess": 1.0, "inference": float(i), "postprocess": 2.0} for i in range(1, 11)]
    s = summarize_timings(speeds)
    assert s["inference"]["median"] == pytest.approx(5.5)
    assert s["inference"]["p90"] == pytest.approx(9.1)
    assert s["total"]["median"] == pytest.approx(8.5)


def test_accuracy_from_robustness(tmp_path):
    path = tmp_path / "robustness.json"
    assert accuracy_from_robustness(path) is None
    results = {"clean": {"all": {"map50_95": {"point": 0.7}}},
               "blur_1": {"all": {"map50_95": {"point": 0.6}}},
               "blur_2": {"all": {"map50_95": {"point": 0.4}}}}
    path.write_text(json.dumps({"results": results}))
    assert accuracy_from_robustness(path) == {"clean": 0.7, "degraded_mean": pytest.approx(0.5)}


def test_equivalence_tolerance():
    pt = {"map50": 0.98, "map50_95": 0.712}
    assert equivalence(pt, {"map50": 0.98, "map50_95": 0.7125}, 0.002)["ok"]
    bad = equivalence(pt, {"map50": 0.9, "map50_95": 0.69}, 0.002)
    assert not bad["ok"] and bad["diff_map50_95"] == pytest.approx(-0.022)


def row(name, gpu=True):
    t = {"stages": {"inference": {"median": 4.0, "p90": 5.0}, "total": {"median": 7.0, "p90": 9}},
         "wall_ms": 10.0, "fps": 100.0}
    timing = {"pytorch_cpu": t, "onnx_cpu": t}
    if gpu:
        timing = {"pytorch_gpu": t, **timing}
    return {"name": name, "params_m": 2.59, "gflops": 6.4, "pt_mb": 5.5, "onnx_mb": 10.1,
            "accuracy": {"clean": 0.712, "degraded_mean": 0.66},
            "equivalence": {"diff_map50_95": 0.0003, "ok": True}, "timing": timing}


def test_markdown_and_plot(tmp_path):
    rows = [row("robust_yolo11n"), row("robust_yolo11s")]
    md = benchmark_markdown(rows, {"gpu": "Tesla T4", "cpu": "Xeon", "cpu_threads": 2})
    assert "| robust_yolo11n | 2.59 | 6.4 |" in md and "+0.0003 ok" in md
    assert md.count("pytorch_gpu") == 2 and md.count("onnx_cpu") == 2
    assert plot_accuracy_vs_speed(rows, tmp_path / "a.png", "pytorch_gpu").stat().st_size > 0
