"""Regression tests for deploying a trained TCN and its normalization statistics."""

import json
import shlex
import sys
from pathlib import Path

import numpy as np
import onnxruntime
import pytest
import torch
from datasets import Dataset
from omegaconf import OmegaConf

from scripts import export_onnx, train_tcn

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
pytestmark = [
    pytest.mark.filterwarnings("ignore:.*weight_norm.*:FutureWarning"),
    pytest.mark.filterwarnings("ignore:.*LeafSpec.*is deprecated:FutureWarning"),
]


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    run = tmp_path / "training run"
    cache = tmp_path / "cache"
    cfg = OmegaConf.load(CONFIGS / "config.yaml")
    cfg.model = OmegaConf.load(CONFIGS / "model/tcn.yaml")
    cfg.training.output_dir = str(run)
    cfg.training.num_epochs = 1
    participants = ["BT01", "BT02", "BT03"]
    cfg.data = {
        "hf_repo": "test/local",
        "cache_dir": str(cache),
        "splits": {"train": ["BT01"], "val": ["BT02"], "test": ["BT03"]},
        "num_workers": 0,
        "preprocessing": {"normalize": True},
    }
    rng = np.random.default_rng(32)
    trials = Dataset.from_dict(
        {
            "participant": participants,
            "trial_name": ["walk"] * 3,
            "mass_kg": [70.0] * 3,
            "sequence_length": [32] * 3,
            "imu_features": rng.normal(size=(3, 32, 24)).tolist(),
            "angle_features": rng.normal(size=(3, 32, 4)).tolist(),
            "moment_targets": rng.normal(size=(3, 32, 4)).tolist(),
        }
    )
    monkeypatch.setattr(
        "exoskeleton_ml.data.datasets.download_and_cache_dataset", lambda **kw: trials
    )
    monkeypatch.setattr(train_tcn, "get_device", lambda: torch.device("cpu"))
    train_tcn.train(cfg)
    cached_stats = cache / "normalization_stats.json"
    assert json.loads((run / cached_stats.name).read_text()) == json.loads(cached_stats.read_text())
    cached_stats.write_text("{}")
    return run / "best_model.pt"


@pytest.mark.parametrize("window", [187, 64])
def test_export_checkpoint(checkpoint, tmp_path, monkeypatch, capsys, window):
    run = checkpoint.parent
    output = tmp_path / "pi bundle" / "model.onnx"
    if window != 187:
        config = OmegaConf.load(run / "config.yaml")
        config.model.effective_history = window
        OmegaConf.save(config, run / "config.yaml")
        output = run / "model.onnx"

    stats = (run / "normalization_stats.json").read_bytes()
    monkeypatch.setattr(
        sys, "argv", ["export_onnx", "--model-path", str(checkpoint), "--output", str(output)]
    )
    export_onnx.main()
    log = capsys.readouterr().out
    assert ("Pi WINDOW_LEN of 187" in log) == (window != 187)
    exported_stats = output.with_name("normalization_stats.json")
    assert exported_stats.read_bytes() == stats
    assert shlex.split(log.splitlines()[-1]) == [
        "scp",
        str(output),
        str(exported_stats),
        "pi@<PI_IP>:~/exoskeleton-ai/outputs/",
    ]
    if window == 187:
        assert set(output.parent.iterdir()) == {output, exported_stats}
    session = onnxruntime.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    inputs = np.zeros((1, window, 28), dtype=np.float32)
    (predictions,) = session.run(None, {session.get_inputs()[0].name: inputs})
    assert predictions.shape == (1, window, 4)


def test_missing_stats(tmp_path):
    with pytest.raises(FileNotFoundError, match="normalization_stats.json"):
        export_onnx.export(tmp_path / "checkpoint.pt", tmp_path / "model.onnx")


def test_mismatch_preserves_previous_export(checkpoint, tmp_path, monkeypatch):
    output = tmp_path / "model.onnx"
    stats = tmp_path / "normalization_stats.json"
    output.write_bytes(b"previous model")
    stats.write_bytes(b"previous stats")
    monkeypatch.setattr(
        onnxruntime.InferenceSession,
        "run",
        lambda *args: [np.full((1, 187, 4), np.nan, dtype=np.float32)],
    )
    with pytest.raises(AssertionError):
        export_onnx.export(checkpoint, output)
    assert output.read_bytes() == b"previous model"
    assert stats.read_bytes() == b"previous stats"


def test_missing_checkpoint_config(tmp_path):
    model_path = tmp_path / "best_model.pt"
    torch.save({}, model_path)
    with pytest.raises(ValueError, match="config.yaml"):
        export_onnx.load_model(model_path)
