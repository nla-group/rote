"""Public PyTorch adapter for user-defined next-symbol predictors.

This adapter reuses the paper experiment's data split, optimizer, early
stopping, teacher-forced metrics, and closed-loop rollout without changing
the archived experiment runner.
"""

from dataclasses import dataclass, replace
import math
import random
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .benchmark import evaluate_loader, prepare_data, rollout, train_and_evaluate
from .generation import ALPHABET, _shortest_period, lzw_complexity


ModelFactory = Callable[[int], nn.Module]


@dataclass(frozen=True)
class EvaluationConfig:
    """Settings for fitting one model to one repeated symbolic seed.

    ``sequence_length`` includes the withheld ``forecast_horizon`` symbols.
    ``seed`` controls factory initialization and training order. A prebuilt
    module retains the caller's initial weights.
    """

    sequence_length: int = 320
    window_size: int = 12
    forecast_horizon: int = 24
    batch_size: int = 32
    max_epochs: int = 12
    patience: int = 4
    stopping_loss: float = 0.05
    optimizer: str = "AdamW"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    lr_step_size: int = 100
    lr_gamma: float = 0.5
    device: str = "cpu"
    seed: int = 3407

    def __post_init__(self) -> None:
        for field in (
            "sequence_length", "window_size", "forecast_horizon", "batch_size",
            "max_epochs", "patience", "lr_step_size",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if self.sequence_length < self.window_size + self.forecast_horizon + 12:
            raise ValueError("sequence_length must be at least window_size + forecast_horizon + 12")
        if self.optimizer not in {"Adam", "AdamW"}:
            raise ValueError("optimizer must be 'Adam' or 'AdamW'")
        for field in ("learning_rate", "weight_decay", "stopping_loss", "lr_gamma"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{field} must be finite")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay nonnegative")
        if self.stopping_loss < 0 or self.lr_gamma <= 0:
            raise ValueError("stopping_loss must be nonnegative and lr_gamma positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer between 0 and 2**32 - 1")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer between 0 and 2**32 - 1")
        torch.device(self.device)


def _check_model_contract(model: nn.Module, data: dict, *, require_trainable: bool = True) -> None:
    parameters = list(model.parameters())
    if require_trainable and not any(parameter.requires_grad for parameter in parameters):
        raise ValueError("model must have at least one trainable parameter")
    model.eval()
    with torch.no_grad():
        x = data["X_train"][:2]
        logits = model(x)
    expected = (len(x), len(data["symbols"]))
    if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != expected:
        shape = getattr(logits, "shape", type(logits).__name__)
        raise ValueError(f"model(x) must return logits of shape {expected}, got {shape}")
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("model(x) must return finite floating-point logits")


def evaluate_model(
    model: nn.Module | ModelFactory,
    seed_string: str,
    *,
    config: EvaluationConfig | None = None,
    name: str | None = None,
    run: int = 0,
) -> dict:
    """Train and evaluate one custom ``torch.nn.Module`` on a symbolic seed.

    ``model`` may be a prebuilt module or a ``factory(n_symbols)`` returning a
    fresh module. Its forward pass receives float one-hot windows with shape
    ``(batch, window_size, n_symbols)`` and must return unnormalized floating
    logits with shape ``(batch, n_symbols)``. ROTE repeats ``seed_string`` to
    ``sequence_length`` and withholds the last ``forecast_horizon`` symbols.

    The returned dict is CSV-ready. It includes one-step loss and accuracy,
    rollout DL/JW, target and forecast strings, total parameter count and trainable subset, and
    timing. ``memory_mb`` is zero on CPU (only CUDA allocation is tracked).
    The supplied module is fitted in place; RNG states are restored afterward.
    """
    config = config or EvaluationConfig()
    if not isinstance(config, EvaluationConfig):
        raise TypeError("config must be an EvaluationConfig")
    if not isinstance(seed_string, str) or not seed_string:
        raise ValueError("seed_string must be a nonempty string")
    if any(symbol not in ALPHABET for symbol in seed_string):
        raise ValueError("seed_string must use ASCII letters from the ROTE alphabet")
    if isinstance(run, bool) or not isinstance(run, int) or run < 0:
        raise ValueError("run must be a nonnegative integer")

    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but CUDA is unavailable")
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    mps_module = getattr(torch, "mps", None)
    mps_state = (
        mps_module.get_rng_state()
        if mps_module is not None and torch.backends.mps.is_available()
        and hasattr(mps_module, "get_rng_state")
        else None
    )
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            random.seed(config.seed)
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)
            if cuda_devices:
                torch.cuda.manual_seed_all(config.seed)
            data = prepare_data(
                seed_string, config.sequence_length, config.window_size,
                config.forecast_horizon, device,
            )
            if not isinstance(model, nn.Module):
                if not callable(model):
                    raise TypeError("model must be an nn.Module or a factory(n_symbols)")
                model = model(len(data["symbols"]))
            if not isinstance(model, nn.Module):
                raise TypeError("model factory must return a torch.nn.Module")
            model = model.to(device)
            _check_model_contract(model, data)
            n_params = sum(p.numel() for p in model.parameters())
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            metrics = train_and_evaluate(
                model, data, config, config.optimizer, config.learning_rate,
                config.weight_decay, config.batch_size,
            )
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        if mps_state is not None:
            mps_module.set_rng_state(mps_state)
    complexity = lzw_complexity(_shortest_period(seed_string))
    return {
        "model": name or model.__class__.__name__,
        "symbols": len(data["symbols"]),
        "complexity": complexity,
        "seed_string": seed_string,
        "target_string": data["target"],
        "sequence_length": data["sequence_length"],
        "window_size": config.window_size,
        "forecast_horizon": config.forecast_horizon,
        "optimizer": config.optimizer,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "batch_size": config.batch_size,
        "run": run,
        "seed": config.seed,
        "model_params": n_params,
        "trainable_params": n_trainable,
        "model_size_m": n_params / 1e6,
        **metrics,
    }


def evaluate_models(
    models: Mapping[str, ModelFactory], seed_string: str, *,
    config: EvaluationConfig | None = None, runs: int = 1,
) -> pd.DataFrame:
    """Evaluate named fresh-model factories under common settings.

    Each model/run pair is constructed afresh with seed ``config.seed + run``.
    The DataFrame has one row per fit and can be saved with ``to_csv``.
    """
    if not models:
        raise ValueError("models must contain at least one factory")
    if isinstance(runs, bool) or not isinstance(runs, int) or runs < 1:
        raise ValueError("runs must be a positive integer")
    config = config or EvaluationConfig()
    rows = []
    for run in range(runs):
        run_config = replace(config, seed=config.seed + run)
        for name, factory in models.items():
            if not name or not callable(factory):
                raise ValueError("models must map nonempty names to factories")
            rows.append(evaluate_model(factory, seed_string, config=run_config, name=name, run=run))
    return pd.DataFrame(rows)


def score_model(
    model: nn.Module,
    seed_string: str,
    *,
    config: EvaluationConfig | None = None,
    name: str | None = None,
) -> dict:
    """Score a trained model without changing its weights.

    Uses the same chronological test windows and withheld rollout target as
    :func:`evaluate_model`, but performs no optimization or early stopping.
    The checkpoint must use the same alphabet order and window size. This
    method moves the module to ``config.device`` and leaves it in eval mode.
    Training-related config fields are ignored. The returned record can be
    plotted with either generic visualization function.
    """
    config = config or EvaluationConfig()
    if not isinstance(config, EvaluationConfig):
        raise TypeError("config must be an EvaluationConfig")
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a trained torch.nn.Module")
    if not isinstance(seed_string, str) or not seed_string:
        raise ValueError("seed_string must be a nonempty string")
    if any(symbol not in ALPHABET for symbol in seed_string):
        raise ValueError("seed_string must use ASCII letters from the ROTE alphabet")

    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but CUDA is unavailable")
    data = prepare_data(
        seed_string, config.sequence_length, config.window_size,
        config.forecast_horizon, device,
    )
    model = model.to(device)
    _check_model_contract(model, data, require_trainable=False)
    test_loader = DataLoader(
        TensorDataset(data["X_test"], data["y_test"]),
        batch_size=config.batch_size,
    )
    test_loss, test_accuracy = evaluate_loader(model, test_loader, nn.CrossEntropyLoss())
    roll = rollout(
        model, data["initial_context"], data["target"], data["symbols"],
        config.window_size, device,
    )
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "model": name or model.__class__.__name__,
        "symbols": len(data["symbols"]),
        "complexity": lzw_complexity(_shortest_period(seed_string)),
        "seed_string": seed_string,
        "target_string": data["target"],
        "sequence_length": data["sequence_length"],
        "window_size": config.window_size,
        "forecast_horizon": config.forecast_horizon,
        "model_params": n_params,
        "trainable_params": n_trainable,
        "model_size_m": n_params / 1e6,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "DL": roll["dl"],
        "JW": roll["jw"],
        "forecast": roll["forecast"],
    }


def run_benchmark(
    models: Mapping[str, ModelFactory],
    seed_string: str | Mapping[str, str],
    *,
    config: EvaluationConfig | None = None,
    runs: int = 1,
    output_dir: str | Path | None = None,
    metadata: dict | None = None,
):
    """Evaluate models on one or more seeds and persist their full records.

    A fresh directory is created at ``output_dir`` or beneath
    ``./rote_runs`` (override the base with ``ROTE_RUNS_DIR``). Each completed
    fit is appended to ``records.jsonl`` and atomically reflected in
    ``results.csv``. Pass a mapping of seed IDs to strings for a measured
    LZW-complexity sweep; each row includes its ``seed_id``. ``config.json``, ``manifest.json``, and ``events.jsonl``
    record settings and progress. Existing explicit directories are never
    overwritten. Returns :class:`rote_benchmark.artifacts.SavedRun`.
    """
    from dataclasses import asdict
    import platform

    from . import __version__
    from .artifacts import SCHEMA_VERSION, _RunWriter, load_run

    config = config or EvaluationConfig()
    if not isinstance(config, EvaluationConfig):
        raise TypeError("config must be an EvaluationConfig")
    if isinstance(seed_string, str):
        seeds = {"default": seed_string}
    elif isinstance(seed_string, Mapping) and seed_string:
        seeds = dict(seed_string)
    else:
        raise ValueError("seed_string must be a nonempty string or seed mapping")
    for seed_id, value in seeds.items():
        if not isinstance(seed_id, str) or not seed_id.strip():
            raise ValueError("seed IDs must be nonempty strings")
        if not isinstance(value, str) or not value:
            raise ValueError("every seed must be a nonempty string")
        if any(symbol not in ALPHABET for symbol in value):
            raise ValueError("seed strings must use ASCII letters from the ROTE alphabet")
    if not isinstance(models, Mapping) or not models or any(
        not isinstance(name, str) or not name.strip() or not callable(factory)
        for name, factory in models.items()
    ):
        raise ValueError("models must map nonempty string names to factories")
    if isinstance(runs, bool) or not isinstance(runs, int) or runs < 1:
        raise ValueError("runs must be a positive integer")
    if config.seed + runs - 1 >= 2**32:
        raise ValueError("config.seed + runs - 1 must be below 2**32")
    if metadata is not None and not isinstance(metadata, dict):
        raise TypeError("metadata must be a JSON-serializable dict")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "evaluation": asdict(config),
        "seed_strings": seeds,
        "models": list(models),
        "runs": runs,
        "metadata": metadata or {},
        "software": {
            "rote_bench": __version__,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
    }
    writer = _RunWriter(payload, len(seeds) * len(models) * runs, output_dir)
    try:
        for seed_id, value in seeds.items():
            for run in range(runs):
                run_config = replace(config, seed=config.seed + run)
                for name, factory in models.items():
                    result = evaluate_model(factory, value, config=run_config, name=name, run=run)
                    result["seed_id"] = seed_id
                    writer.append(result)
        writer.finish()
    except BaseException as error:
        writer.fail(error)
        raise
    return load_run(writer.path)
