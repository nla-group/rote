"""Parameter selection and final-checkpoint evaluation at a fixed update budget."""

import math
import time

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

from rote_benchmark.benchmark import evaluate_loader, make_optimizer, prepare_data, rollout
from rote_benchmark.models import TRANSFORMER_STYLE_MODELS, get_model

from .design import HORIZON, SEQUENCE_LENGTH, WINDOW


TARGET_PARAMS = 185_000
PARAMETER_TOLERANCE = 0.05
EPOCHS = 40


def build_model(config, window):
    model, n = config["model"], config["symbols"]
    return get_model(model, n, config["units"], n, num_layers=config["layers"],
                     d_model=config["d_model"], window_size=window)


def trainable_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def matched_widths(models, alphabets, layers, window):
    rows = []
    for n in alphabets:
        for model in models:
            attention_width = model in TRANSFORMER_STYLE_MODELS
            grid = range(48, 105, 8) if attention_width else range(100, 185, 2)
            candidates = []
            for width in grid:
                config = {
                    "model": model, "symbols": n, "layers": layers,
                    "units": 4 * width if attention_width else width,
                    "d_model": width if attention_width else 0,
                }
                count = trainable_parameters(build_model(config, window))
                error = (count - TARGET_PARAMS) / TARGET_PARAMS
                candidates.append({**config, "model_params": count,
                                   "target_params": TARGET_PARAMS, "parameter_rel_error": error})
            nearest = min(candidates, key=lambda row: abs(row["parameter_rel_error"]))
            if abs(nearest["parameter_rel_error"]) > PARAMETER_TOLERANCE:
                raise RuntimeError(
                    f"{model} at n={n} has no width within {PARAMETER_TOLERANCE:.0%} of "
                    f"{TARGET_PARAMS:,} parameters; nearest is {nearest['model_params']:,}"
                )
            rows.append(nearest)
    return rows


def evaluate_budget(row, config, protocol, *, epochs=EPOCHS, window=WINDOW,
                    sequence_length=SEQUENCE_LENGTH, horizon=HORIZON, initialization):
    """Train with shared batch ordering and evaluate the final checkpoint."""
    device = torch.device(protocol.device)
    torch.manual_seed(initialization)
    data = prepare_data(row["seed_string"], sequence_length, window, horizon, device)
    model = build_model(config, window).to(device)
    if trainable_parameters(model) != config["model_params"]:
        raise RuntimeError("The fitted model does not match the registered parameter count")
    # Model initialization and dropout may use different random streams. The
    # sampler has its own generator so batch ordering stays paired by string.
    batch_rng = torch.Generator().manual_seed(initialization)
    loaders = {
        split: DataLoader(
            TensorDataset(data[f"X_{split}"], data[f"y_{split}"]),
            batch_size=protocol.batch_size, shuffle=(split == "train"),
            generator=batch_rng if split == "train" else None,
        )
        for split in ("train", "valid", "test")
    }
    optimizer = make_optimizer(model, protocol.optimizer, protocol.learning_rate,
                               protocol.weight_decay)
    scheduler = StepLR(optimizer, step_size=protocol.lr_step_size, gamma=protocol.lr_gamma)
    criterion = nn.CrossEntropyLoss()
    updates = tokens = 0
    on_gpu = device.type == "cuda"
    if on_gpu:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for batch, labels in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at update {updates + 1}")
            loss.backward()
            optimizer.step()
            updates += 1
            tokens += batch.shape[0] * batch.shape[1]
        scheduler.step()
    train_windows = len(loaders["train"].dataset)
    if updates != epochs * len(loaders["train"]) or tokens != epochs * train_windows * window:
        raise RuntimeError("The observed update or token count differs from the fixed budget")
    valid_loss, valid_accuracy = evaluate_loader(model, loaders["valid"], criterion)
    test_loss, test_accuracy = evaluate_loader(model, loaders["test"], criterion)
    if not (math.isfinite(valid_loss) and math.isfinite(test_loss)):
        raise FloatingPointError("Non-finite loss at the final checkpoint")
    prediction = rollout(model, data["initial_context"], data["target"], data["symbols"],
                         window, device)
    if on_gpu:
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return {
        **row, **config, "random_seed": initialization, "epochs": epochs,
        "optimizer": protocol.optimizer, "learning_rate": protocol.learning_rate,
        "weight_decay": protocol.weight_decay, "batch_size": protocol.batch_size,
        "window_size": window, "sequence_length": data["sequence_length"],
        "forecast_horizon": horizon, "train_windows": train_windows,
        "valid_windows": len(loaders["valid"].dataset), "test_windows": len(loaders["test"].dataset),
        "optimizer_updates": updates, "input_tokens": tokens, "train_time": elapsed,
        "time_per_epoch": elapsed / epochs,
        "memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if on_gpu else 0.0,
        "valid_loss": valid_loss, "valid_accuracy": valid_accuracy,
        "test_loss": test_loss, "test_accuracy": test_accuracy,
        "DL": prediction["dl"], "JW": prediction["jw"], "forecast": prediction["forecast"],
        "target_string": data["target"], "exact": int(prediction["forecast"] == data["target"]),
    }
