from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from picnnt.data.preprocessing import CellTrajectory, Scaler, WindowDataset, cell_to_features, sequential_eval_batches, split_cells
from picnnt.evaluate import Metrics, compute_metrics
from picnnt.models.baselines import CNNBaseline, LSTMBaseline
from picnnt.models.losses import LossWeights
from picnnt.models.pi_cnn_transformer import PICNNTransformer
from picnnt.train import TrainConfig, _batched_gather, train_model
from picnnt.utils import set_seed


@dataclasses.dataclass
class DataBundle:
    train_ds: WindowDataset
    val_ds: WindowDataset
    test_ds: WindowDataset
    scaler: Scaler
    cells: list[CellTrajectory]


def build_data_from_cells(
    cells: list[CellTrajectory], split_seed: int = 0, seq_len: int = 50, stride: int = 1,
    smooth_resistance: bool = False,
) -> DataBundle:
    splits = split_cells(cells, seed=split_seed)
    train_feats = np.concatenate(
        [cell_to_features(c, smooth_resistance=smooth_resistance) for c in splits.train], axis=0,
    )
    scaler = Scaler.fit(train_feats)
    train_ds = WindowDataset(splits.train, scaler, seq_len=seq_len, stride=stride, smooth_resistance=smooth_resistance)
    val_ds = WindowDataset(splits.val, scaler, seq_len=seq_len, stride=stride, smooth_resistance=smooth_resistance)
    test_ds = WindowDataset(splits.test, scaler, seq_len=seq_len, stride=stride, smooth_resistance=smooth_resistance)
    return DataBundle(train_ds=train_ds, val_ds=val_ds, test_ds=test_ds, scaler=scaler, cells=cells)


def build_real_data(
    name: str, processed_dir: str, split_seed: int = 0, seq_len: int = 50, stride: int = 1,
    smooth_resistance: bool = False,
) -> DataBundle:
    from picnnt.data.real import load_dataset
    cells = load_dataset(name, processed_dir)
    return build_data_from_cells(
        cells, split_seed=split_seed, seq_len=seq_len, stride=stride, smooth_resistance=smooth_resistance,
    )


def build_samsung_data(
    raw_dir: str, split_seed: int = 0, seq_len: int = 50, stride: int = 1,
) -> DataBundle:
    from picnnt.data.real_samsung import load_dataset
    cells = load_dataset(raw_dir)
    return build_data_from_cells(cells, split_seed=split_seed, seq_len=seq_len, stride=stride)


def build_nasa_data(
    raw_dir: str, split_seed: int = 0, seq_len: int = 20, stride: int = 1,
) -> DataBundle:
    from picnnt.data.real_nasa import load_dataset
    cells = load_dataset(raw_dir)
    return build_data_from_cells(cells, split_seed=split_seed, seq_len=seq_len, stride=stride)


def build_zhang_data(
    raw_dir: str, split_seed: int = 0, seq_len: int = 10, stride: int = 1,
) -> DataBundle:
    from picnnt.data.real_zhang import load_dataset
    cells = load_dataset(raw_dir)
    return build_data_from_cells(cells, split_seed=split_seed, seq_len=seq_len, stride=stride)


@torch.no_grad()
def predict(model: nn.Module, dataset: WindowDataset, device: torch.device, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, trues = [], []
    for idxs in sequential_eval_batches(dataset, batch_size):
        x, y, r, c = _batched_gather(dataset, idxs, device)
        y_hat = model(x)
        preds.append(y_hat.cpu().numpy())
        trues.append(y.cpu().numpy())
    return np.concatenate(trues), np.concatenate(preds)


def run_seeds(
    model_fn: Callable[[], nn.Module],
    data: DataBundle,
    device: torch.device,
    config: TrainConfig,
    n_seeds: int = 10,
    seed_offset: int = 0,
) -> list[Metrics]:
    results = []
    for i in range(n_seeds):
        seed = seed_offset + i
        set_seed(seed)
        model = model_fn()
        model, _ = train_model(model, data.train_ds, data.val_ds, device, config, verbose=False)
        y_true, y_pred = predict(model, data.test_ds, device, config.batch_size)
        results.append(compute_metrics(y_true, y_pred))
    return results


def run_ensemble(
    model_fn: Callable[[], nn.Module],
    data: DataBundle,
    device: torch.device,
    config: TrainConfig,
    n_seeds: int = 5,
    seed_offset: int = 0,
) -> Metrics:
    all_preds = []
    y_true = None
    for i in range(n_seeds):
        seed = seed_offset + i
        set_seed(seed)
        model = model_fn()
        model, _ = train_model(model, data.train_ds, data.val_ds, device, config, verbose=False)
        y_true, y_pred = predict(model, data.test_ds, device, config.batch_size)
        all_preds.append(y_pred)
    ensembled_pred = np.mean(np.stack(all_preds, axis=0), axis=0)
    return compute_metrics(y_true, ensembled_pred)


def val_rmse_for_config(
    model_fn: Callable[[], nn.Module], data: DataBundle, device: torch.device, config: TrainConfig, seed: int = 0,
) -> float:
    set_seed(seed)
    model = model_fn()
    _, history = train_model(model, data.train_ds, data.val_ds, device, config, verbose=False)
    return history["best_val_rmse"]


def _default_train_config(use_physics: bool, weights: LossWeights | None = None) -> TrainConfig:
    return TrainConfig(use_physics=use_physics, loss_weights=weights or LossWeights())


MODEL_REGISTRY: dict[str, tuple[Callable[[], nn.Module], TrainConfig]] = {
    "pi_cnn_transformer": (lambda: PICNNTransformer(), _default_train_config(True)),
    "vanilla_transformer": (lambda: PICNNTransformer(), _default_train_config(False)),
    "cnn": (lambda: CNNBaseline(), _default_train_config(False)),
    "lstm": (lambda: LSTMBaseline(), _default_train_config(False)),
    "pi_cnn_transformer_small": (
        lambda: PICNNTransformer(d_model=32, ffn_dim=64, num_encoder_layers=1), _default_train_config(True),
    ),
    "vanilla_transformer_small": (
        lambda: PICNNTransformer(d_model=32, ffn_dim=64, num_encoder_layers=1), _default_train_config(False),
    ),
}

def summarize(results: list[Metrics]) -> dict:
    rmse = np.array([m.rmse for m in results])
    mae = np.array([m.mae for m in results])
    r2 = np.array([m.r2 for m in results])
    mape = np.array([m.mape for m in results])
    return {
        "rmse_mean": rmse.mean(), "rmse_std": rmse.std(),
        "mae_mean": mae.mean(), "mae_std": mae.std(),
        "r2_mean": r2.mean(), "r2_std": r2.std(),
        "mape_mean": mape.mean(), "mape_std": mape.std(),
        "rmse_values": rmse,
    }
