from __future__ import annotations

import dataclasses

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

FEATURE_NAMES = ["V_mean", "I_mean", "T_max", "Q", "R_int"]
SEQ_LEN = 50


@dataclasses.dataclass
class CellTrajectory:
    cell_id: str
    dataset: str
    c_rate: float
    temperature_c: float
    chemistry: str
    cycles: np.ndarray
    V: np.ndarray
    I: np.ndarray
    T: np.ndarray
    Q: np.ndarray
    R_int: np.ndarray
    eol_cycle: int
    nominal_capacity: float = 1.0
    has_real_resistance: bool = False


@dataclasses.dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    @staticmethod
    def fit(x: np.ndarray) -> "Scaler":
        return Scaler(mean=x.mean(axis=0), std=x.std(axis=0) + 1e-8)


def cell_to_resistance_proxy(cell: CellTrajectory) -> np.ndarray:
    if cell.has_real_resistance and not np.all(np.isnan(cell.R_int)):
        r = cell.R_int.astype(np.float32).copy()
        valid = ~np.isnan(r)
        if valid.any():
            r[~valid] = np.mean(r[valid])
            return np.clip(r, 1e-4, None)
    v_nominal = float(np.max(cell.V)) + 0.05
    return np.clip((v_nominal - cell.V) / np.clip(cell.I, 1e-3, None), 1e-4, None).astype(np.float32)


def cell_to_features(cell: CellTrajectory) -> np.ndarray:
    r = cell_to_resistance_proxy(cell)
    return np.stack([cell.V, cell.I, cell.T, cell.Q, r], axis=1).astype(np.float32)


def cell_to_rul(cell: CellTrajectory) -> np.ndarray:
    return np.clip(cell.eol_cycle - cell.cycles, 0, None).astype(np.float32)


def cell_to_capacity_fraction(cell: CellTrajectory) -> np.ndarray:
    return np.clip(cell.Q / cell.nominal_capacity, 0.0, 1.2).astype(np.float32)


@dataclasses.dataclass
class SplitCells:
    train: list[CellTrajectory]
    val: list[CellTrajectory]
    test: list[CellTrajectory]


def split_cells(cells: list[CellTrajectory], seed: int = 0) -> SplitCells:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(cells))
    n_train = int(0.6 * len(cells))
    n_val = int(0.2 * len(cells))
    train = [cells[i] for i in idx[:n_train]]
    val = [cells[i] for i in idx[n_train:n_train + n_val]]
    test = [cells[i] for i in idx[n_train + n_val:]]
    return SplitCells(train=train, val=val, test=test)


class WindowDataset(Dataset):
    def __init__(self, cells: list[CellTrajectory], scaler: Scaler, seq_len: int = SEQ_LEN, stride: int = 1):
        self.seq_len = seq_len
        self.stride = stride
        self.windows: list[np.ndarray] = []
        self.targets: list[float] = []
        self.resistance: list[float] = []
        self.capacity_fraction: list[float] = []
        self.cycle_idx: list[float] = []
        self.cell_ids: list[str] = []
        self.cell_of_window: list[int] = []
        self.cell_boundaries: list[tuple[int, int]] = []

        flat_start = 0
        for cell in cells:
            feats = scaler.transform(cell_to_features(cell))
            rul = cell_to_rul(cell)
            r_proxy = cell_to_resistance_proxy(cell)
            cap_frac = cell_to_capacity_fraction(cell)
            n = len(cell.cycles)
            if n < seq_len:
                continue
            cell_window_start = flat_start
            for t in range(seq_len - 1, n, stride):
                window = feats[t - seq_len + 1: t + 1]
                self.windows.append(window)
                self.targets.append(rul[t])
                self.resistance.append(r_proxy[t])
                self.capacity_fraction.append(cap_frac[t])
                self.cycle_idx.append(float(cell.cycles[t]))
                self.cell_ids.append(cell.cell_id)
                flat_start += 1
            self.cell_boundaries.append((cell_window_start, flat_start))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.windows[idx]),
            torch.tensor(self.targets[idx], dtype=torch.float32),
            torch.tensor(self.resistance[idx], dtype=torch.float32),
            torch.tensor(self.cycle_idx[idx], dtype=torch.float32),
        )


class TemporalBatchSampler(Sampler[list[int]]):
    def __init__(
        self, dataset: WindowDataset, batch_size: int, batches_per_epoch: int,
        segment_len: int = 8, seed: int = 0,
    ):
        self.dataset = dataset
        self.segment_len = segment_len
        self.num_segments = max(1, batch_size // segment_len)
        self.batches_per_epoch = batches_per_epoch
        self.rng = np.random.default_rng(seed)
        self.usable_cells = [(s, e) for s, e in dataset.cell_boundaries if e - s >= segment_len]

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            indices = []
            cell_choices = self.rng.choice(len(self.usable_cells), size=self.num_segments, replace=True)
            for ci in cell_choices:
                s, e = self.usable_cells[ci]
                n_avail = e - s
                start = s + self.rng.integers(0, n_avail - self.segment_len + 1)
                indices.extend(range(start, start + self.segment_len))
            yield indices

    def __len__(self) -> int:
        return self.batches_per_epoch


def sequential_eval_batches(dataset: WindowDataset, batch_size: int):
    for s, e in dataset.cell_boundaries:
        for start in range(s, e, batch_size):
            yield list(range(start, min(start + batch_size, e)))
