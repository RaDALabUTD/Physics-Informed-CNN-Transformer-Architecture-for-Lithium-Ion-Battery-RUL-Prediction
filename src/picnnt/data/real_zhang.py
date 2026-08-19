from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from picnnt.data.preprocessing import CellTrajectory

EOL_FRACTION = 0.80
CELLS = [
    "25C01", "25C02", "25C03", "25C04", "25C05", "25C06", "25C07", "25C08",
    "35C01", "35C02", "45C01", "45C02",
]


def _cell_temperature(cell: str) -> float:
    return float(re.match(r"(\d+)C", cell).group(1))


def _data_col_count(path: Path) -> int:
    with open(path) as f:
        next(f)
        for line in f:
            parts = [p for p in line.rstrip("\n").split("\t") if p != ""]
            if parts:
                return len(parts)
    return 0


def _load_capacity(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_cols = _data_col_count(path)
    has_vi = n_cols >= 6
    if has_vi:
        cycle_col, oxred_col, v_col, i_col, q_col = 1, 2, 3, 4, 5
    else:
        cycle_col, oxred_col, q_col = 1, 2, 3

    cycles: dict[int, dict[str, list[float]]] = {}
    with open(path) as f:
        next(f)
        for line in f:
            parts = [p for p in line.rstrip("\n").split("\t") if p != ""]
            if len(parts) < (q_col + 1):
                continue
            try:
                cyc = int(float(parts[cycle_col]))
                oxred = int(float(parts[oxred_col]))
                v = float(parts[v_col]) if has_vi else 0.0
                i = float(parts[i_col]) if has_vi else 0.0
                q = float(parts[q_col])
            except ValueError:
                continue
            if oxred != 0:
                continue
            d = cycles.setdefault(cyc, {"v": [], "i": [], "q": []})
            d["v"].append(v)
            d["i"].append(abs(i))
            d["q"].append(q)

    cyc_nums = sorted(cycles.keys())
    V = np.array([np.mean(cycles[c]["v"]) for c in cyc_nums], dtype=np.float32)
    I = np.array([np.mean(cycles[c]["i"]) for c in cyc_nums], dtype=np.float32)
    Q = np.array([max(cycles[c]["q"]) / 1000.0 for c in cyc_nums], dtype=np.float32)
    return np.array(cyc_nums, dtype=np.int64), V, I, Q


def _load_resistance(path: Path) -> dict[int, float]:
    cycle_col, freq_col, re_col = 1, 2, 3
    best_freq: dict[int, float] = {}
    r_at_best: dict[int, float] = {}
    with open(path) as f:
        lines = f.readlines()
    if lines and not lines[0].split("\t")[0].strip().replace(".", "", 1).isdigit():
        lines = lines[1:]
    for line in lines:
        parts = line.rstrip("\n").split("\t")
        if len(parts) <= re_col:
            continue
        try:
            cyc = int(float(parts[cycle_col]))
            freq = float(parts[freq_col])
            r = float(parts[re_col])
        except ValueError:
            continue
        if cyc not in best_freq or freq > best_freq[cyc]:
            best_freq[cyc] = freq
            r_at_best[cyc] = r
    return r_at_best


def load_cell(raw_dir: Path, cell: str) -> CellTrajectory | None:
    cap_path = raw_dir / "Capacity data" / f"Data_Capacity_{cell}.txt"
    eis_path = raw_dir / "EIS data" / f"EIS_state_I_{cell}.txt"
    if not cap_path.exists() or not eis_path.exists():
        return None

    cycles, V, I, Q = _load_capacity(cap_path)
    if len(cycles) < 10:
        return None
    r_by_cycle = _load_resistance(eis_path)

    nominal_capacity = float(Q[0])
    eol_thresh = EOL_FRACTION * nominal_capacity
    below = np.where(Q < eol_thresh)[0]
    eol_cycle_idx = int(below[0]) if len(below) > 0 else len(Q) - 1
    end = min(eol_cycle_idx + 5, len(Q))

    R = np.full(end, np.nan, dtype=np.float32)
    last_r = float("nan")
    measured_cycles = sorted(r_by_cycle.keys())
    for idx in range(end):
        cyc = int(cycles[idx])
        if cyc in r_by_cycle:
            last_r = r_by_cycle[cyc]
        elif measured_cycles and cyc > measured_cycles[0]:
            pass
        R[idx] = last_r
    has_resistance = bool(np.any(~np.isnan(R)))

    return CellTrajectory(
        cell_id=f"ZHANG_{cell}",
        dataset="ZHANG",
        c_rate=float("nan"),
        temperature_c=_cell_temperature(cell),
        chemistry="LCO",
        cycles=cycles[:end],
        V=V[:end],
        I=I[:end],
        T=np.full(end, _cell_temperature(cell), dtype=np.float32),
        Q=Q[:end],
        R_int=R,
        eol_cycle=int(cycles[eol_cycle_idx]),
        nominal_capacity=nominal_capacity,
        has_real_resistance=has_resistance,
    )


def load_dataset(raw_dir: str | Path) -> list[CellTrajectory]:
    raw_dir = Path(raw_dir)
    cells = []
    for cell in CELLS:
        try:
            traj = load_cell(raw_dir, cell)
        except Exception as e:
            print(f"skipping {cell}: {e}")
            continue
        if traj is not None:
            cells.append(traj)
    print(f"ZHANG: loaded {len(cells)} cells from {raw_dir}")
    return cells
