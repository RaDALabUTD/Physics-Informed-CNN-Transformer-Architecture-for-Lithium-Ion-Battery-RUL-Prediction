from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from picnnt.data.preprocessing import CellTrajectory

T_FALLBACK_C = 30.0
EOL_FRACTION = 0.80


def _ir_step_resistance(V: np.ndarray, I: np.ndarray, dQ: np.ndarray) -> float:
    started = np.where(dQ > 0)[0]
    if started.size == 0:
        return float("nan")
    idx0 = int(started[0])
    if idx0 < 1:
        return float("nan")
    after = min(idx0 + 2, len(V) - 1)
    v_before, v_after = V[idx0 - 1], V[after]
    i_before, i_after = I[idx0 - 1], I[after]
    di = i_before - i_after
    if abs(di) < 1e-3:
        return float("nan")
    r = (v_before - v_after) / di
    if not (0 < r < 1.0):
        return float("nan")
    return float(r)


def _cycle_summary(cycle: dict) -> tuple[float, float, float, float, float] | None:
    dQ = np.asarray(cycle["discharge_capacity_in_Ah"], dtype=np.float64)
    if dQ.size == 0 or dQ.max() <= 0:
        return None
    V = np.asarray(cycle["voltage_in_V"], dtype=np.float64)
    I = np.asarray(cycle["current_in_A"], dtype=np.float64)
    mask = dQ > 0
    if not mask.any():
        return None
    v_mean = float(V[mask].mean())
    i_mean = float(np.abs(I[mask]).mean())
    q = float(dQ.max())
    temp = cycle.get("temperature_in_C")
    if temp is not None and len(temp) > 0:
        t_max = float(np.nanmax(np.asarray(temp, dtype=np.float64)))
    else:
        t_max = T_FALLBACK_C
    resist = cycle.get("internal_resistance_in_ohm")
    if resist is None:
        r_int = float("nan")
    elif np.isscalar(resist):
        r_int = float(resist) if (not np.isnan(resist) and resist > 0) else float("nan")
    else:
        valid = np.asarray(resist, dtype=np.float64)
        valid = valid[~np.isnan(valid) & (valid > 0)]
        r_int = float(valid.mean()) if valid.size > 0 else float("nan")
    if np.isnan(r_int):
        r_int = _ir_step_resistance(V, I, dQ)
    return v_mean, i_mean, t_max, q, r_int


def load_battery_data_pkl(path: Path) -> CellTrajectory | None:
    with open(path, "rb") as f:
        bd = pickle.load(f)

    cell_id = bd["cell_id"]
    nominal_capacity = float(bd.get("nominal_capacity_in_Ah") or 0.0)

    cycles_v, cycles_i, cycles_t, cycles_q, cycles_r, cycle_nums = [], [], [], [], [], []
    for cyc in bd["cycle_data"]:
        summary = _cycle_summary(cyc)
        if summary is None:
            continue
        v, i, t, q, r = summary
        cycles_v.append(v)
        cycles_i.append(i)
        cycles_t.append(t)
        cycles_q.append(q)
        cycles_r.append(r)
        cycle_nums.append(cyc["cycle_number"])

    if len(cycles_q) < 60:
        return None

    Q = np.array(cycles_q, dtype=np.float32)
    if nominal_capacity <= 0:
        nominal_capacity = float(np.percentile(Q[: max(1, len(Q) // 20)], 90))

    eol_thresh = EOL_FRACTION * nominal_capacity
    below = np.where(Q < eol_thresh)[0]
    eol_cycle_idx = int(below[0]) if len(below) > 0 else len(Q) - 1

    end = min(eol_cycle_idx + 20, len(Q))

    return CellTrajectory(
        cell_id=cell_id,
        dataset="MATR" if cell_id.upper().startswith("MATR") or cell_id.lower().startswith("b") else "HUST",
        c_rate=float("nan"),
        temperature_c=float("nan"),
        chemistry=str(bd.get("cathode_material") or "unknown"),
        cycles=np.arange(1, end + 1, dtype=np.int64),
        V=np.array(cycles_v[:end], dtype=np.float32),
        I=np.array(cycles_i[:end], dtype=np.float32),
        T=np.array(cycles_t[:end], dtype=np.float32),
        Q=Q[:end],
        R_int=np.array(cycles_r[:end], dtype=np.float32),
        eol_cycle=eol_cycle_idx + 1,
        nominal_capacity=nominal_capacity,
        has_real_resistance=bool(np.any(~np.isnan(cycles_r[:end]))),
    )


def load_dataset(name: str, processed_dir: str | Path) -> list[CellTrajectory]:
    processed_dir = Path(processed_dir)
    cells = []
    for path in sorted(processed_dir.glob("*.pkl")):
        try:
            cell = load_battery_data_pkl(path)
        except Exception as e:
            print(f"skipping {path.name}: {e}")
            continue
        if cell is not None:
            cells.append(cell)
    print(f"{name}: loaded {len(cells)} cells from {processed_dir}")
    return cells
