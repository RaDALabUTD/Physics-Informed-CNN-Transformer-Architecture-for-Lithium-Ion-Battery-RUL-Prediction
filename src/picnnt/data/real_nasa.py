from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio

from picnnt.data.preprocessing import CellTrajectory

EOL_FRACTION = 0.80


def load_cell_mat(path: Path) -> CellTrajectory | None:
    mat = sio.loadmat(str(path))
    var_name = path.stem
    if var_name not in mat:
        return None
    cycles = mat[var_name][0, 0]["cycle"][0]

    V, I, T, Q, R = [], [], [], [], []
    last_rct = float("nan")

    for c in cycles:
        ctype = c["type"][0]
        d = c["data"][0, 0]
        if ctype == "impedance":
            names = d.dtype.names
            if names and "Rct" in names:
                try:
                    rct = complex(d["Rct"][0, 0]).real
                    if 0 < rct < 1.0:
                        last_rct = float(rct)
                except (ValueError, IndexError, TypeError):
                    pass
            continue
        if ctype != "discharge":
            continue
        names = d.dtype.names
        if "Capacity" not in names or "Voltage_measured" not in names:
            continue
        try:
            q = float(d["Capacity"][0, 0])
            v_arr = np.asarray(d["Voltage_measured"], dtype=np.float64).ravel()
            i_arr = np.asarray(d["Current_measured"], dtype=np.float64).ravel()
            t_arr = np.asarray(d["Temperature_measured"], dtype=np.float64).ravel()
        except (ValueError, IndexError):
            continue
        if v_arr.size == 0 or q <= 0:
            continue
        V.append(float(v_arr.mean()))
        I.append(float(np.abs(i_arr).mean()))
        T.append(float(t_arr.max()) if t_arr.size else float("nan"))
        Q.append(q)
        R.append(last_rct)

    if len(Q) < 60:
        return None

    Q = np.array(Q, dtype=np.float32)
    settle = min(5, len(Q) - 1)
    nominal_capacity = float(np.max(Q[: max(settle + 1, 10)]))
    eol_thresh = EOL_FRACTION * nominal_capacity
    below = np.where(Q[settle:] < eol_thresh)[0]
    eol_cycle_idx = int(below[0] + settle) if len(below) > 0 else len(Q) - 1
    end = min(eol_cycle_idx + 20, len(Q))

    R_arr = np.array(R[:end], dtype=np.float32)
    has_resistance = bool(np.any(~np.isnan(R_arr)))

    return CellTrajectory(
        cell_id=var_name,
        dataset="NASA",
        c_rate=float("nan"),
        temperature_c=float("nan"),
        chemistry="LCO",
        cycles=np.arange(1, end + 1, dtype=np.int64),
        V=np.array(V[:end], dtype=np.float32),
        I=np.array(I[:end], dtype=np.float32),
        T=np.array(T[:end], dtype=np.float32),
        Q=Q[:end],
        R_int=R_arr,
        eol_cycle=eol_cycle_idx + 1,
        nominal_capacity=nominal_capacity,
        has_real_resistance=has_resistance,
    )


def load_dataset(raw_dir: str | Path) -> list[CellTrajectory]:
    raw_dir = Path(raw_dir)
    cells = []
    seen_ids = set()
    for path in sorted(raw_dir.rglob("B*.mat")):
        if path.stem in seen_ids:
            continue
        try:
            cell = load_cell_mat(path)
        except Exception as e:
            print(f"skipping {path.name}: {e}")
            continue
        if cell is not None:
            cells.append(cell)
            seen_ids.add(path.stem)
    print(f"NASA: loaded {len(cells)} cells from {raw_dir}")
    return cells
