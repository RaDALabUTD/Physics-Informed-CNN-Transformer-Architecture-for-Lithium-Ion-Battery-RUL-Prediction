from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np

from picnnt.data.preprocessing import CellTrajectory

EOL_FRACTION = 0.80


def _parse_run_time_to_seconds(s: str) -> float:
    h, m, rest = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest)


def _extract_discharge_cycles(
    zyk_bytes_list: list[bytes],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    v_means, i_means, t_maxs, qs, r_ints = [], [], [], [], []
    prev_block_last_v, prev_block_last_i = None, None
    for raw in zyk_bytes_list:
        text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()
        header = lines[0].split(",")
        col = {name: idx for idx, name in enumerate(header)}

        cur_block_type = None
        block_t, block_v, block_i, block_temp = [], [], [], []

        def flush_discharge_block():
            nonlocal prev_block_last_v, prev_block_last_i
            if len(block_t) < 2:
                return
            if cur_block_type != "42":
                prev_block_last_v, prev_block_last_i = block_v[-1], block_i[-1]
                return
            t = np.array(block_t)
            i = np.array(block_i)
            v = np.array(block_v)
            temp = np.array(block_temp)
            q = float(np.trapz(np.abs(i), t) / 3600.0)
            if q <= 0:
                prev_block_last_v, prev_block_last_i = block_v[-1], block_i[-1]
                return
            v_means.append(float(v.mean()))
            i_means.append(float(np.abs(i).mean()))
            t_maxs.append(float(np.nanmax(temp)) if np.isfinite(temp).any() else float("nan"))
            qs.append(q)
            if prev_block_last_v is not None:
                after = min(2, len(v) - 1)
                di = prev_block_last_i - i[after]
                r = (prev_block_last_v - v[after]) / di if abs(di) > 1e-3 else float("nan")
                r_ints.append(float(r) if 0 < r < 1.0 else float("nan"))
            else:
                r_ints.append(float("nan"))
            prev_block_last_v, prev_block_last_i = block_v[-1], block_i[-1]

        for line in lines[1:]:
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < len(header):
                continue
            st = parts[col["step_type"]]
            if st != cur_block_type:
                flush_discharge_block()
                cur_block_type = st
                block_t, block_v, block_i, block_temp = [], [], [], []
            try:
                t_sec = _parse_run_time_to_seconds(parts[col["run_time"]])
                block_t.append(t_sec)
                block_v.append(float(parts[col["c_vol"]]))
                block_i.append(float(parts[col["c_cur"]]))
                block_temp.append(float(parts[col["c_surf_temp"]]))
            except (ValueError, IndexError):
                continue
        flush_discharge_block()

    return np.array(v_means), np.array(i_means), np.array(t_maxs), np.array(qs), np.array(r_ints)


def load_cell_zip(path: Path) -> CellTrajectory | None:
    cell_id = path.stem
    with zipfile.ZipFile(path) as z:
        zyk_names = sorted(
            [n for n in z.namelist() if re.search(r"_ZYK\.csv$", n)],
            key=lambda n: int(re.search(r"_(\d+)_ZYK\.csv$", n).group(1)),
        )
        if not zyk_names:
            return None
        zyk_bytes = [z.read(n) for n in zyk_names]

    V, I, T, Q, R = _extract_discharge_cycles(zyk_bytes)
    if len(Q) < 60:
        return None

    nominal_capacity = float(np.percentile(Q[: max(1, len(Q) // 20)], 90))
    eol_thresh = EOL_FRACTION * nominal_capacity
    below = np.where(Q < eol_thresh)[0]
    eol_cycle_idx = int(below[0]) if len(below) > 0 else len(Q) - 1
    end = min(eol_cycle_idx + 20, len(Q))

    return CellTrajectory(
        cell_id=cell_id,
        dataset="SAMSUNG",
        c_rate=float("nan"),
        temperature_c=float("nan"),
        chemistry="NCA",
        cycles=np.arange(1, end + 1, dtype=np.int64),
        V=V[:end].astype(np.float32),
        I=I[:end].astype(np.float32),
        T=T[:end].astype(np.float32),
        Q=Q[:end].astype(np.float32),
        R_int=R[:end].astype(np.float32),
        eol_cycle=eol_cycle_idx + 1,
        nominal_capacity=nominal_capacity,
        has_real_resistance=bool(np.any(~np.isnan(R[:end]))),
    )


def load_dataset(raw_dir: str | Path) -> list[CellTrajectory]:
    raw_dir = Path(raw_dir)
    cells = []
    for path in sorted(raw_dir.glob("*.zip")):
        try:
            cell = load_cell_zip(path)
        except Exception as e:
            print(f"skipping {path.name}: {e}")
            continue
        if cell is not None:
            cells.append(cell)
    print(f"SAMSUNG: loaded {len(cells)} cells from {raw_dir}")
    return cells
