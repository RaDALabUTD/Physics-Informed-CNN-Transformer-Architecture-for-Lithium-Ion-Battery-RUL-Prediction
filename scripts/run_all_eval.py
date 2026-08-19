import torch
import sys
from pathlib import Path
sys.path.append(str(Path("src").absolute()))

from picnnt.experiments import build_nasa_data, build_samsung_data, build_real_data, MODEL_REGISTRY, run_seeds, summarize
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_fn, config = MODEL_REGISTRY["pi_cnn_transformer"]

def evaluate_dataset(name, data_builder):
    print(f"\nEvaluating {name}...")
    try:
        data = data_builder()
        if not data.cells:
            print(f"No cells found for {name}.")
            return
        results = run_seeds(model_fn, data, device, config, n_seeds=3)
        summary = summarize(results)
        print(f"Results for {name}:")
        print(f"RMSE: {summary['rmse_mean']:.3f} +/- {summary['rmse_std']:.3f}")
        print(f"MAE: {summary['mae_mean']:.3f} +/- {summary['mae_std']:.3f}")
    except Exception as e:
        print(f"Error evaluating {name}: {e}")

evaluate_dataset("NASA", lambda: build_nasa_data("data/raw/NASA"))
evaluate_dataset("SAMSUNG", lambda: build_samsung_data("data/raw/SAMSUNG"))
evaluate_dataset("CALCE", lambda: build_real_data("CALCE", "data/processed/CALCE"))
