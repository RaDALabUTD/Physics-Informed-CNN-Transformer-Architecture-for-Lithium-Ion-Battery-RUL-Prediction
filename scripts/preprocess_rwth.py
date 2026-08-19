import sys
from pathlib import Path
sys.path.append(str(Path("external/BatteryML").absolute()))

from batteryml.preprocess.preprocess_RWTH import RWTHPreprocessor
import os

os.makedirs("data/processed/RWTH", exist_ok=True)
preprocessor = RWTHPreprocessor(output_dir="data/processed/RWTH")
processed, skipped = preprocessor.process("data/raw/RWTH")
print(f"Finished RWTH preprocessing: {processed} processed, {skipped} skipped.")
