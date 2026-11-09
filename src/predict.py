"""Een opgeslagen model laden en voorspellen voor nieuwe machines.

python -m src.predict data/nieuwe_data.csv
python -m src.predict data/nieuwe_data.csv --zonder-schema
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from . import schema
from .config import alle_features, laad_config
from .data import laad_data


def laad_model(pad: str | Path) -> Pipeline:
    return joblib.load(pad)


def voorspel(model: Pipeline, df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """De kans op machinefalen, één getal per machine."""
    return model.predict_proba(df[alle_features(cfg)])[:, 1]


def main(argv: list[str]) -> None:
    cfg = laad_config()
    pad = argv[0]
    valideer = None if "--zonder-schema" in argv else schema.INVOER

    df = laad_data(pad, valideer_met=valideer)
    kansen = voorspel(laad_model(cfg["model"]["pad"]), df, cfg)

    threshold = cfg.get("threshold", 0.5)
    alarmen = (kansen >= threshold).sum()

    print(f"{len(kansen)} machines geëvalueerd.")
    print(f"Gemiddelde faalkans: {kansen.mean():.1%}")
    print(f"Alarmen (threshold={threshold}): {alarmen}")
    print(f"Machines boven threshold: {alarmen}/{len(kansen)}")


if __name__ == "__main__":
    main(sys.argv[1:])
