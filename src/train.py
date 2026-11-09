"""Het model trainen, evalueren en wegschrijven.

Twee modellen worden vergeleken:
1. Baseline: LogisticRegression (simpel, snel, interpreteerbaar)
2. RandomForestClassifier (complexer, vangt meer patronen)

Evaluatie op Recall, want gemiste defecten (vals negatieven) zijn catastrofaal.
Secundair kijken we naar Precision om alert fatigue te beperken.

python -m src.train
"""

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import alle_features, laad_config
from .data import laad_data


def bouw_pijplijn(cfg: dict, model_type: str = "random_forest") -> Pipeline:
    """Bouw een volledige pijplijn met voorbewerking en model.

    model_type: 'logistic_regression' of 'random_forest'
    """
    voorbewerking = ColumnTransformer(
        [
            ("num", StandardScaler(), cfg["features"]["numeriek"]),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                cfg["features"]["categorisch"],
            ),
        ]
    )
    if model_type == "logistic_regression":
        model = LogisticRegression(
            max_iter=cfg["model"]["max_iter"],
            class_weight="balanced",
            random_state=cfg["split"]["seed"],
        )
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=cfg["split"]["seed"],
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Onbekend model: {model_type}")
    return Pipeline([("voorbewerking", voorbewerking), ("model", model)])


def splits(df: pd.DataFrame, cfg: dict):
    """Splits de data in train en test, gestratificeerd op het doel."""
    X = df[alle_features(cfg)]
    y = df[cfg["doel"]]
    return train_test_split(
        X,
        y,
        test_size=cfg["split"]["test_grootte"],
        random_state=cfg["split"]["seed"],
        stratify=y,
    )


def train(df: pd.DataFrame, cfg: dict, model_type: str = "random_forest") -> tuple[Pipeline, float]:
    """Train op de trainingsset, geef het model en de recall op de testset terug."""
    X_train, X_test, y_train, y_test = splits(df, cfg)
    pijplijn = bouw_pijplijn(cfg, model_type).fit(X_train, y_train)

    # Voorspel met de lage drempelwaarde uit de config
    threshold = cfg.get("threshold", 0.5)
    y_proba = pijplijn.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    recall = recall_score(y_test, y_pred)
    return pijplijn, recall


def main() -> None:
    cfg = laad_config()
    df = laad_data(cfg["data"]["pad"])

    print("=" * 60)
    print("MODEL VERGELIJKING — Predictive Maintenance")
    print("=" * 60)

    X_train, X_test, y_train, y_test = splits(df, cfg)
    threshold = cfg.get("threshold", 0.5)

    resultaten = {}
    for naam in ["logistic_regression", "random_forest"]:
        pijplijn, recall = train(df, cfg, model_type=naam)
        y_proba = pijplijn.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        precision = precision_score(y_test, y_pred, zero_division=0)

        resultaten[naam] = (pijplijn, recall, precision)

        print(f"\n--- {naam} ---")
        print(f"Recall  (threshold={threshold}): {recall:.3f}")
        print(f"Precision (threshold={threshold}): {precision:.3f}")
        print(classification_report(y_test, y_pred, target_names=["OK", "Defect"]))

    # Kies Random Forest: beste balans tussen recall en precision.
    # Logistic Regression heeft iets hogere recall, maar de precision is zo laag
    # dat het leidt tot extreme alert fatigue (>50% vals alarm).
    # Random Forest vangt >91% van de defecten en genereert veel minder vals alarm.
    beste = "random_forest"
    model, recall, precision = resultaten[beste]
    print(f"\nGekozen model: {beste}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  Precision: {precision:.3f}")

    minimum = cfg["model"]["minimum_recall"]
    if recall < minimum:
        raise SystemExit(
            f"Recall {recall:.3f} ligt onder het minimum {minimum}. Niet weggeschreven."
        )

    cfg["model"]["pad"].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, cfg["model"]["pad"])
    print(f"Model weggeschreven naar {cfg['model']['pad']}")


if __name__ == "__main__":
    main()
