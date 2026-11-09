"""Test een model zonder te weten wat het exact moet voorspellen.

Wat je wél kan vastleggen:
1. vorm en type van de output
2. invarianten: kansen liggen altijd tussen 0 en 1
3. de pijplijn: ruwe invoer erin, voorspelling eruit, zonder te breken
4. persistentie: opgeslagen en teruggeladen geeft hetzelfde
5. reproduceerbaarheid: zelfde seed, zelfde model
6. een ondergrens: slechter dan dit mag niet naar productie
"""

import joblib
import numpy as np

from src.predict import laad_model, voorspel
from src.train import train


def test_vorm_een_kans_per_machine(model, df, cfg):
    kansen = voorspel(model, df.head(50), cfg)
    assert kansen.shape == (50,)
    assert kansen.dtype == np.float64


def test_invariant_kansen_liggen_tussen_0_en_1(model, df, cfg):
    kansen = voorspel(model, df, cfg)
    assert np.all((kansen >= 0) & (kansen <= 1))


def test_pijplijn_neemt_ruwe_rijen(model, df, cfg):
    """Alle voorbewerking moet in de pijplijn zitten, niet in een notebookcel ervoor."""
    ruw = df.head(1).copy()
    ruw["iets_onverwachts"] = "negeer mij"
    assert voorspel(model, ruw, cfg).shape == (1,)


def test_persistentie(model, df, cfg, tmp_path):
    pad = tmp_path / "model.joblib"
    joblib.dump(model, pad)

    herladen = laad_model(pad)
    na = voorspel(herladen, df, cfg)

    np.testing.assert_allclose(na, voorspel(model, df, cfg))


def test_zelfde_seed_zelfde_model(df, cfg):
    eerste, _ = train(df, cfg, model_type="random_forest")
    tweede, _ = train(df, cfg, model_type="random_forest")
    np.testing.assert_allclose(voorspel(eerste, df, cfg), voorspel(tweede, df, cfg))


def test_beter_dan_de_ondergrens(model_en_recall, cfg):
    """Recall moet boven het minimum liggen uit de config."""
    _, recall = model_en_recall
    assert recall >= cfg["model"]["minimum_recall"]
