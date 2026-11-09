"""Test de data, niet alleen de code. Garbage in, garbage out.

Elke test hieronder neemt de echte data, maakt er bewust iets aan stuk, en controleert dat
het schema het tegenhoudt.
"""

import pandera.errors
import pytest

from src import schema


def test_echte_data_is_geldig(df):
    """De echte dataset moet door het schema komen."""
    schema.TRAINING.validate(df)


def test_hernoemde_kolom_wordt_geweigerd(df):
    """Een hernoemde kolom moet een fout geven."""
    kapot = df.rename(columns={"Torque [Nm]": "Torque_Nm"})
    with pytest.raises(pandera.errors.SchemaError):
        schema.TRAINING.validate(kapot)


def test_temperatuur_buiten_bereik_wordt_geweigerd(df):
    """Temperatuur ver boven het bereik (bv. Fahrenheit i.p.v. Kelvin) moet falen."""
    kapot = df.copy()
    kapot["Air temperature [K]"] = kapot["Air temperature [K]"] + 200
    with pytest.raises(pandera.errors.SchemaError):
        schema.TRAINING.validate(kapot)


def test_onbekend_type_wordt_geweigerd(df):
    """Een producttype dat niet L/M/H is, moet falen."""
    kapot = df.copy()
    kapot.loc[0, "Type"] = "X"
    with pytest.raises(pandera.errors.SchemaError):
        schema.TRAINING.validate(kapot)


def test_negatieve_slijtage_wordt_geweigerd(df):
    """Negatieve tool wear is fysiek onmogelijk."""
    kapot = df.copy()
    kapot.loc[0, "Tool wear [min]"] = -5
    with pytest.raises(pandera.errors.SchemaError):
        schema.TRAINING.validate(kapot)


def test_voorspellen_vraagt_geen_doelkolom(df, cfg):
    """Bij het voorspellen ken je het antwoord nog niet. Het invoerschema mag het niet eisen."""
    schema.INVOER.validate(df.drop(columns=[cfg["doel"]]))
