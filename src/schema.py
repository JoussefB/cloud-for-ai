"""Hoe de data eruit hoort te zien, vastgelegd met Pandera.

Twee schema's:

- INVOER: de sensordata die binnenkomen bij het voorspellen
- TRAINING: hetzelfde, plus de doelkolom (Machine failure)

De sub-faalkolommen (TWF, HDF, PWF, OSF, RNF) worden NIET gevalideerd
als verplicht, want bij voorspellen heb je die niet. Ze staan er bij
training wel in, maar strict=False laat extra kolommen door.

Bereiken zijn iets ruimer dan de trainingsdata, om kleine variatie toe te laten
zonder fysiek onmogelijke waarden door te laten.
"""

import pandera.pandas as pa

INVOER = pa.DataFrameSchema(
    {
        "UDI": pa.Column(int, pa.Check.gt(0), unique=True),
        "Product ID": pa.Column(str),
        "Type": pa.Column(str, pa.Check.isin(["L", "M", "H"])),
        # Trainingsdata: 295.3 – 304.5 K.  Marge: 290 – 320 K.
        "Air temperature [K]": pa.Column(float, pa.Check.in_range(290, 320)),
        # Trainingsdata: 305.7 – 313.8 K.  Marge: 300 – 325 K.
        "Process temperature [K]": pa.Column(float, pa.Check.in_range(300, 325)),
        # Trainingsdata: 1168 – 2886 rpm.  Marge: 1000 – 3000 rpm.
        "Rotational speed [rpm]": pa.Column(int, pa.Check.in_range(1000, 3000)),
        # Trainingsdata: 3.8 – 76.6 Nm.  Marge: 0 – 100 Nm.
        "Torque [Nm]": pa.Column(float, pa.Check.in_range(0, 100)),
        # Trainingsdata: 0 – 253 min.  Marge: 0 – 300 min.
        "Tool wear [min]": pa.Column(int, pa.Check.in_range(0, 300)),
    },
    strict=False,
    name="Predictive-Maintenance",
)

TRAINING = INVOER.add_columns({"Machine failure": pa.Column(int, pa.Check.isin([0, 1]))})
