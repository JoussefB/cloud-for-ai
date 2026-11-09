# %% [markdown]
# # Predictive Maintenance — Data Exploratie
#
# Dataset: AI4I 2020 Predictive Maintenance Dataset (Kaggle / UCI)
# 10.000 rijen, sensordata van een freesmachine.
#
# Doel: binaire classificatie — voorspel of een machine binnenkort
#       zal falen (`Machine failure` = 1) of niet (= 0).

# %%
import sys

sys.path.insert(0, "..")


from src.config import laad_config
from src.data import laad_data

cfg = laad_config()
df = laad_data(cfg["data"]["pad"], valideer_met=None)

# %% [markdown]
# ## 1. Eerste blik op de data

# %%
print(f"Shape: {df.shape}")
print(f"\nKolommen:\n{df.dtypes}")

# %%
df.head(10)

# %%
df.describe()

# %% [markdown]
# ## 2. Doelvariabele: Machine failure
#
# De dataset is sterk **imbalanced**: slechts ~3.4% van de rijen is een defect.
# Dit verklaart waarom **recall** de juiste metriek is: accuracy zou misleidend
# hoog zijn (>96%) ook al voorspel je altijd "geen defect".

# %%
print("Verdeling Machine failure:")
print(df["Machine failure"].value_counts())
print(f"\nPercentage defecten: {df['Machine failure'].mean():.1%}")

# %% [markdown]
# ## 3. Product types
#
# Drie kwaliteitsklassen: L (Low, 50%), M (Medium, 30%), H (High, 20%).
# De faalratio verschilt per type.

# %%
print("Verdeling per type:")
print(df.groupby("Type")["Machine failure"].agg(["count", "mean", "sum"]))

# %% [markdown]
# ## 4. Sensor features
#
# Vijf continue sensorwaardes:
# - Air temperature [K]: omgevingstemperatuur (~300 K)
# - Process temperature [K]: procestemperatuur (~310 K)
# - Rotational speed [rpm]: draaisnelheid
# - Torque [Nm]: koppel
# - Tool wear [min]: cumulatieve slijtage van het gereedschap

# %%
features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
print(df[features].describe().round(2))

# %% [markdown]
# ## 5. Correlaties

# %%
corr = df[features + ["Machine failure"]].corr()["Machine failure"].drop("Machine failure")
print("Correlatie met Machine failure:")
print(corr.sort_values(ascending=False).round(3))

# %% [markdown]
# ## 6. Faalmodussen (sub-labels)
#
# De dataset bevat vijf specifieke faalmodussen:
# - TWF: Tool Wear Failure
# - HDF: Heat Dissipation Failure
# - PWF: Power Failure
# - OSF: Overstrain Failure
# - RNF: Random Failure
#
# **BELANGRIJK**: Deze kolommen zijn sub-labels van het doel.
# Ze gebruiken als features zou **data leakage** zijn — je vertelt het model
# letterlijk het antwoord. Ze staan daarom NIET in de feature-lijst.

# %%
faalmodussen = ["TWF", "HDF", "PWF", "OSF", "RNF"]
print("Aantal per faalmodus:")
print(df[faalmodussen].sum())

# %% [markdown]
# ## 7. Wat is er stuk aan de data?
#
# **Missende waarden**: Geen. De dataset is synthetisch en compleet.
#
# **Imbalance**: Slechts ~3.4% is positief. Zonder correctie leert een model
# gewoon "altijd 0" te voorspellen. Oplossing: `class_weight='balanced'`
# in het model, wat de minderheidsklasse extra gewicht geeft.
#
# **Irrelevante kolommen**: `UDI` en `Product ID` zijn identifiers,
# geen voorspellers. Die laten we weg uit de features.
#
# **Eenheden**: Temperaturen in Kelvin, snelheid in rpm, koppel in Nm.
# De ranges zijn consistent en er zitten geen uitschieters in die
# wijzen op foute eenheden (bv. Celsius i.p.v. Kelvin).

# %%
print("Missende waarden per kolom:")
print(df.isnull().sum())

# %% [markdown]
# ## 8. Conclusie
#
# De data is schoon en compleet. De voornaamste uitdagingen zijn:
# 1. **Imbalanced classes** → opgelost met class_weight en recall als metriek
# 2. **Data leakage risico** → sub-faalkolommen uitgesloten uit features
# 3. **Lage drempelwaarde** → threshold op 0.15 om meer defecten te vangen
