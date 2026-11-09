# Predictive Maintenance — Machine Failure Prediction

Voorspel of een machine binnenkort zal falen op basis van sensordata, zodat een technicus
gericht kan worden ingeschakeld vóór het defect optreedt. Gebouwd als binair classificatiemodel
op de [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset) (10.000 rijen sensordata van freesmachines).

**Auteur:** Joussef Boulbane  
**Repository:** [github.com/JoussefB/cloud-for-ai](https://github.com/JoussefB/cloud-for-ai)

---

## Snel starten

```bash
# 1. Virtuele omgeving aanmaken en activeren
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate

# 2. Afhankelijkheden installeren
pip install -r requirements.txt

# 3. Dataset downloaden (indien nog niet aanwezig)
python -m src.download_data

# 4. Tests draaien (12 tests voor Pandera-schema, invarianten en pijplijn)
pytest                          # 12 passed

# 5. Stijl- en codekwaliteitscontroles
ruff check .
ruff format --check .

# 6. Model trainen (vergelijkt baseline vs challenger, slaat het beste model op)
python -m src.train

# 7. Voorspellen op data (berekent storingskansen met lage drempelwaarde 0.15)
python -m src.predict data/ai4i2020.csv

# 8. Het 2-pagina's PDF-rapport en de figuren opnieuw genereren
python -m src.generate_report
```

Draai alle commando's **vanuit de projectmap**, niet vanuit `src/`.

---

## Projectstructuur

Conform de indeling uit les 5:

```
configs/
  default.yaml                 paden, features, doel, seed, drempelwaarde (0.15), minimum recall
ruff.toml                      linter- en formatterregels
src/
  __init__.py                  pakketdefinitie
  config.py                    inlezen en valideren van de YAML-config
  schema.py                    Pandera-schema: types, ID-uniciteit, categorieën en fysieke grenzen
  data.py                      data laden en lazy schema-validatie
  train.py                     modellen trainen, vergelijken op Recall en wegschrijven
  predict.py                   opgeslagen model laden en batches evalueren
  download_data.py             dataset downloaden van het officiële UCI Machine Learning Repository
  generate_report.py           grafieken en het 2-pagina's PDF-rapport bouwen (ReportLab)
tests/
  conftest.py                  herberekende pytest fixtures (data en getrainde pijplijn)
  test_data.py                 schema-tests (weigering bij foute types, hernoemde kolommen, extreme eenheden)
  test_model.py                ML-tests (uitvoervorm, kansinvarianten [0, 1], persistentie, ondergrens)
notebooks/
  exploratie.py                data-exploratie, correlaties, klasse-onbalans (draaibaar als notebook)
  modelvergelijking.png        vergelijkende verwarringsmatrices bij drempel 0.15
data/
  ai4i2020.csv                 de AI4I 2020 Predictive Maintenance dataset (10.000 rijen)
models/                        bevat het opgeslagen model.joblib (aangemaakt door src.train)
Cloud_for_AI_Deel_2_Rapport.pdf  het officiële 2-pagina's evaluatierapport
```

---

## Modellen & Resultaten

Twee modellen worden vergeleken op de testset (2.000 rijen, 68 defecten) bij de in Deel 1 gekozen drempelwaarde van **0.15**:

| Model | Recall (drempel 0.15) | Precision | F1-Score | Accuracy | Gemiste Defecten (FN) | Valse Alarmen (FP) |
|---|---|---|---|---|---|---|
| **Logistic Regression (Baseline)** | **95.6%** (65/68) | 6.2% | 0.117 | 50.8% | 3 | 979 |
| **Random Forest (Gekozen)** | **91.2%** (62/68) | **34.8%** | **0.504** | **94.0%** | 6 | **116** |

### Waarom we Random Forest behouden
Hoewel de baseline op papier fractioneel hoger scoort op ruwe recall (95.6% vs 91.2%), veroorzaakt het een onacceptabel aantal valse alarmen (979 op 1.932 gezonde machines, precisie slechts 6.2%). Dit leidt onherroepelijk tot **alert fatigue** (meldingsmoeheid bij technici), waardoor meldingen genegeerd worden.  

**Random Forest** vangt ruim 91% van alle storingen en vermindert het aantal valse alarmen met **88%** (van 979 naar 116). Financieel levert dit een **netto meerwaarde van €99.450** op over de testset. Bovendien blijft de inferentietijd onder **1 milliseconde per machine**, wat ruimschoots voldoet aan de softwaremetriek (< 1 seconde).

---

## Data Leakage & Eerlijke Split

1. **Waar zat het risico in de data?**  
   De AI4I 2020 dataset bevat 5 storingsmodus-kolommen: `TWF`, `HDF`, `PWF`, `OSF` en `RNF`. Deze kolommen zijn deterministische sub-labels van de doelvariabele `Machine failure`. Zouden deze als feature worden meegegeven, dan ontstaat 100% data leakage: het model leert triviale identiteitsregels die in een operationele omgeving waardeloos zijn omdat deze diagnoses pas *na demontage* gekend zijn. Deze kolommen en identifiers (`UDI`, `Product ID`) zijn daarom strikt uitgesloten in `configs/default.yaml`.

2. **Hoe weten we dat de split eerlijk is?**  
   We passen een gestratificeerde split toe (`train_test_split(..., stratify=y, test_size=0.2, random_state=42)`), zodat beide subsets exact 3.39% storingen bevatten. Bovendien zitten alle voorbewerkingen (`StandardScaler` en `OneHotEncoder`) hermetisch opgesloten in een Scikit-learn `Pipeline`. De transformers worden uitsluitend gefit op trainingsdata en transformeren testdata blind, waardoor er geen gemiddelden of varianties lekken.

---

## Indienen

- **Repository:** https://github.com/JoussefB/cloud-for-ai
- **PDF-bestand:** [`Cloud_for_AI_Deel_2_Rapport.pdf`](./Cloud_for_AI_Deel_2_Rapport.pdf) (exact 2 pagina's)
