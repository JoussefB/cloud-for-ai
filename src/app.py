"""FastAPI REST API voor Predictive Maintenance Machine Failure Prediction.

Draaien met:
uvicorn src.app:app --host 0.0.0.0 --port 7860 --reload
"""

from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from src.config import alle_features, laad_config
from src.predict import laad_model

# Globale context voor het model en configuratie
model_pipeline: Any = None
cfg: dict[str, Any] = {}


def laad_model_en_config():
    """Hulpfunctie om model en configuratie te initialiseren indien nog niet geladen."""
    global model_pipeline, cfg
    if not cfg:
        cfg = laad_config()
    if model_pipeline is None:
        model_pad = cfg["model"]["pad"]
        if not model_pad.exists():
            raise RuntimeError(
                f"Modelbestand niet gevonden op {model_pad}. Draai eerst 'python -m src.train'."
            )
        model_pipeline = laad_model(model_pad)
    return model_pipeline, cfg


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Laad het getrainde model en configuratie bij het opstarten van de applicatie."""
    laad_model_en_config()
    yield


app = FastAPI(
    title="Predictive Maintenance API",
    description=(
        "Near-realtime ML endpoint voor het voorspellen van machinefalen bij freesmachines "
        "(BMSVision use-case, AI4I 2020 dataset). Evaluatie op Recall bij drempelwaarde 0.15."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware toestaan zodat clients vanuit browsers en webapps requests kunnen sturen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Pydantic Schemas voor Invoer & Uitvoer
# ==========================================


class MachineSensorData(BaseModel):
    """Sensordata voor één machine-observatie.

    Ondersteunt zowel de originele dataset-kolomnamen als clean snake_case aliassen.
    """

    model_config = ConfigDict(populate_by_name=True)

    type: Annotated[
        Literal["L", "M", "H"],
        Field(
            alias="Type",
            description="Kwaliteitsvariant van het product: L (Low 50%), M (Medium 30%), H (High 20%).",
            examples=["L"],
        ),
    ]
    air_temperature_k: Annotated[
        float,
        Field(
            alias="Air temperature [K]",
            ge=250.0,
            le=350.0,
            description="Omgevingstemperatuur in Kelvin (fysisch bereik: 250 - 350 K).",
            examples=[298.1],
        ),
    ]
    process_temperature_k: Annotated[
        float,
        Field(
            alias="Process temperature [K]",
            ge=250.0,
            le=350.0,
            description="Procestemperatuur in Kelvin (fysisch bereik: 250 - 350 K).",
            examples=[308.6],
        ),
    ]
    rotational_speed_rpm: Annotated[
        int,
        Field(
            alias="Rotational speed [rpm]",
            ge=500,
            le=4000,
            description="Draaisnelheid van de spindel in rpm (500 - 4000 rpm).",
            examples=[1551],
        ),
    ]
    torque_nm: Annotated[
        float,
        Field(
            alias="Torque [Nm]",
            ge=0.0,
            le=150.0,
            description="Geleverd koppel in Newtonmeter (0 - 150 Nm).",
            examples=[42.8],
        ),
    ]
    tool_wear_min: Annotated[
        int,
        Field(
            alias="Tool wear [min]",
            ge=0,
            le=400,
            description="Cumulatieve snijtijd van het gereedschap in minuten (0 - 400 min).",
            examples=[0],
        ),
    ]


class PredictionResult(BaseModel):
    """Resultaat van een predictie met berekende faalkans en alarmstatus."""

    failure_probability: float = Field(
        description="Berekende kans op machinefalen door het model (tussen 0.0 en 1.0)."
    )
    threshold: float = Field(
        default=0.15,
        description="Beslissingsdrempel (0.15 wegens asymmetrische kosten €150 FP vs €10.000 FN).",
    )
    alarm: bool = Field(
        description="True als failure_probability >= threshold (machine dreigt uit te vallen)."
    )
    status: Literal["WARNING_FAILURE_PREDICTED", "NORMAL_OPERATION"] = Field(
        description="Categorische storingsstatus van de machine."
    )
    action: str = Field(description="Aanbevolen actie voor de onderhoudstechnicus.")


class BatchPredictionResult(BaseModel):
    """Batch resultaat voor meerdere machines tegelijk."""

    count: int
    alarms_count: int
    predictions: list[PredictionResult]


# ==========================================
# Inputvalidatie & Foutafhandeling
# ==========================================


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError):
    """Nette JSON foutmelding bij ongeldige invoer (geen 500 stacktrace)."""
    errors = []
    for error in exc.errors():
        veld = " -> ".join(str(loc) for loc in error["loc"] if loc != "body")
        errors.append(
            {
                "field": veld or "payload",
                "message": error["msg"],
                "type": error["type"],
            }
        )
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validatiefout in invoergegevens",
            "message": "Een of meerdere sensormetingen zijn ongeldig of ontbreken.",
            "details": errors,
        },
    )


# ==========================================
# Hulpfuncties
# ==========================================


def converteer_naar_dataframe(data: MachineSensorData) -> pd.DataFrame:
    """Zet een gevalideerd Pydantic object om naar een DataFrame met de exacte featurenamen."""
    row = {
        "Type": data.type,
        "Air temperature [K]": data.air_temperature_k,
        "Process temperature [K]": data.process_temperature_k,
        "Rotational speed [rpm]": data.rotational_speed_rpm,
        "Torque [Nm]": data.torque_nm,
        "Tool wear [min]": data.tool_wear_min,
    }
    df = pd.DataFrame([row])
    # Controleer of alle vereiste model features aanwezig zijn
    features = alle_features(cfg)
    return df[features]


def bereken_predictie(prob: float, threshold: float = 0.15) -> PredictionResult:
    """Bouwt het gestructureerde resultaat op basis van de kans en de drempelwaarde."""
    alarm = bool(prob >= threshold)
    return PredictionResult(
        failure_probability=round(float(prob), 4),
        threshold=threshold,
        alarm=alarm,
        status="WARNING_FAILURE_PREDICTED" if alarm else "NORMAL_OPERATION",
        action=(
            "Spoedinspectie vereist: machine dreigt uit te vallen (preventieve interventie ca. €150)."
            if alarm
            else "Geen actie vereist: machine functioneert binnen veilige operationele grenzen."
        ),
    )


# ==========================================
# API Endpoints
# ==========================================


@app.get("/", tags=["Info"])
def root():
    """Welkomstpagina met metadata en link naar Swagger docs."""
    return {
        "project": "Cloud for AI — Predictive Maintenance API",
        "author": "Joussef Boulbane",
        "repo": "https://github.com/JoussefB/cloud-for-ai",
        "use_case": "Predictive maintenance voor freesmachines (BMSVision scenario)",
        "model_type": "Random Forest Classifier (200 bomen, balanced class weights)",
        "decision_threshold": cfg.get("threshold", 0.15),
        "docs_url": "/docs",
        "health_check": "/health",
    }


@app.get("/health", tags=["Monitoring"])
def health():
    """Health-check endpoint voor uptime monitoring en container orchestration."""
    pipeline, c = laad_model_en_config()
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None,
        "threshold": c.get("threshold", 0.15),
    }


@app.post(
    "/predict",
    response_model=PredictionResult,
    status_code=status.HTTP_200_OK,
    tags=["Predictie"],
)
def predict(sensor_data: MachineSensorData):
    """Voorspel het faalrisico voor één machine-observatie.

    Accepteert zowel 'Air temperature [K]' als 'air_temperature_k'.
    Geeft faalkans, alarm en een concreet actieadvies terug.
    """
    pipeline, c = laad_model_en_config()
    df = converteer_naar_dataframe(sensor_data)
    threshold = c.get("threshold", 0.15)
    prob = float(pipeline.predict_proba(df)[:, 1][0])

    return bereken_predictie(prob, threshold)


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResult,
    status_code=status.HTTP_200_OK,
    tags=["Predictie"],
)
def predict_batch(batch: list[MachineSensorData]):
    """Voorspel faalrisico voor een batch van meerdere machines tegelijk."""
    pipeline, c = laad_model_en_config()

    if not batch:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="De verzonden batch bevat geen data.",
        )

    threshold = c.get("threshold", 0.15)
    rows = []
    for item in batch:
        rows.append(
            {
                "Type": item.type,
                "Air temperature [K]": item.air_temperature_k,
                "Process temperature [K]": item.process_temperature_k,
                "Rotational speed [rpm]": item.rotational_speed_rpm,
                "Torque [Nm]": item.torque_nm,
                "Tool wear [min]": item.tool_wear_min,
            }
        )

    df_batch = pd.DataFrame(rows)[alle_features(c)]
    probs = pipeline.predict_proba(df_batch)[:, 1]

    results = [bereken_predictie(float(p), threshold) for p in probs]
    alarms = sum(1 for r in results if r.alarm)

    return BatchPredictionResult(
        count=len(results),
        alarms_count=alarms,
        predictions=results,
    )
