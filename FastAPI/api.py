import pandas as pd
import joblib
import uvicorn
from pydantic import BaseModel
from typing import Literal, List, Union
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import RedirectResponse

description = """
Estimation du prix de location journalier d'une voiture à partir de ses caractéristiques.

Utilisez l'endpoint **`POST /predict`** en transmettant les attributs du véhicule pour obtenir un prix suggéré en euros par jour.
"""

tags_metadata = [
    {
        "name": "Prédictions",
        "description": "Estimation du prix de location journalier."
    }
]

app = FastAPI(
    title="GetAround — Prédiction du prix de location",
    description=description,
    version="1.0",
    openapi_tags=tags_metadata
)


class Car(BaseModel):
    model_key: Literal['Citroën', 'Peugeot', 'PGO', 'Renault', 'Audi', 'BMW', 'Mercedes',
                       'Opel', 'Volkswagen', 'Ferrari', 'Mitsubishi', 'Nissan', 'SEAT',
                       'Subaru', 'Toyota', 'other']
    mileage: Union[int, float]
    engine_power: Union[int, float]
    fuel: Literal['diesel', 'petrol', 'other']
    paint_color: Literal['black', 'grey', 'white', 'red', 'silver', 'blue', 'beige', 'brown', 'other']
    car_type: Literal['convertible', 'coupe', 'estate', 'hatchback', 'sedan', 'subcompact', 'suv', 'van']
    private_parking_available: bool
    has_gps: bool
    has_air_conditioning: bool
    automatic_car: bool
    has_getaround_connect: bool
    has_speed_regulator: bool
    winter_tires: bool


# Chargement du modèle et du preprocessor au démarrage (une seule fois)
preprocessor = joblib.load("preprocessor.joblib")
model = joblib.load("model.joblib")


@app.get("/", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url='/docs')


@app.post("/predict", tags=["Prédictions"])
async def predict(cars: List[Car]):
    """
    Retourne le prix de location journalier estimé pour une ou plusieurs voitures.

    **Entrée** : liste de voitures avec leurs caractéristiques (voir schéma ci-dessous)

    **Sortie** : `{"prediction": [prix_en_euros, ...]}`
    """
    car_features = pd.DataFrame(jsonable_encoder(cars))
    car_features_transformed = preprocessor.transform(car_features)
    prediction = model.predict(car_features_transformed)
    return {"prediction": prediction.tolist()}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
