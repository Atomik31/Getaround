"""
Script de test de l'API GetAround.
Lancer d'abord l'API avec : uvicorn api:app --port 8000
"""
import requests

# Test avec une seule voiture
response = requests.post("http://localhost:8000/predict", json=[
    {
        "model_key": "Citroën",
        "mileage": 13929,
        "engine_power": 317,
        "fuel": "petrol",
        "paint_color": "grey",
        "car_type": "convertible",
        "private_parking_available": True,
        "has_gps": True,
        "has_air_conditioning": False,
        "automatic_car": False,
        "has_getaround_connect": False,
        "has_speed_regulator": True,
        "winter_tires": True
    }
])

print("Status :", response.status_code)
print("Réponse :", response.json())

# Test avec plusieurs voitures
response2 = requests.post("http://localhost:8000/predict", json=[
    {
        "model_key": "BMW",
        "mileage": 5000,
        "engine_power": 200,
        "fuel": "petrol",
        "paint_color": "black",
        "car_type": "sedan",
        "private_parking_available": True,
        "has_gps": True,
        "has_air_conditioning": True,
        "automatic_car": True,
        "has_getaround_connect": True,
        "has_speed_regulator": True,
        "winter_tires": False
    },
    {
        "model_key": "Renault",
        "mileage": 80000,
        "engine_power": 90,
        "fuel": "diesel",
        "paint_color": "white",
        "car_type": "hatchback",
        "private_parking_available": False,
        "has_gps": False,
        "has_air_conditioning": True,
        "automatic_car": False,
        "has_getaround_connect": False,
        "has_speed_regulator": False,
        "winter_tires": True
    }
])

print("\nTest multi-voitures :")
print("Status :", response2.status_code)
print("Réponse :", response2.json())
