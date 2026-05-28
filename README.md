# GetAround — Retards & pricing

Projet réalisé dans le cadre du bloc 5 de la certification CDSD (Jedha).

---

## Contexte

GetAround c'est l'Airbnb des voitures. Le problème récurrent : des conducteurs rendent le véhicule en retard, ce qui bloque la location suivante. Ce projet répond à deux questions concrètes posées par le Product Manager :

1. Quel délai minimum imposer entre deux locations, et sur quel périmètre ?
2. Comment estimer automatiquement le prix de location optimal d'une voiture ?

---

## Ce que j'ai fait

**Dashboard (Streamlit)**

Analyse des retards sur 21 000+ locations avec simulation interactive du seuil : combien de checkins problématiques sont résolus, combien de locations sont perdues, selon le délai et le type de contrat (Connect / Mobile).

**API de prédiction (FastAPI)**

Endpoint `/predict` qui prend les caractéristiques d'une voiture (marque, km, puissance, équipements...) et retourne une estimation du prix journalier en euros.

**Tracking ML (MLflow)**

Comparaison de 6 modèles (LinearRegression, Ridge, Lasso, RandomForest, GridSearchCV RF, XGBoost) avec métriques loggées sur un serveur MLflow dédié.

---

## Démos en ligne

| Service | URL |
|---------|-----|
| Dashboard | https://huggingface.co/spaces/Atomik31/getaround-dashboard |
| API + docs | https://atomik31-getaround-api.hf.space/docs |
| MLflow | https://atomik31-mlflow.hf.space |

---

## Modèle retenu

Random Forest avec GridSearchCV — R² ≈ 0.74, MAE ≈ 10.76 €/jour. Le preprocessing (StandardScaler + OneHotEncoder) et le modèle sont sérialisés en `.joblib` et chargés au démarrage de l'API.

---

## Stack

- Python — FastAPI, Streamlit, Scikit-learn, MLflow, Docker
- Données : 21 310 locations (retards) + 4 841 voitures (pricing)

---

Julien CHARLIER — [(Github : Atomik31)](https://github.com/Atomik31)
