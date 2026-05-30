![GetAround](GetAround_logo.png)

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
| MLflow | https://atomik31-mlflow-cdsd.hf.space |

---

## Modèle retenu

Random Forest avec GridSearchCV — R² ≈ 0.74, MAE ≈ 10.76 €/jour.

Au démarrage, l'API charge le modèle directement depuis le **Model Registry MLflow** via l'alias `production` :

```python
model = mlflow.sklearn.load_model("models:/GetAround_price_predictor@production")
```

Cela signifie qu'il suffit de promouvoir un nouveau modèle en `production` dans le registry pour que l'API serve automatiquement la version la plus précise — sans redéploiement. Si on affine le modèle (nouveaux hyperparamètres, feature engineering, XGBoost...), il suffit d'entraîner un nouveau run MLflow, d'enregistrer le modèle dans le registry, puis de déplacer l'alias `production` vers cette nouvelle version.

---

## Stack

- Python — FastAPI, Streamlit, Scikit-learn, MLflow, Docker
- Données : 21 310 locations (retards) + 4 841 voitures (pricing)

---

## Structure

```
Deployment-GetAround/
├── data/
│   ├── get_around_delay_analysis.xlsx
│   └── get_around_pricing_project.csv
├── docs/
│   └── 01-Getaround_analysis.ipynb        # Énoncé du projet
├── notebooks/
│   └── getaround_eda.ipynb                # EDA retards + pricing
├── reports/
│   └── figures/
│       ├── 01_repartition_checkin.png
│       ├── 02_analyse_retards.png
│       ├── 03_retards_par_type.png
│       ├── 04_simulation_seuil.png
│       ├── 05_distribution_prix.png
│       ├── 06_correlation_matrix.png
│       ├── 07_impact_equipements.png
│       ├── 08_rf_feature_importance_residus.png
│       ├── 09_predictions_vs_realite.png
│       └── 10_comparaison_modeles_pricing.png
├── FastAPI/
│   ├── api.py
│   ├── request.py
│   ├── requirements.txt
│   └── Dockerfile
├── Streamlit/
│   ├── app.py
│   ├── data_calc.py
│   ├── requirements.txt
│   └── Dockerfile
├── MLflow/
│   ├── train.py
│   ├── train.ipynb
│   ├── requirements.txt
│   └── Dockerfile
├── GetAround_logo.png
└── README.md
```

---

Julien CHARLIER — [(Github : Atomik31)](https://github.com/Atomik31)
