import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from PIL import Image
from data_calc import load_data, filter_scope

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GetAround Dashboard",
    page_icon=":blue_car:",
    layout="wide",
)

API_URL = "https://atomik31-getaround-api.hf.space/predict"

# ── Sidebar ───────────────────────────────────────────────────────────────────
try:
    logo = Image.open("GetAround_logo.png")
    st.sidebar.image(logo, use_container_width=True)
except Exception:
    st.sidebar.title("GetAround")

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["📊 Analyse des délais", "💰 Prédiction du prix"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Julien CHARLIER — Bloc 5 CDSD")


# ── Chargement des données ────────────────────────────────────────────────────
@st.cache_data
def get_data():
    return load_data()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — ANALYSE DES DÉLAIS
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Analyse des délais":

    st.title("Analyse des retards GetAround ⏱")
    st.markdown(
        "Outil d'aide à la décision pour le Product Manager : "
        "quel **seuil** de délai minimum entre deux locations ? "
        "Quel **périmètre** d'application ?"
    )
    st.markdown("---")

    with st.spinner("Chargement des données..."):
        data = get_data()

    # ── KPIs globaux ──────────────────────────────────────────────────────────
    total = len(data)
    late = (data["previous_late"]).sum()
    impacted = (data["impacted"]).sum()

    k1, k2, k3 = st.columns(3)
    k1.metric("Locations consécutives analysées", f"{total:,}")
    k2.metric("Avec location précédente en retard", f"{late:,}", f"{late/total*100:.1f}%")
    k3.metric("Checkins impactés par ce retard", f"{impacted:,}", f"{impacted/total*100:.1f}%")

    st.markdown("---")

    # ── Filtres ───────────────────────────────────────────────────────────────
    col_slider, col_scope = st.columns([3, 1])
    with col_slider:
        threshold = st.slider(
            "Seuil de délai minimum (minutes)",
            min_value=0, max_value=720, value=60, step=30,
        )
    with col_scope:
        scope = st.selectbox("Périmètre", ["Tous", "Connect", "Mobile"])

    scoped_data = filter_scope(data, scope if scope != "Tous" else "all")

    # ── Métriques simulées ────────────────────────────────────────────────────
    if threshold > 0:
        lost = len(scoped_data[
            scoped_data["time_delta_with_previous_rental_in_minutes"] < threshold
        ])
        impacted_scope = scoped_data[scoped_data["impacted"]]
        avoided = len(impacted_scope[impacted_scope["overlap"] < threshold])
    else:
        lost, avoided = 0, 0

    m1, m2 = st.columns(2)
    m1.metric(
        "Locations perdues avec ce seuil",
        f"{lost:,}",
        delta=f"-{lost/len(scoped_data)*100:.1f}% du périmètre" if len(scoped_data) else "",
        delta_color="inverse",
    )
    m2.metric(
        "Checkins problématiques résolus",
        f"{avoided:,}",
        delta=f"+{avoided/max(impacted_scope.shape[0],1)*100:.1f}% des cas" if threshold > 0 else "",
    )

    st.markdown("---")

    # ── Graphiques ────────────────────────────────────────────────────────────
    st.subheader("Impact selon le seuil choisi")

    thresholds = list(range(0, 721, 30))
    rows = []
    for thr in thresholds:
        s = scoped_data
        l = len(s[s["time_delta_with_previous_rental_in_minutes"] < thr]) if thr > 0 else 0
        imp = s[s["impacted"]]
        av = len(imp[imp["overlap"] < thr]) if thr > 0 else 0
        rows.append({"Seuil (min)": thr, "Locations perdues": l, "Checkins résolus": av})

    sim_df = pd.DataFrame(rows)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sim_df["Seuil (min)"], y=sim_df["Locations perdues"],
        name="Locations perdues", marker_color="#eb2f2f", opacity=0.8,
    ))
    fig.add_trace(go.Bar(
        x=sim_df["Seuil (min)"], y=sim_df["Checkins résolus"],
        name="Checkins résolus", marker_color="#317AC1", opacity=0.8,
    ))
    fig.add_vline(x=threshold, line_dash="dash", line_color="orange",
                  annotation_text=f"Seuil actuel : {threshold} min")
    fig.update_layout(
        barmode="group",
        xaxis_title="Seuil (minutes)",
        yaxis_title="Nombre de locations",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Distribution des retards ──────────────────────────────────────────────
    st.subheader("Distribution des retards au checkout")

    delays = scoped_data[
        scoped_data["delay_at_checkout_in_minutes_previous"] > 0
    ]["delay_at_checkout_in_minutes_previous"]
    delays = delays[delays < 720]

    fig2 = px.histogram(
        delays, nbins=48,
        labels={"value": "Retard (minutes)", "count": "Nombre"},
        color_discrete_sequence=["#317AC1"],
    )
    fig2.add_vline(x=threshold, line_dash="dash", line_color="orange",
                   annotation_text=f"{threshold} min")
    fig2.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    # ── Répartition Mobile / Connect ──────────────────────────────────────────
    st.subheader("Répartition Mobile / Connect")
    col_pie1, col_pie2 = st.columns(2)

    raw_full = pd.read_excel(
        "https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/get_around_delay_analysis.xlsx"
    )
    with col_pie1:
        fig3 = px.pie(
            raw_full, names="checkin_type", hole=0.35,
            title="Type de checkin",
            color="checkin_type",
            color_discrete_map={"mobile": "#317AC1", "connect": "#eb7a2f"},
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_pie2:
        late_raw = raw_full.copy()
        late_raw["delay_at_checkout_in_minutes"] = late_raw["delay_at_checkout_in_minutes"].fillna(0)
        late_raw["statut"] = late_raw["delay_at_checkout_in_minutes"].apply(
            lambda x: "En retard" if x > 0 else "À l'heure"
        )
        fig4 = px.pie(
            late_raw, names="statut", hole=0.35,
            title="Proportion de retards",
            color="statut",
            color_discrete_map={"En retard": "#eb2f2f", "À l'heure": "#317AC1"},
        )
        st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PRÉDICTION DU PRIX
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.title("Prédiction du prix de location 💰")
    st.markdown(
        "Renseigne les caractéristiques de la voiture pour obtenir "
        "une estimation du **prix de location journalier optimal**."
    )
    st.markdown("---")

    with st.form("predict_form"):
        col1, col2 = st.columns(2)

        with col1:
            model_key = st.selectbox("Marque", [
                "Citroën", "Peugeot", "PGO", "Renault", "Audi", "BMW",
                "Mercedes", "Opel", "Volkswagen", "Ferrari", "Maserati",
                "Mitsubishi", "Nissan", "SEAT", "Subaru", "Toyota",
                "other",
            ])
            mileage = st.number_input("Kilométrage", min_value=0, max_value=500000, value=80000, step=1000)
            engine_power = st.number_input("Puissance moteur (ch)", min_value=50, max_value=500, value=120)
            fuel = st.selectbox("Carburant", ["diesel", "petrol", "hybrid_petrol", "electro"])
            paint_color = st.selectbox("Couleur", [
                "black", "white", "grey", "blue", "red", "silver",
                "beige", "brown", "green", "orange",
            ])
            car_type = st.selectbox("Type", [
                "sedan", "hatchback", "suv", "van", "estate",
                "convertible", "coupe", "subcompact",
            ])

        with col2:
            st.markdown("**Équipements**")
            private_parking = st.checkbox("Parking privé", value=True)
            has_gps = st.checkbox("GPS", value=True)
            has_ac = st.checkbox("Climatisation", value=True)
            automatic = st.checkbox("Boîte automatique", value=False)
            has_connect = st.checkbox("GetAround Connect", value=False)
            speed_reg = st.checkbox("Régulateur de vitesse", value=True)
            winter_tires = st.checkbox("Pneus hiver", value=True)

        submitted = st.form_submit_button("Estimer le prix", type="primary", use_container_width=True)

    if submitted:
        payload = [{
            "model_key": model_key,
            "mileage": mileage,
            "engine_power": engine_power,
            "fuel": fuel,
            "paint_color": paint_color,
            "car_type": car_type,
            "private_parking_available": private_parking,
            "has_gps": has_gps,
            "has_air_conditioning": has_ac,
            "automatic_car": automatic,
            "has_getaround_connect": has_connect,
            "has_speed_regulator": speed_reg,
            "winter_tires": winter_tires,
        }]
        try:
            resp = requests.post(API_URL, json=payload, timeout=10)
            if resp.status_code == 200:
                price = resp.json()["prediction"][0]
                st.success(f"### Prix estimé : **{price:.0f} €/jour**")
                st.caption("Estimation basée sur un modèle Random Forest entraîné sur 4 800+ voitures GetAround.")
            else:
                st.error(f"Erreur API ({resp.status_code}) : {resp.text}")
        except requests.exceptions.ConnectionError:
            st.error("Impossible de joindre l'API. Vérifie que le Space HuggingFace est actif.")
        except Exception as e:
            st.error(f"Erreur : {e}")
