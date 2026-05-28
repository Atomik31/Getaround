import pandas as pd

DATA_URL = "https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/get_around_delay_analysis.xlsx"


def load_data():
    raw = pd.read_excel(DATA_URL)
    data = raw[raw["state"] == "ended"].copy()
    data = data.merge(
        data[["rental_id", "delay_at_checkout_in_minutes", "checkin_type"]],
        how="left",
        left_on="previous_ended_rental_id",
        right_on="rental_id",
        suffixes=["", "_previous"],
    )
    data = data.drop("rental_id_previous", axis=1)
    data = data[data["delay_at_checkout_in_minutes_previous"].notna()]
    data = data[data["delay_at_checkout_in_minutes_previous"].between(-2000, 2000)]
    data["overlap"] = (
        data["delay_at_checkout_in_minutes_previous"]
        - data["time_delta_with_previous_rental_in_minutes"]
    )
    data["previous_late"] = data["delay_at_checkout_in_minutes_previous"] > 0
    data["impacted"] = data["overlap"] > 0
    return data


def filter_scope(data, scope):
    if scope.lower() == "connect":
        return data[data["checkin_type"] == "connect"]
    if scope.lower() == "mobile":
        return data[data["checkin_type"] == "mobile"]
    return data


def compute_metrics(data, threshold):
    """Retourne (locations_perdues, retards_evites, annulations_evitees)."""
    scoped = filter_scope(data, "all")
    lost = len(scoped[scoped["time_delta_with_previous_rental_in_minutes"] < threshold])
    impacted = scoped[scoped["impacted"]]
    avoided_delays = len(impacted[impacted["overlap"] < threshold])
    canceled = scoped[scoped["impacted"] & (scoped["state"] == "canceled")] if "state" in scoped.columns else pd.DataFrame()
    avoided_cancel = len(canceled[canceled["overlap"] < threshold]) if not canceled.empty else 0
    return lost, avoided_delays, avoided_cancel
