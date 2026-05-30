"""
Generate all figures for the GetAround delay & pricing project.
Run with: /opt/anaconda3/bin/python generate_figures.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os

FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(FIGURES_DIR, '../../data')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})

BLUE  = '#317AC1'
RED   = '#eb2f2f'
ORANGE = '#eb7a2f'
GREEN  = '#2ca02c'

# ═══════════════════════════════════════════════════════════════════════════════
# PARTIE 1 — ANALYSE DES RETARDS
# ═══════════════════════════════════════════════════════════════════════════════

df = pd.read_excel(os.path.join(DATA_DIR, 'get_around_delay_analysis.xlsx'))
df['delay_at_checkout_in_minutes'] = df['delay_at_checkout_in_minutes'].fillna(0)
df['on_time'] = df['delay_at_checkout_in_minutes'].apply(
    lambda x: 'A l\'heure / en avance' if x <= 0 else 'En retard'
)

# ── 1. Répartition checkin type ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

checkin_counts = df['checkin_type'].value_counts()
axes[0].pie(checkin_counts.values,
            labels=checkin_counts.index,
            colors=[BLUE, ORANGE],
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[0].set_title('Répartition Mobile / Connect', fontsize=12, fontweight='bold')

status_data = df.groupby(['checkin_type', 'state']).size().unstack(fill_value=0)
status_data.plot(kind='bar', ax=axes[1], color=[RED, BLUE], rot=0, width=0.5,
                 edgecolor='white')
axes[1].set_title('Statut des locations par type de checkin', fontsize=12, fontweight='bold')
axes[1].set_xlabel('')
axes[1].set_ylabel('Nombre de locations')
axes[1].legend(['Annulée', 'Terminée'])

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '01_repartition_checkin.png'), bbox_inches='tight')
plt.close()
print("01_repartition_checkin.png OK")

# ── 2. Fréquence des retards ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

on_time_counts = df['on_time'].value_counts()
axes[0].pie(on_time_counts.values,
            labels=on_time_counts.index,
            colors=[BLUE, RED],
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[0].set_title('Proportion de retards au checkout', fontsize=12, fontweight='bold')

delays = df[df['delay_at_checkout_in_minutes'] > 0]['delay_at_checkout_in_minutes']
delays_filtered = delays[delays < 720]
axes[1].hist(delays_filtered, bins=50, color=RED, edgecolor='white', alpha=0.85)
med = delays_filtered.median()
axes[1].axvline(med, color='black', linestyle='--', linewidth=1.5,
                label=f'Médiane : {med:.0f} min')
axes[1].set_title('Distribution des retards au checkout (< 12h)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Retard (minutes)')
axes[1].set_ylabel('Nombre de locations')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '02_analyse_retards.png'), bbox_inches='tight')
plt.close()
print("02_analyse_retards.png OK")

# ── 3. Retards par type de checkin ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))

categories = ['Mobile', 'Connect']
late_pcts = []
for ctype in ['mobile', 'connect']:
    sub = df[df['checkin_type'] == ctype]
    late_pct = (sub['delay_at_checkout_in_minutes'] > 0).sum() / len(sub) * 100
    late_pcts.append(late_pct)

bars = ax.bar(categories, late_pcts, color=[BLUE, ORANGE], width=0.4, edgecolor='white')
for bar, pct in zip(bars, late_pcts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{pct:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_title('Taux de retard par type de checkin', fontsize=13, fontweight='bold')
ax.set_ylabel('% de locations en retard')
ax.set_ylim(0, max(late_pcts) * 1.2)
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '03_retards_par_type.png'), bbox_inches='tight')
plt.close()
print("03_retards_par_type.png OK")

# ── 4. Simulation du seuil ────────────────────────────────────────────────────
df_prev = df[['rental_id', 'checkin_type', 'state',
              'delay_at_checkout_in_minutes']].copy()
df_prev.columns = ['previous_ended_rental_id', 'previous_checkin_type',
                   'previous_state', 'previous_delay_at_checkout_in_minutes']

full_df = df.merge(df_prev, on='previous_ended_rental_id', how='left')
full_df['time_delta_with_previous_rental_in_minutes'] = \
    full_df['time_delta_with_previous_rental_in_minutes'].fillna(1440)

full_df['delayed_checkin'] = (
    full_df['previous_delay_at_checkout_in_minutes'] -
    full_df['time_delta_with_previous_rental_in_minutes']
).clip(lower=0)

impacted = full_df[full_df['delayed_checkin'] > 0]

thresholds = [30, 60, 90, 120, 180, 240]
scopes = {
    'Tous': full_df,
    'Connect uniquement': full_df[full_df['checkin_type'] == 'connect'],
    'Mobile uniquement' : full_df[full_df['checkin_type'] == 'mobile'],
}

results = []
for scope_name, scope_df in scopes.items():
    impacted_scope = scope_df[scope_df['delayed_checkin'] > 0]
    canceled_scope = impacted_scope[impacted_scope['state'] == 'canceled']
    for thr in thresholds:
        avoided_delays  = len(impacted_scope[impacted_scope['delayed_checkin'] < thr])
        avoided_cancel  = len(canceled_scope[canceled_scope['delayed_checkin'] < thr])
        lost_rentals    = len(scope_df[scope_df['time_delta_with_previous_rental_in_minutes'] < thr])
        results.append({
            'Périmètre': scope_name,
            'Seuil (min)': thr,
            'Retards évités': avoided_delays,
            'Annulations évitées': avoided_cancel,
            'Locations perdues': lost_rentals,
        })

results_df = pd.DataFrame(results)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors_scope = [BLUE, ORANGE, RED]

for ax, metric in zip(axes, ['Retards évités', 'Locations perdues']):
    for (scope, grp), color in zip(results_df.groupby('Périmètre'), colors_scope):
        ax.plot(grp['Seuil (min)'], grp[metric], marker='o', label=scope, color=color)
    ax.set_title(f'{metric} selon le seuil', fontsize=12, fontweight='bold')
    ax.set_xlabel('Seuil (minutes)')
    ax.set_ylabel(metric)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, linestyle='--')

plt.suptitle('Simulation du seuil de délai minimum entre deux locations',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '04_simulation_seuil.png'), bbox_inches='tight')
plt.close()
print("04_simulation_seuil.png OK")

# ═══════════════════════════════════════════════════════════════════════════════
# PARTIE 2 — MODÈLE PRICING
# ═══════════════════════════════════════════════════════════════════════════════

df_price = pd.read_csv(os.path.join(DATA_DIR, 'get_around_pricing_project.csv'), index_col=0)
df_price = df_price[(df_price['mileage'] >= 0) & (df_price['engine_power'] > 0)]

# ── 5. Distribution du prix ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df_price['rental_price_per_day'], bins=50, color=BLUE,
             edgecolor='white', alpha=0.85)
med_price = df_price['rental_price_per_day'].median()
axes[0].axvline(med_price, color='red', linestyle='--', linewidth=1.5,
                label=f'Médiane : {med_price:.0f} €')
axes[0].set_title('Distribution du prix de location journalier', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Prix (€/jour)')
axes[0].set_ylabel('Nombre de voitures')
axes[0].legend()

brand_median = df_price.groupby('model_key')['rental_price_per_day'].median().sort_values(ascending=False)
axes[1].bar(range(len(brand_median)), brand_median.values, color=BLUE, edgecolor='white')
axes[1].set_xticks(range(len(brand_median)))
axes[1].set_xticklabels(brand_median.index, rotation=45, ha='right', fontsize=8)
axes[1].set_title('Prix médian par marque', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Prix médian (€/jour)')
axes[1].grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '05_distribution_prix.png'), bbox_inches='tight')
plt.close()
print("05_distribution_prix.png OK")

# ── 6. Corrélation ────────────────────────────────────────────────────────────
numeric_cols = df_price.select_dtypes(include=[np.number]).columns.tolist()
corr_matrix = df_price[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', center=0, ax=ax,
            linewidths=0.5, linecolor='white',
            annot_kws={'size': 9})
ax.set_title('Matrice de corrélation — variables numériques', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '06_correlation_matrix.png'), bbox_inches='tight')
plt.close()
print("06_correlation_matrix.png OK")

# ── 7. Impact des équipements ─────────────────────────────────────────────────
boolean_cols = ['private_parking_available', 'has_gps', 'has_air_conditioning',
                'automatic_car', 'has_getaround_connect', 'has_speed_regulator',
                'winter_tires']

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, col in enumerate(boolean_cols):
    groups = [
        df_price[df_price[col] == False]['rental_price_per_day'],
        df_price[df_price[col] == True]['rental_price_per_day'],
    ]
    bp = axes[i].boxplot(groups, patch_artist=True, widths=0.5,
                         medianprops={'color': 'black', 'linewidth': 2})
    for patch, color in zip(bp['boxes'], [RED, BLUE]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[i].set_title(col.replace('_', ' '), fontsize=9, fontweight='bold')
    axes[i].set_xticklabels(['Non', 'Oui'])
    axes[i].set_ylabel('Prix (€/j)' if i % 4 == 0 else '')
    axes[i].grid(axis='y', alpha=0.3, linestyle='--')

axes[-1].axis('off')
plt.suptitle('Impact des équipements sur le prix de location', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '07_impact_equipements.png'), bbox_inches='tight')
plt.close()
print("07_impact_equipements.png OK")

# ── 8. RandomForest — Feature importance & résidus ───────────────────────────
target = 'rental_price_per_day'
Y = df_price[target]
X = df_price.drop(target, axis=1)

numeric_features = [c for c in X.columns if X[c].dtype in [np.float64, np.int64]]
categorical_features = [c for c in X.columns if X[c].dtype == object or X[c].dtype == bool]

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False),
     categorical_features),
])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train_t = preprocessor.fit_transform(X_train)
X_test_t  = preprocessor.transform(X_test)

rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train_t, Y_train)
Y_pred = rf.predict(X_test_t)

r2  = r2_score(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
rmse = mean_squared_error(Y_test, Y_pred) ** 0.5

# Feature importance — top 15 (sur features numériques uniquement)
feat_names = (numeric_features +
              list(preprocessor.named_transformers_['cat']
                   .get_feature_names_out(categorical_features)))
importances = rf.feature_importances_
top_idx = np.argsort(importances)[-15:]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].barh([feat_names[i] for i in top_idx],
             importances[top_idx],
             color=BLUE, edgecolor='white')
axes[0].set_title('Feature Importance — RandomForest (Top 15)',
                  fontsize=12, fontweight='bold')
axes[0].set_xlabel('Importance')
axes[0].grid(axis='x', alpha=0.3, linestyle='--')

residuals = Y_test - Y_pred
axes[1].scatter(Y_pred, residuals, alpha=0.3, s=15, color=BLUE)
axes[1].axhline(0, color='red', linestyle='--', linewidth=1.5)
axes[1].set_title(f'Résidus — RandomForest\nR²={r2:.3f}  MAE={mae:.1f}€  RMSE={rmse:.1f}€',
                  fontsize=12, fontweight='bold')
axes[1].set_xlabel('Valeurs prédites (€/jour)')
axes[1].set_ylabel('Résidus (réel - prédit)')
axes[1].grid(alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '08_rf_feature_importance_residus.png'), bbox_inches='tight')
plt.close()
print(f"08_rf_feature_importance_residus.png OK  (R²={r2:.3f}, MAE={mae:.1f}€, RMSE={rmse:.1f}€)")

# ── 9. Prédictions vs réalité ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(Y_test, Y_pred, alpha=0.3, s=15, color=BLUE)
min_val = min(Y_test.min(), Y_pred.min())
max_val = max(Y_test.max(), Y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5,
        label='Prédiction parfaite')
ax.set_title(f'Prédictions vs Réalité — RandomForest\nR²={r2:.3f}  MAE={mae:.1f}€/jour',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Prix réel (€/jour)')
ax.set_ylabel('Prix prédit (€/jour)')
ax.legend()
ax.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '09_predictions_vs_realite.png'), bbox_inches='tight')
plt.close()
print("09_predictions_vs_realite.png OK")

# ── 10. Comparaison modèles (barres R² / MAE) ────────────────────────────────
# Valeurs issues du MLflow tracking (runs enregistrés dans le projet)
model_names  = ['LinReg', 'Ridge', 'Lasso', 'RandomForest\n(max_depth=10)', 'RF\n(GridSearchCV)']
r2_scores    = [0.658, 0.658, 0.601, 0.739, 0.762]
mae_scores   = [17.2,  17.2,  19.1,  12.4,  11.6]

x = np.arange(len(model_names))
width = 0.35

fig, ax1 = plt.subplots(figsize=(11, 5))
ax2 = ax1.twinx()

b1 = ax1.bar(x - width/2, r2_scores, width, label='R²', color=BLUE,
             alpha=0.85, edgecolor='white')
b2 = ax2.bar(x + width/2, mae_scores, width, label='MAE (€/j)', color=ORANGE,
             alpha=0.85, edgecolor='white')

ax1.set_ylabel('R²', color=BLUE)
ax2.set_ylabel('MAE (€/jour)', color=ORANGE)
ax1.set_ylim(0.4, 0.9)
ax2.set_ylim(0, 30)
ax1.set_xticks(x)
ax1.set_xticklabels(model_names, fontsize=10)
ax1.set_title('Comparaison des modèles de pricing', fontsize=13, fontweight='bold')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
ax1.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '10_comparaison_modeles_pricing.png'), bbox_inches='tight')
plt.close()
print("10_comparaison_modeles_pricing.png OK")

print(f"\nFigures Bloc-5 GetAround generees dans : {FIGURES_DIR}")
