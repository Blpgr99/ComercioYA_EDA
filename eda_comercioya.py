# eda_comercioya.py
# Proyecto EDA ComercioYA - módulo Análisis Exploratorio de Datos
# Autor: (tu nombre)
# Requisitos: numpy, pandas, matplotlib, seaborn, statsmodels, scikit-learn

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ------------------------------------------------------------
# LECCIÓN 1: Generar dataset + distinguir variables + IDA
# ------------------------------------------------------------

def generate_dataset(n: int = 2500) -> pd.DataFrame:
    """
    Genera un dataset sintético realista de e-commerce:
    - Compras, visitas, montos, devoluciones, reseñas, canal, categoría, membresía, región, etc.
    Incluye algunos valores faltantes e inconsistencias controladas para IDA.
    """
    customer_id = np.arange(1, n + 1)

    # Categóricas
    region = np.random.choice(["RM", "Valparaíso", "Biobío", "Araucanía", "Los Lagos"], size=n, p=[0.45, 0.18, 0.15, 0.12, 0.10])
    channel = np.random.choice(["Organic", "Ads", "Email", "Referral"], size=n, p=[0.42, 0.33, 0.15, 0.10])
    membership = np.random.choice(["No", "Sí"], size=n, p=[0.68, 0.32])
    category_pref = np.random.choice(["Tecnología", "Hogar", "Moda", "Deportes", "Belleza"], size=n, p=[0.30, 0.22, 0.20, 0.15, 0.13])
    device = np.random.choice(["Mobile", "Desktop"], size=n, p=[0.72, 0.28])

    # Numéricas base
    tenure_months = np.clip(np.random.normal(loc=18, scale=10, size=n), 0, 60).round().astype(int)

    # Visitas (sesiones/mes) dependen de canal + membresía
    base_visits = np.random.gamma(shape=3.0, scale=6.0, size=n)  # positivo sesgado
    visits = base_visits + (membership == "Sí") * np.random.normal(6, 2, size=n) + (channel == "Ads") * np.random.normal(3, 2, size=n)
    visits = np.clip(visits, 1, None).round().astype(int)

    # Tasa de conversión aproximada según canal (para generar compras)
    conv = np.where(channel == "Email", 0.10, np.where(channel == "Organic", 0.07, np.where(channel == "Referral", 0.08, 0.06)))
    conv += (membership == "Sí") * 0.02
    conv = np.clip(conv, 0.02, 0.18)

    # Compras en 90 días (discreta)
    purchases_90d = np.random.binomial(n=np.clip(visits // 2, 1, 60), p=conv).astype(int)

    # Ticket promedio (AOV) depende categoría + membresía + región levemente
    cat_aov = {
        "Tecnología": 55000,
        "Hogar": 32000,
        "Moda": 28000,
        "Deportes": 36000,
        "Belleza": 24000
    }
    aov = np.array([cat_aov[c] for c in category_pref], dtype=float)
    aov *= np.where(membership == "Sí", 1.10, 1.00)
    aov *= np.where(region == "RM", 1.03, 1.00)
    aov *= np.random.normal(1.0, 0.18, size=n)
    aov = np.clip(aov, 6000, None)

    # Monto total 90 días (target típico para estrategia comercial)
    total_spend_90d = purchases_90d * aov
    # agregar ruido y comportamiento heavy-spenders
    heavy = np.random.binomial(1, 0.04, size=n)
    total_spend_90d = total_spend_90d * (1 + heavy * np.random.uniform(0.8, 2.5, size=n))
    total_spend_90d = np.clip(total_spend_90d + np.random.normal(0, 8000, size=n), 0, None)

    # Devoluciones: aumenta con compras y algunas categorías (moda)
    base_return_prob = 0.04 + 0.01 * (category_pref == "Moda") + 0.005 * (category_pref == "Tecnología")
    return_count_90d = np.random.binomial(n=np.clip(purchases_90d, 0, 50), p=np.clip(base_return_prob, 0.01, 0.25)).astype(int)

    # Reseña promedio (1 a 5): mejora con membresía + baja con devoluciones
    rating = 4.1 + (membership == "Sí") * 0.15 - return_count_90d * 0.12 + np.random.normal(0, 0.35, size=n)
    rating = np.clip(rating, 1.0, 5.0).round(1)

    # Días desde última compra: si compra mucho, debería ser menor
    days_since_last_purchase = np.where(purchases_90d > 0,
                                        np.clip(np.random.exponential(scale=35, size=n) - purchases_90d * 2, 0, 180),
                                        np.clip(np.random.exponential(scale=80, size=n) + 30, 0, 365))
    days_since_last_purchase = np.round(days_since_last_purchase).astype(int)

    # Variable objetivo de marketing: probabilidad de recompra 30 días (binaria)
    # Depende de visitas, compras, rating y días desde última compra
    logit = -2.2 + 0.03 * visits + 0.18 * purchases_90d + 0.45 * (rating - 4.0) - 0.015 * days_since_last_purchase + 0.35 * (membership == "Sí")
    p_repurchase = 1 / (1 + np.exp(-logit))
    repurchase_30d = np.random.binomial(1, np.clip(p_repurchase, 0.02, 0.98), size=n)

    df = pd.DataFrame({
        "customer_id": customer_id,
        "region": region,
        "channel": channel,
        "membership": membership,
        "category_pref": category_pref,
        "device": device,
        "tenure_months": tenure_months,
        "visits_month": visits,
        "purchases_90d": purchases_90d,
        "aov_clp": np.round(aov, 0).astype(int),
        "total_spend_90d_clp": np.round(total_spend_90d, 0).astype(int),
        "return_count_90d": return_count_90d,
        "rating_avg": rating,
        "days_since_last_purchase": days_since_last_purchase,
        "repurchase_30d": repurchase_30d
    })

    # Introducir valores faltantes (IDA)
    # rating faltante para algunos sin reseñas
    mask_no_review = (df["purchases_90d"] == 0) & (np.random.rand(n) < 0.35)
    df.loc[mask_no_review, "rating_avg"] = np.nan

    # aov faltante si no hubo compras (tiene sentido)
    df.loc[df["purchases_90d"] == 0, "aov_clp"] = np.nan

    # inconsistencia: algunos total_spend_90d negativos por error de carga (simulado)
    bad_idx = np.random.choice(df.index, size=int(n * 0.01), replace=False)
    df.loc[bad_idx, "total_spend_90d_clp"] = df.loc[bad_idx, "total_spend_90d_clp"] * -1

    return df


def save_dataset(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


def basic_ida(df: pd.DataFrame) -> None:
    print("\n==================== IDA (Análisis inicial) ====================")
    print("Dimensiones:", df.shape)
    print("\nTipos de datos:\n", df.dtypes)

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    print("\nVariables numéricas:", num_cols)
    print("Variables categóricas:", cat_cols)

    print("\nValores faltantes por columna:\n", df.isna().sum().sort_values(ascending=False))

    # Inconsistencias simples
    neg_spend = (df["total_spend_90d_clp"] < 0).sum()
    print(f"\nInconsistencia: total_spend_90d_clp negativo -> {neg_spend} filas (simulado)")

    # Recomendación de limpieza
    print("\nPropuesta de limpieza:")
    print("- total_spend_90d_clp negativos: convertir a NaN o a valor absoluto según regla de negocio (aquí: NaN).")
    print("- aov_clp NaN: esperado cuando purchases_90d == 0 (mantener).")
    print("- rating_avg NaN: esperado para clientes sin reseñas (imputar opcionalmente o analizar como grupo 'sin reseña').")


# Cargar/generar dataset
csv_path = os.path.join(DATA_DIR, "comercioya_clientes.csv")
df = generate_dataset(n=2500)
save_dataset(df, csv_path)

basic_ida(df)

# Limpieza mínima para EDA/Modelos
df_clean = df.copy()
df_clean.loc[df_clean["total_spend_90d_clp"] < 0, "total_spend_90d_clp"] = np.nan

# ------------------------------------------------------------
# LECCIÓN 2: Estadística descriptiva + histogramas + boxplots + outliers
# ------------------------------------------------------------

def descriptive_stats(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    desc = df[cols].describe().T
    desc["var"] = df[cols].var(numeric_only=True)
    desc["mode"] = df[cols].mode(numeric_only=True).iloc[0]
    return desc

num_cols = ["tenure_months", "visits_month", "purchases_90d", "aov_clp",
            "total_spend_90d_clp", "return_count_90d", "rating_avg",
            "days_since_last_purchase"]

stats_table = descriptive_stats(df_clean, num_cols)
print("\n==================== Estadística descriptiva ====================")
print(stats_table)

# Cuartiles y percentiles
print("\n==================== Cuartiles y percentiles (ejemplo) ====================")
for col in ["total_spend_90d_clp", "visits_month", "rating_avg"]:
    q = df_clean[col].quantile([0.25, 0.5, 0.75, 0.9, 0.95])
    print(f"\n{col}:\n{q}")

# Histogramas
plt.figure()
df_clean["total_spend_90d_clp"].dropna().plot(kind="hist", bins=40)
plt.title("Histograma: Gasto total 90 días (CLP)")
plt.xlabel("CLP")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "hist_total_spend_90d.png"), dpi=200)
plt.close()

# Boxplot + outliers (IQR)
plt.figure()
sns.boxplot(x=df_clean["total_spend_90d_clp"])
plt.title("Boxplot: Gasto total 90 días (outliers)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "box_total_spend_90d.png"), dpi=200)
plt.close()

def iqr_outliers(series: pd.Series) -> pd.Series:
    s = series.dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return (series < low) | (series > high)

out_mask = iqr_outliers(df_clean["total_spend_90d_clp"])
print("\n==================== Outliers (IQR) ====================")
print("Outliers en total_spend_90d_clp:", out_mask.sum())

# ------------------------------------------------------------
# LECCIÓN 3: Correlación + scatterplots + heatmap + correlación espuria
# ------------------------------------------------------------

# Para correlación necesitamos solo numéricas (y sin NaNs en columnas clave)
corr_df = df_clean[num_cols].copy()

corr = corr_df.corr(method="pearson", numeric_only=True)
print("\n==================== Matriz de correlación (Pearson) ====================")
print(corr)

plt.figure(figsize=(10, 7))
sns.heatmap(corr, annot=False)
plt.title("Heatmap de correlación (Pearson)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "heatmap_correlaciones.png"), dpi=200)
plt.close()

# Scatterplots relevantes
plt.figure()
sns.scatterplot(data=df_clean, x="visits_month", y="purchases_90d", hue="membership", alpha=0.6)
plt.title("Visitas vs Compras (90d) segmentado por membresía")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "scatter_visits_purchases.png"), dpi=200)
plt.close()

plt.figure()
sns.scatterplot(data=df_clean, x="purchases_90d", y="total_spend_90d_clp", hue="category_pref", alpha=0.6)
plt.title("Compras (90d) vs Gasto total (90d)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "scatter_purchases_spend.png"), dpi=200)
plt.close()

# Ejemplo de correlación espuria:
# days_since_last_purchase y total_spend_90d pueden correlacionar porque ambos dependen de purchases_90d.
# Lo documentamos como posible variable confusora.
print("\nNota correlación espuria (para tu informe):")
print("- days_since_last_purchase puede correlacionar con total_spend_90d porque ambas variables están influenciadas por purchases_90d (confusor).")

# ------------------------------------------------------------
# LECCIÓN 4: Regresión lineal (simple y múltiple) + métricas + significancia
# ------------------------------------------------------------

# Objetivo: explicar total_spend_90d_clp
model_df = df_clean[["total_spend_90d_clp", "purchases_90d", "visits_month", "return_count_90d",
                     "days_since_last_purchase", "tenure_months"]].dropna()

# Train/test
X = model_df.drop(columns=["total_spend_90d_clp"])
y = model_df["total_spend_90d_clp"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_SEED)

# Statsmodels requiere constante
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

ols = sm.OLS(y_train, X_train_sm).fit()
print("\n==================== Regresión múltiple (OLS) ====================")
print(ols.summary())

# Predicción y métricas
y_pred = ols.predict(X_test_sm)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = ols.rsquared

print("\nMétricas del modelo:")
print(f"R2 (train): {r2:.3f}")
print(f"MSE (test): {mse:.2f}")
print(f"MAE (test): {mae:.2f}")

# Visualizar regresión simple (Seaborn): purchases_90d vs total_spend_90d_clp
plt.figure()
sns.regplot(data=model_df, x="purchases_90d", y="total_spend_90d_clp", scatter_kws={"alpha": 0.5})
plt.title("Regresión simple: Compras (90d) -> Gasto total (90d)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "reg_simple_purchases_spend.png"), dpi=200)
plt.close()

# ------------------------------------------------------------
# LECCIÓN 5: Visualizaciones Seaborn avanzadas
# ------------------------------------------------------------

# Pairplot (muestra subset para no saturar)
pair_cols = ["visits_month", "purchases_90d", "total_spend_90d_clp", "return_count_90d", "days_since_last_purchase"]
pair_df = df_clean[pair_cols + ["membership"]].dropna().sample(600, random_state=RANDOM_SEED)

g = sns.pairplot(pair_df, hue="membership", corner=True)
g.fig.suptitle("Pairplot: variables clave (subset)", y=1.02)
g.savefig(os.path.join(PLOTS_DIR, "pairplot_subset.png"), dpi=200)
plt.close("all")

# Violinplot: rating por canal
plt.figure(figsize=(9, 5))
sns.violinplot(data=df_clean, x="channel", y="rating_avg")
plt.title("Violinplot: Rating promedio por canal")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "violin_rating_channel.png"), dpi=200)
plt.close()

# Jointplot: visitas vs gasto
jp = sns.jointplot(data=df_clean.dropna(subset=["total_spend_90d_clp"]), x="visits_month", y="total_spend_90d_clp", kind="hex")
jp.fig.suptitle("Jointplot: Visitas vs Gasto (hexbin)", y=1.02)
jp.savefig(os.path.join(PLOTS_DIR, "joint_visits_spend.png"), dpi=200)
plt.close("all")

# FacetGrid: gasto por compras segmentado por canal
facet_df = df_clean.dropna(subset=["total_spend_90d_clp"]).copy()
fg = sns.FacetGrid(facet_df, col="channel", col_wrap=2, height=4, sharex=False, sharey=False)
fg.map_dataframe(sns.scatterplot, x="purchases_90d", y="total_spend_90d_clp", alpha=0.5)
fg.fig.suptitle("FacetGrid: Compras vs Gasto por canal", y=1.02)
fg.savefig(os.path.join(PLOTS_DIR, "facet_purchases_spend_channel.png"), dpi=200)
plt.close("all")

# ------------------------------------------------------------
# LECCIÓN 6: Matplotlib personalizado + subplots + exportación PDF
# ------------------------------------------------------------

# Figura final tipo “presentación”
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# 1) Hist gasto
axes[0, 0].hist(df_clean["total_spend_90d_clp"].dropna(), bins=35)
axes[0, 0].set_title("Distribución gasto 90d")
axes[0, 0].set_xlabel("CLP")
axes[0, 0].set_ylabel("Frecuencia")

# 2) Boxplot gasto por membresía
sns.boxplot(data=df_clean, x="membership", y="total_spend_90d_clp", ax=axes[0, 1])
axes[0, 1].set_title("Gasto 90d por membresía")
axes[0, 1].set_xlabel("Membresía")
axes[0, 1].set_ylabel("CLP")

# 3) Heatmap correlación (simple)
sns.heatmap(corr, ax=axes[1, 0])
axes[1, 0].set_title("Correlaciones")

# 4) Regresión simple
sns.regplot(data=model_df, x="purchases_90d", y="total_spend_90d_clp", ax=axes[1, 1], scatter_kws={"alpha": 0.4})
axes[1, 1].set_title("Regresión simple (compras -> gasto)")
axes[1, 1].set_xlabel("Compras 90d")
axes[1, 1].set_ylabel("CLP")

fig.suptitle("ComercioYA - Dashboard EDA (resumen)", fontsize=16)
fig.tight_layout()

png_path = os.path.join(PLOTS_DIR, "dashboard_eda.png")
pdf_path = os.path.join(PLOTS_DIR, "dashboard_eda.pdf")
fig.savefig(png_path, dpi=200)
fig.savefig(pdf_path)
plt.close(fig)

print("\n✅ Listo. Dataset guardado en:", csv_path)
print("✅ Gráficos exportados en:", PLOTS_DIR)

# ------------------------------------------------------------
# Insights recomendados (para tu informe)
# ------------------------------------------------------------

def generate_insights(df: pd.DataFrame) -> None:
    print("\n==================== INSIGHTS SUGERIDOS ====================")

    # Segmentos
    spend_by_member = df.groupby("membership")["total_spend_90d_clp"].median()
    print("\n1) Membresía:")
    print("- Mediana de gasto 90d por membresía:\n", spend_by_member)

    # Canal
    conv_proxy = df.groupby("channel")[["visits_month", "purchases_90d"]].mean()
    print("\n2) Canal:")
    print("- Promedio visitas y compras 90d por canal (proxy conversión):\n", conv_proxy)

    # Devoluciones
    ret_vs_rating = df[["return_count_90d", "rating_avg"]].corr(numeric_only=True).iloc[0, 1]
    print("\n3) Devoluciones vs rating:")
    print(f"- Correlación (Pearson) aprox: {ret_vs_rating:.3f} (esperable negativa)")

generate_insights(df_clean)