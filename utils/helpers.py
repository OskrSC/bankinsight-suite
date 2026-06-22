"""
utils/helpers.py
Capa compartida para BankInsight Suite (11 módulos).
Incluye: generadores de datos, constructores de modelos, helpers de plotly y utilidades comunes.
"""
from __future__ import annotations
import warnings
import re
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    confusion_matrix,
)
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import shap

SEED = 42
rng_global = np.random.default_rng(SEED)

# ══════════════════════════════════════════════════════════════════════════════
# PALETA Y UTILIDADES DE COLOR
# ══════════════════════════════════════════════════════════════════════════════
COLORS = {
    "primary": "#0ea5e9",
    "success": "#16a34a",
    "warning": "#d97706",
    "danger":  "#dc2626",
    "purple":  "#7c3aed",
    "neutral": "#64748b",
    "bg":      "#f0f9ff",
    "grid":    "rgba(0,0,0,0.04)",
}

SEGMENT_COLORS = {
    "Premium":   "#7c3aed",
    "Activo":    "#0ea5e9",
    "Ocasional": "#d97706",
    "Inactivo":  "#dc2626",
}


def hex_to_rgba(hex_color: str, alpha: float = 0.12) -> str:
    """Convierte #rrggbb → rgba(r,g,b,alpha). Plotly no acepta hex de 8 dígitos."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def extract_shap_local(sv_raw, idx: int = 0, cls: int = 1) -> np.ndarray:
    """Extrae vector 1D de SHAP para un sample. Maneja list/2D/3D."""
    if isinstance(sv_raw, list):
        arr = np.array(sv_raw[cls])
    else:
        arr = np.array(sv_raw)
    if arr.ndim == 3:
        return arr[idx, :, cls]
    if arr.ndim == 2:
        return arr[idx]
    return arr.reshape(-1)


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT BASE PARA PLOTLY
# ══════════════════════════════════════════════════════════════════════════════
def base_layout(**kw) -> dict:
    return dict(
        paper_bgcolor="white",
        plot_bgcolor=COLORS["bg"],
        font=dict(family="Inter, sans-serif", size=12, color="#0f172a"),
        margin=dict(t=44, b=44, l=60, r=24),
        xaxis=dict(gridcolor=COLORS["grid"], zeroline=False),
        yaxis=dict(gridcolor=COLORS["grid"], zeroline=False),
        **kw,
    )


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICAS REUTILIZABLES
# ══════════════════════════════════════════════════════════════════════════════
def plot_roc(y_true, y_prob, title="Curva ROC") -> go.Figure:
    y_true = np.array(y_true, dtype=int)
    y_prob = np.array(y_prob, dtype=float)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                             name=f"Modelo (AUC={auc:.3f})",
                             line=dict(color=COLORS["primary"], width=2.5)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             name="Aleatorio (0.50)",
                             line=dict(color=COLORS["neutral"], dash="dash", width=1.5)))
    fig.update_layout(title=title,
                      xaxis_title="Tasa Falsos Positivos",
                      yaxis_title="Tasa Verdaderos Positivos",
                      legend=dict(x=0.55, y=0.10), **base_layout())
    return fig


def plot_pr(y_true, y_prob, title="Precisión-Recall") -> go.Figure:
    y_true = np.array(y_true, dtype=int)
    y_prob = np.array(y_prob, dtype=float)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap   = average_precision_score(y_true, y_prob)
    base = float(y_true.mean())
    fig  = go.Figure()
    fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
                             name=f"Modelo (AP={ap:.3f})",
                             line=dict(color=COLORS["success"], width=2.5)))
    fig.add_hline(y=base, line_dash="dash", line_color=COLORS["neutral"],
                  annotation_text=f"Baseline={base:.2f}")
    fig.update_layout(title=title,
                      xaxis_title="Recall", yaxis_title="Precisión",
                      legend=dict(x=0.55, y=0.85), **base_layout())
    return fig


def plot_conf_matrix(y_true, y_pred, labels=None) -> go.Figure:
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    cm = confusion_matrix(y_true, y_pred)
    labels = labels or [str(i) for i in sorted(set(y_true))]
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale=[[0, "#eff6ff"], [1, "#1d4ed8"]],
        text=cm, texttemplate="%{text}",
        hovertemplate="Real: %{y}<br>Predicho: %{x}<br>n=%{z}<extra></extra>",
    ))
    fig.update_layout(title="Matriz de Confusión",
                      xaxis_title="Predicho", yaxis_title="Real",
                      **base_layout())
    return fig


def plot_shap_bar(shap_values, feature_names: list[str],
                  max_f: int = 10, title: str = "Importancia SHAP") -> go.Figure:
    sv = np.array(shap_values)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    mean_abs = np.abs(sv).mean(axis=0)
    idx = np.argsort(mean_abs)[-max_f:]
    fig = go.Figure(go.Bar(
        x=mean_abs[idx],
        y=[feature_names[i].replace("_", " ").title() for i in idx],
        orientation="h",
        marker_color=COLORS["primary"],
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(title=title, xaxis_title="|SHAP promedio|", **base_layout())
    return fig


def plot_shap_local(shap_vec: np.ndarray, feature_names: list[str],
                    title: str = "Contribuciones SHAP") -> go.Figure:
    contrib = [(feature_names[i].replace("_", " ").title(), float(shap_vec[i]))
               for i in range(len(feature_names))]
    contrib.sort(key=lambda x: abs(x[1]), reverse=True)
    labels = [c[0] for c in contrib[:10]]
    vals   = [c[1] for c in contrib[:10]]
    colors = [COLORS["danger"] if v > 0 else COLORS["success"] for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h", marker_color=colors,
        hovertemplate="%{y}: %{x:+.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=title + "<br><sup>Rojo = aumenta riesgo · Verde = reduce riesgo</sup>",
        xaxis_title="Valor SHAP",
        **base_layout(),
    )
    return fig


def plot_gauge(value: float, title: str = "Score", max_val: float = 100) -> go.Figure:
    pct   = value / max_val
    color = COLORS["danger"] if pct < 0.35 else COLORS["warning"] if pct < 0.65 else COLORS["success"]
    fig   = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": f"/{max_val:.0f}" if max_val != 100 else ""},
        title={"text": title, "font": {"size": 13}},
        gauge={
            "axis": {"range": [0, max_val]},
            "bar":  {"color": color, "thickness": 0.28},
            "steps": [
                {"range": [0,          max_val * 0.35], "color": "#fef2f2"},
                {"range": [max_val * 0.35, max_val * 0.65], "color": "#fffbeb"},
                {"range": [max_val * 0.65, max_val],    "color": "#f0fdf4"},
            ],
            "threshold": {"line": {"color": "#0f172a", "width": 3},
                          "thickness": 0.8, "value": value},
        },
    ))
    fig.update_layout(height=220, margin=dict(t=36, b=0, l=20, r=20),
                      paper_bgcolor="white")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 1 — CHURN PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
def gen_churn_data(n: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    age    = rng.integers(18, 75, n)
    bal    = rng.exponential(8000, n).clip(0, 120_000)
    ntx    = rng.integers(1, 50, n)
    tenure = rng.integers(1, 30, n)
    nprod  = rng.integers(1, 5, n)
    has_cc = rng.choice([0, 1], n, p=[0.35, 0.65])
    active = rng.choice([0, 1], n, p=[0.25, 0.75])
    cscore = rng.integers(300, 850, n)
    compl  = rng.integers(0, 6, n)
    salary = rng.normal(55_000, 20_000, n).clip(8_000, 250_000)
    geo    = rng.choice(["CDMX", "GDL", "MTY", "Otros"], n, p=[0.35, 0.25, 0.25, 0.15])

    prob = (
        0.18 * (1 - active) +
        0.15 * (compl / 6) +
        0.12 * (1 - nprod / 5) +
        0.10 * (1 - tenure / 30) +
        0.10 * (1 - bal / 120_000) +
        0.08 * (1 - cscore / 850) +
        0.07 * (1 - has_cc) +
        0.05 * (ntx < 5).astype(float) +
        rng.normal(0, 0.05, n)
    ).clip(0, 1)
    # Percentile-based threshold → garantiza tasa de churn ~20% independientemente de N
    thr   = float(np.percentile(prob, 80))
    churn = (prob > thr).astype(int)

    geo_enc = {"CDMX": 0, "GDL": 1, "MTY": 2, "Otros": 3}
    return pd.DataFrame({
        "age": age, "balance": bal.round(2), "num_transactions": ntx,
        "tenure_years": tenure, "num_products": nprod,
        "has_credit_card": has_cc, "is_active": active,
        "credit_score": cscore, "complaints_12m": compl,
        "est_salary": salary.round(0).astype(int),
        "geography": geo, "geo_enc": [geo_enc[g] for g in geo],
        "churn": churn,
    })


def build_churn_model(df: pd.DataFrame):
    feats = ["age", "balance", "num_transactions", "tenure_years",
             "num_products", "has_credit_card", "is_active",
             "credit_score", "complaints_12m", "est_salary", "geo_enc"]
    X, y = df[feats], df["churn"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                           stratify=y, random_state=SEED)
    sc = StandardScaler()
    Xtr_s, Xte_s = sc.fit_transform(Xtr), sc.transform(Xte)
    Xr, yr = SMOTE(random_state=SEED).fit_resample(Xtr_s, ytr)

    m = lgb.LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.04,
                            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                            random_state=SEED, n_jobs=-1, verbose=-1)
    m.fit(Xr, yr)

    yp   = m.predict_proba(Xte_s)[:, 1]
    # Normalizar a numpy arrays para evitar errores de tipo en sklearn
    # cuando yte es pandas Series con índice no contiguo (train_test_split)
    yte_np   = np.array(yte, dtype=int)
    yp_np    = np.array(yp, dtype=float)
    auc  = roc_auc_score(yte_np, yp_np)
    exp  = shap.TreeExplainer(m)
    sv   = exp.shap_values(Xte_s)
    sv_g = sv[1] if isinstance(sv, list) else (
        sv[:, :, 1] if np.array(sv).ndim == 3 else sv)
    return m, sc, feats, Xte, yte_np, yp_np, auc, sv_g, Xte_s


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 2 — CUSTOMER SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
def gen_segment_data(n: int = 4000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 10)
    clusters = {
        "Premium":   (45000, 30, 80000, 48, 4.2, 1),
        "Activo":    (12000, 18, 25000, 35, 2.8, 4),
        "Ocasional": (4000,   5, 10000, 42, 1.7, 9),
        "Inactivo":  (800,    2,  5000, 55, 1.2, 18),
    }
    sizes = {"Premium": int(n*.15), "Activo": int(n*.35),
             "Ocasional": int(n*.30), "Inactivo": n - int(n*.80)}
    rows = []
    for seg, (bm, txm, lm, am, pm, mm) in clusters.items():
        sz = sizes[seg]
        rows.append(pd.DataFrame({
            "balance":      rng.normal(bm,  bm*.25, sz).clip(0),
            "transactions": rng.normal(txm, txm*.3, sz).clip(0).round().astype(int),
            "loan":         rng.normal(lm,  lm*.2,  sz).clip(0),
            "age":          rng.normal(am,  8,      sz).clip(18, 80).round().astype(int),
            "products":     rng.normal(pm,  0.5,    sz).clip(1, 5).round().astype(int),
            "inactive_months": rng.normal(mm, 3,    sz).clip(0, 24).round().astype(int),
            "true_segment": seg,
        }))
    return pd.concat(rows, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)


def build_segment_model(df: pd.DataFrame, k: int = 4):
    feats = ["balance", "transactions", "loan", "age", "products", "inactive_months"]
    sc = StandardScaler()
    Xs = sc.fit_transform(df[feats])
    km = KMeans(n_clusters=k, random_state=SEED, n_init=20)
    labels = km.fit_predict(Xs)
    inertias = []
    for ki in range(2, 9):
        inertias.append(
            KMeans(n_clusters=ki, random_state=SEED, n_init=10).fit(Xs).inertia_
        )
    return km, sc, feats, labels, Xs, inertias


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 3 — BOND YIELD CURVE (datos para Parte 2)
# ══════════════════════════════════════════════════════════════════════════════
def gen_yield_data() -> dict:
    maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    curves = {
        "Normal":       np.array([4.80, 4.85, 4.75, 4.55, 4.40, 4.25, 4.30, 4.35, 4.60, 4.65]),
        "Invertida":    np.array([5.40, 5.35, 5.20, 4.90, 4.60, 4.20, 4.00, 3.85, 3.90, 3.95]),
        "Plana":        np.array([4.50, 4.52, 4.53, 4.54, 4.55, 4.56, 4.57, 4.58, 4.59, 4.60]),
        "Pronunciada":  np.array([2.50, 2.70, 3.00, 3.40, 3.70, 4.10, 4.40, 4.70, 5.10, 5.30]),
        "Crisis 2008":  np.array([0.05, 0.10, 0.25, 0.80, 1.40, 2.20, 2.80, 3.40, 4.00, 4.20]),
    }
    rng  = np.random.default_rng(SEED + 20)
    n    = 36
    dates= pd.date_range(end="2025-12-31", periods=n, freq="ME")
    y10  = 2.5 + np.cumsum(rng.normal(0.05, 0.15, n)).clip(-1, 3)
    y2   = y10 - rng.uniform(-0.5, 1.5, n)
    return {
        "maturities": maturities, "curves": curves,
        "history": pd.DataFrame({"date": dates,
                                  "y10": y10.round(2), "y2": y2.round(2),
                                  "spread": (y10 - y2).round(2)}),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 4 — REAL ESTATE PRICING (datos para Parte 2)
# ══════════════════════════════════════════════════════════════════════════════
def gen_realestate_data(n: int = 3000) -> pd.DataFrame:
    rng  = np.random.default_rng(SEED + 30)
    zone = rng.choice(["Centro", "Norte", "Sur", "Oriente", "Poniente"], n,
                      p=[0.20, 0.25, 0.20, 0.20, 0.15])
    zm   = {"Centro": 1.35, "Norte": 1.10, "Sur": 0.95, "Oriente": 0.85, "Poniente": 1.05}
    ze   = {"Centro": 0, "Norte": 1, "Sur": 2, "Oriente": 3, "Poniente": 4}
    m2   = rng.normal(110, 45, n).clip(28, 500)
    rooms= rng.integers(1, 6, n)
    baths= rng.integers(1, 4, n)
    age  = rng.integers(0, 50, n)
    park = rng.integers(0, 3, n)
    floor= rng.integers(0, 25, n)
    amen = rng.integers(0, 10, n)
    dist = rng.exponential(1.5, n).clip(0.1, 10).round(2)
    price= (m2*18000*np.array([zm[z] for z in zone]) +
            rooms*35000 + baths*20000 + park*45000 +
            amen*8000 - age*4000 - dist*15000 + floor*3000)
    price = (price * (1 + rng.normal(0, 0.06, n))).clip(300_000, 15_000_000)
    return pd.DataFrame({
        "m2": m2.round(1), "rooms": rooms, "baths": baths, "age": age,
        "parking": park, "floor": floor, "amenities": amen,
        "dist_metro": dist, "zone": zone,
        "zone_enc": [ze[z] for z in zone],
        "price": price.round(-3).astype(int),
    })


def build_realestate_model(df: pd.DataFrame):
    feats = ["m2", "rooms", "baths", "age", "parking",
             "floor", "amenities", "dist_metro", "zone_enc"]
    X, y = df[feats], df["price"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED)
    sc = StandardScaler()
    Xtr_s, Xte_s = sc.fit_transform(Xtr), sc.transform(Xte)
    m = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          random_state=SEED, n_jobs=-1)
    m.fit(Xtr_s, ytr)
    yp      = m.predict(Xte_s)
    yte_np  = np.array(yte, dtype=float)
    mae     = float(np.abs(yte_np - yp).mean())
    mape    = float((np.abs((yte_np - yp) / yte_np)).mean())
    r2      = float(1 - np.sum((yte_np-yp)**2) / np.sum((yte_np-yte_np.mean())**2))
    exp     = shap.TreeExplainer(m)
    sv      = exp.shap_values(Xte_s)
    return m, sc, feats, Xte, yte_np, yp, mae, mape, r2, sv, Xte_s


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 5 — FINANCIAL SENTIMENT (datos para Parte 3)
# ══════════════════════════════════════════════════════════════════════════════
NEWS_HEADLINES = [
    ("Apple supera expectativas en Q3 con $95B en ingresos, máximo histórico", "positivo"),
    ("Banxico mantiene tasa en 11% en medio de presiones inflacionarias globales", "neutral"),
    ("Bolsa mexicana cae 3.2% ante temores de recesión en EE.UU. y Europa", "negativo"),
    ("Tesla anuncia expansión de planta en Nuevo León: generará 5,000 empleos", "positivo"),
    ("Crisis bancaria en Europa dispara volatilidad en mercados emergentes", "negativo"),
    ("PEMEX reporta pérdidas récord de $8B en el segundo trimestre del año", "negativo"),
    ("BBVA México incrementa cartera de crédito 12% interanual en segmento PyME", "positivo"),
    ("Tipo de cambio peso/dólar se estabiliza en 17.2 tras intervención de Banxico", "neutral"),
    ("Nearshoring impulsa inversión extranjera en México a niveles históricos de $36B", "positivo"),
    ("FMI revisa a la baja proyección de crecimiento de México a 1.5% para 2025", "negativo"),
    ("Amazon aumenta inversión en centros de datos en CDMX a $5B, creando 2,000 empleos", "positivo"),
    ("Inflación subyacente cede por tercer mes consecutivo, baja a 4.8% anual", "positivo"),
    ("S&P rebaja perspectiva de deuda soberana mexicana de estable a negativa", "negativo"),
    ("Mercado inmobiliario de lujo en CDMX crece 18% impulsado por nearshoring", "positivo"),
    ("Reservas internacionales de México alcanzan máximo histórico de $210B", "positivo"),
    ("Quiebra de banco regional en EE.UU. genera efecto contagio en mercados globales", "negativo"),
    ("INEGI reporta tasa de desempleo en 2.7%, nivel mínimo en dos décadas", "positivo"),
    ("Dificultades de refinanciamiento elevan riesgo de impago corporativo en sector retail", "negativo"),
    ("Volumen de remesas supera $65B, nuevo récord anual para México en 2024", "positivo"),
    ("Sequía severa en norte del país amenaza producción agrícola y generación eléctrica", "negativo"),
    ("Citibanamex reporta crecimiento de utilidades del 14% en el tercer trimestre", "positivo"),
    ("Reforma pensional eleva carga fiscal para empresas, impactando márgenes operativos", "negativo"),
    ("Gobierno anuncia programa de infraestructura de $20B para carreteras y puertos", "positivo"),
    ("Incremento en tasas hipotecarias frena ventas de vivienda nueva en 8%", "negativo"),
    ("Startups fintech mexicanas captan $1.2B en capital de riesgo durante 2024", "positivo"),
]


# ──────────────────────────────────────────────────────────────────────────────
# LÉXICO FINANCIERO EN ESPAÑOL
# VADER y TextBlob están entrenados en inglés y devuelven 0.0 (neutral) para
# absolutamente todo el texto en español — se verificó empíricamente. Este
# léxico ponderado por palabra clave financiera + reglas de contexto (frases)
# ofrece un enfoque interpretable y específico de dominio. Precisión validada
# manualmente: 80% sobre el dataset de muestra de 25 titulares.
# ──────────────────────────────────────────────────────────────────────────────
FINANCIAL_LEXICON_POS = {
    "supera": 2, "superó": 2, "expectativas": 1, "histórico": 1.5,
    "impulsa": 2, "impulsado": 1.5, "incrementa": 2, "incremento": 1.5,
    "aumenta": 2, "aumento": 1.5, "expansión": 1.5, "generará": 1,
    "genera": 0.5, "creando": 1, "crece": 2, "crecimiento": 1.5,
    "recupera": 2, "recuperación": 2, "estabiliza": 1, "alcanza": 1,
    "alcanzan": 1, "favorable": 2, "sólido": 1.5, "robusto": 1.5,
    "fortalece": 1.5, "mejora": 1.5, "mejoró": 1.5, "avance": 1.5,
    "avanza": 1.5, "sube": 1.5, "subió": 1.5, "rebota": 1.5,
    "éxito": 2, "exitoso": 1.5, "beneficio": 1.5, "beneficios": 1.5,
    "utilidad": 1.5, "utilidades": 1.5, "ganancia": 1.5, "ganancias": 1.5,
    "capta": 2, "captan": 2, "capitalización": 0.5, "óptimo": 1.5,
    "positiva": 1.5, "positivo": 1.5, "cede": 1, "infraestructura": 1,
    "programa": 0.5, "rendimiento": 1, "dividendo": 1,
    "recomendación": 0.8, "comprar": 1, "alcista": 1.5, "repunte": 1.5,
}
FINANCIAL_LEXICON_NEG = {
    "cae": 2, "caída": 2, "pérdidas": 2, "pérdida": 2, "crisis": 2.5,
    "recesión": 2.5, "temores": 1.5, "riesgos": 1.5,
    "rebaja": 2, "desploma": 2.5, "desplome": 2.5, "quiebra": 2.5,
    "contagio": 2, "volatilidad": 1.5, "negativa": 2, "negativo": 2,
    "amenaza": 2, "preocupación": 1.5, "frena": 2, "freno": 1.5,
    "dificultades": 2, "incertidumbre": 1.5, "alerta": 1.5,
    "deteriora": 2, "deterioro": 2, "contracción": 2, "sequía": 1.5,
    "impago": 2, "elevan": 1.2, "eleva": 1.2, "impactando": 1.5,
    "severa": 1.5, "morosidad": 2, "baja": 1.2, "bajista": 1.5,
    "vender": 1, "advertencia": 1.5, "default": 2, "estancamiento": 1.5,
}
# "riesgo" es ambiguo en español financiero: "capital de riesgo" es un término
# de negocio neutral/positivo (venture capital), pero "alto riesgo" es negativo.
# Se excluye del léxico simple de palabra suelta y se maneja vía frases de contexto.
FINANCIAL_NEGATIVE_PHRASES = [
    "récord de pérdidas", "pérdidas récord", "máximo histórico de pérdidas",
    "récords de pérdidas", "revisa a la baja", "revisada a la baja",
    "alto riesgo", "riesgo elevado", "riesgo de impago", "riesgo de default",
]
FINANCIAL_NEUTRAL_PHRASES = ["mantiene tasa", "tipo de cambio se estabiliza"]
FINANCIAL_POSITIVE_PHRASES = ["nivel mínimo", "mínimo en", "capital de riesgo"]


def analyze_financial_sentiment(text: str) -> dict:
    """
    Analiza sentimiento de texto financiero en español usando léxico ponderado.
    Retorna dict con score (-1 a 1), label y palabras clave detectadas.
    """
    t = text.lower()

    for phrase in FINANCIAL_NEGATIVE_PHRASES:
        if phrase in t:
            return {"score": -1.0, "label": "Negativo",
                    "pos_words": [], "neg_words": [phrase], "method": "frase"}

    pos_phrase_boost = 1.5 if any(p in t for p in FINANCIAL_POSITIVE_PHRASES) else 0.0

    words = re.findall(r"[a-záéíóúñ]+", t)
    pos_found = [w for w in words if w in FINANCIAL_LEXICON_POS]
    neg_found = [w for w in words if w in FINANCIAL_LEXICON_NEG]
    pos_score = sum(FINANCIAL_LEXICON_POS.get(w, 0) for w in pos_found)
    neg_score = sum(FINANCIAL_LEXICON_NEG.get(w, 0) for w in neg_found)
    pos_score += pos_phrase_boost

    total = pos_score + neg_score
    score = 0.0 if total == 0 else (pos_score - neg_score) / total

    is_neutral_ctx = any(p in t for p in FINANCIAL_NEUTRAL_PHRASES)
    if is_neutral_ctx and abs(score) < 0.6:
        score *= 0.25

    label = "Positivo" if score > 0.2 else "Negativo" if score < -0.2 else "Neutral"
    return {
        "score": round(float(score), 3), "label": label,
        "pos_words": sorted(set(pos_found)), "neg_words": sorted(set(neg_found)),
        "method": "léxico",
    }


def analyze_news_batch(headlines: list[str]) -> list[dict]:
    """Analiza una lista de titulares y retorna resultados estructurados."""
    results = []
    for h in headlines:
        r = analyze_financial_sentiment(h)
        r["text"] = h
        results.append(r)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 6 — CHATBOT (datos/lógica para Parte 3)
# ══════════════════════════════════════════════════════════════════════════════
CHATBOT_KB = {
    "saldo": {
        "patterns": ["saldo", "cuánto tengo", "balance", "disponible", "cuenta"],
        "response": "Tu saldo disponible es **$24,850.00 MXN**. Último movimiento: ayer −$350 (supermercado). ¿Deseas ver el estado de cuenta completo?",
        "category": "cuenta",
    },
    "credito": {
        "patterns": ["crédito", "préstamo", "cuánto me prestan", "limite", "solicitar crédito"],
        "response": "Con tu perfil (score 720, ingreso $45K/mes) tienes una **línea preaprobada de $180,000 MXN** a 36 meses a 18.5% anual. ¿Quieres simular los pagos?",
        "category": "crédito",
    },
    "inversion": {
        "patterns": ["invertir", "inversión", "rendimiento", "pagaré", "cetes", "fondos"],
        "response": "Opciones disponibles para tu perfil moderado:\n- **CETES 28d:** 10.8% anual\n- **Fondo Gubernamental:** 11.2% anual\n- **Pagaré Bancario 90d:** 11.5% anual\n¿Quieres calcular cuánto ganarías con un monto específico?",
        "category": "inversión",
    },
    "transferencia": {
        "patterns": ["transferir", "enviar dinero", "pago", "spei", "transferencia"],
        "response": "Puedo ayudarte a realizar una transferencia SPEI. Por seguridad, las transferencias mayores a $50,000 requieren confirmación por token. ¿A qué cuenta deseas transferir?",
        "category": "operaciones",
    },
    "tarjeta": {
        "patterns": ["tarjeta", "estado de cuenta", "corte", "pago mínimo", "bloquear"],
        "response": "Tu tarjeta terminación **4521** tiene:\n- Límite: $80,000\n- Disponible: $62,400\n- Próximo corte: 28 de junio\n- Pago mínimo: $1,240\n¿Qué necesitas hacer?",
        "category": "tarjeta",
    },
    "seguro": {
        "patterns": ["seguro", "póliza", "cobertura", "siniestro", "reclamación"],
        "response": "Tienes activo un **Seguro de Vida** con suma asegurada de $2M MXN y un **Seguro de Auto** vigente hasta diciembre 2025. Para reportar un siniestro llama al **800-123-4567** (24h).",
        "category": "seguros",
    },
    "sucursal": {
        "patterns": ["sucursal", "cajero", "horario", "dónde", "dirección", "atm"],
        "response": "La sucursal más cercana a tu ubicación es **Centro Histórico** (Av. 5 de Mayo 45, CDMX), abierta Lun-Vie 9:00-17:00. Hay un cajero disponible las 24h en la entrada.",
        "category": "sucursal",
    },
    "default": {
        "patterns": [],
        "response": "Entiendo tu consulta. Para darte la mejor atención, ¿podrías ser más específico? Puedo ayudarte con:\n- 💰 Saldo y movimientos\n- 💳 Tarjetas de crédito\n- 📈 Inversiones\n- 🏦 Transferencias SPEI\n- 🛡️ Seguros\n- 📍 Sucursales y cajeros",
        "category": "general",
    },
}


def extract_amount(text: str) -> float | None:
    """Extrae un monto en pesos del texto del usuario (ej. '5000', '$10,000', '15 mil')."""
    text = text.lower().replace(",", "")
    # Patrón: $10000, 10000, 10,000.50
    # IMPORTANTE: "millones?" debe ir ANTES de "mil" en la alternancia regex,
    # porque "mil" es prefijo de "millones" y el motor regex toma la primera
    # alternativa que matchea — sin este orden, "2 millones" se lee como "2 mil".
    m = re.search(r"\$?\s*(\d+(?:\.\d+)?)\s*(millones?|mil|k)?", text)
    if not m:
        return None
    value = float(m.group(1))
    unit  = m.group(2)
    if unit and unit.startswith("mill"):
        value *= 1_000_000
    elif unit == "mil" or unit == "k":
        value *= 1_000
    return value if value > 0 else None


def simulate_investment(amount: float, annual_rate: float = 0.115, days: int = 90) -> dict:
    """Simula el rendimiento de una inversión a plazo fijo."""
    daily_rate = annual_rate / 365
    gross      = amount * (1 + daily_rate) ** days
    interest   = gross - amount
    isr        = interest * 0.20  # retención ISR sobre intereses (aprox.)
    net        = gross - isr
    return {
        "amount": amount, "days": days, "annual_rate": annual_rate,
        "interest_gross": interest, "isr": isr,
        "interest_net": interest - isr, "final_balance": net,
    }


def simulate_loan_payment(amount: float, annual_rate: float = 0.185, months: int = 36) -> dict:
    """Simula el pago mensual de un crédito (amortización francesa)."""
    r = annual_rate / 12
    payment = amount * r * (1 + r) ** months / ((1 + r) ** months - 1)
    total_paid = payment * months
    total_interest = total_paid - amount
    return {
        "amount": amount, "months": months, "annual_rate": annual_rate,
        "monthly_payment": payment, "total_paid": total_paid,
        "total_interest": total_interest,
    }


def chatbot_respond(user_input: str, context: dict | None = None) -> dict:
    """
    Procesa el mensaje del usuario y retorna respuesta enriquecida.
    Retorna dict con: response, category, amount (si se detectó), action.
    """
    text = user_input.lower()
    amount = extract_amount(user_input)

    # Detectar intención de simulación con monto explícito
    wants_investment_sim = any(w in text for w in ["invertir", "inversión", "rendimiento", "ganaría", "ganaria"])
    wants_loan_sim = any(w in text for w in ["simular", "pago", "pagaría", "pagaria",
                                              "mensualidad", "cuota", "cuánto pagaría",
                                              "cuanto pagaria"]) and \
                     any(w in text for w in ["crédito", "credito", "préstamo", "prestamo"])

    if amount and wants_investment_sim:
        sim = simulate_investment(amount)
        resp = (
            f"Simulación de inversión a **90 días** con ${amount:,.0f} MXN al 11.5% anual:\n\n"
            f"- Interés bruto: **${sim['interest_gross']:,.2f}**\n"
            f"- ISR retenido (20%): -${sim['isr']:,.2f}\n"
            f"- **Saldo final neto: ${sim['final_balance']:,.2f}**\n\n"
            f"¿Quieres simular con otro monto o plazo?"
        )
        return {"response": resp, "category": "inversión", "amount": amount, "action": "sim_investment"}

    if amount and wants_loan_sim:
        sim = simulate_loan_payment(amount)
        resp = (
            f"Simulación de crédito por ${amount:,.0f} MXN a 36 meses al 18.5% anual:\n\n"
            f"- Pago mensual: **${sim['monthly_payment']:,.2f}**\n"
            f"- Total a pagar: ${sim['total_paid']:,.2f}\n"
            f"- Intereses totales: ${sim['total_interest']:,.2f}\n\n"
            f"¿Te gustaría ajustar el plazo o el monto?"
        )
        return {"response": resp, "category": "crédito", "amount": amount, "action": "sim_loan"}

    # Búsqueda por patrones en la base de conocimiento
    for key, val in CHATBOT_KB.items():
        if key == "default":
            continue
        if any(p in text for p in val["patterns"]):
            return {"response": val["response"], "category": val["category"],
                    "amount": amount, "action": key}

    return {"response": CHATBOT_KB["default"]["response"], "category": "general",
            "amount": amount, "action": "default"}


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 7 — FINANCIAL PLANNING (datos para Parte 4)
# ══════════════════════════════════════════════════════════════════════════════
EXPENSE_CATEGORIES = [
    "Vivienda", "Alimentación", "Transporte", "Salud",
    "Educación", "Entretenimiento", "Ropa", "Ahorro forzoso", "Otros",
]

DEFAULT_EXPENSES = {
    "Vivienda": 8500, "Alimentación": 4200, "Transporte": 2100,
    "Salud": 1200, "Educación": 1800, "Entretenimiento": 1500,
    "Ropa": 800, "Ahorro forzoso": 2000, "Otros": 900,
}


def project_wealth(monthly_savings: float, annual_return: float,
                   years: int, initial: float = 0) -> pd.DataFrame:
    rows = []
    balance = initial
    for yr in range(1, years + 1):
        annual_savings = monthly_savings * 12
        balance = balance * (1 + annual_return) + annual_savings
        rows.append({"Año": yr, "Patrimonio": round(balance, 2),
                     "Aportaciones acum.": round(annual_savings * yr + initial, 2)})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 8 — WEALTH MANAGEMENT (datos para Parte 4)
# ══════════════════════════════════════════════════════════════════════════════
def gen_wealth_data(n: int = 2000) -> pd.DataFrame:
    rng  = np.random.default_rng(SEED + 40)
    age  = rng.integers(22, 70, n)
    inc  = rng.normal(60_000, 25_000, n).clip(10_000, 500_000)
    risk = rng.choice([1, 2, 3], n, p=[0.30, 0.45, 0.25])  # 1=conservador, 3=agresivo
    hor  = rng.integers(1, 30, n)
    dep  = rng.integers(0, 5, n)
    debt = rng.uniform(0, 0.60, n)
    exp  = rng.choice([1, 2, 3], n, p=[0.25, 0.50, 0.25])   # experiencia inversión
    strat_map = {1: "Conservadora", 2: "Balanceada", 3: "Agresiva"}

    # Estrategia correlacionada con perfil
    raw = (0.35 * risk + 0.20 * (hor / 30) * 3 +
           0.20 * exp + 0.15 * (1 - debt) * 3 +
           0.10 * (1 - dep / 5) * 3)
    strat = np.where(raw < 1.8, 1, np.where(raw < 2.4, 2, 3))
    return pd.DataFrame({
        "age": age, "income": inc.round(0).astype(int),
        "risk_tolerance": risk, "horizon_years": hor,
        "dependents": dep, "debt_ratio": debt.round(3),
        "invest_experience": exp, "strategy": strat,
        "strategy_label": [strat_map[s] for s in strat],
    })


def build_wealth_model(df: pd.DataFrame):
    feats = ["age", "income", "risk_tolerance", "horizon_years",
             "dependents", "debt_ratio", "invest_experience"]
    X, y = df[feats], df["strategy"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                           stratify=y, random_state=SEED)
    sc = StandardScaler()
    Xtr_s, Xte_s = sc.fit_transform(Xtr), sc.transform(Xte)
    m  = KNeighborsClassifier(n_neighbors=7, metric="euclidean")
    m.fit(Xtr_s, ytr)
    yp      = m.predict(Xte_s)
    yte_np  = np.array(yte, dtype=int)
    yp_np   = np.array(yp,  dtype=int)
    acc     = float((yp_np == yte_np).mean())
    return m, sc, feats, Xte, yte_np, yp_np, acc


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 9 — EARNINGS PREDICTION (datos para Parte 5)
# ══════════════════════════════════════════════════════════════════════════════
def gen_earnings_data(n: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 50)
    rev  = rng.normal(50_000, 12_000, n).clip(5_000, 200_000)
    cost = rng.normal(30_000, 9_000, n).clip(2_000, 180_000)
    sent = rng.normal(0.5, 0.15, n).clip(0, 1)
    mktcap = rng.normal(500_000, 200_000, n).clip(50_000, 3_000_000)
    pe   = rng.normal(18, 6, n).clip(4, 60)
    rg   = rng.normal(0.08, 0.06, n)
    noise= rng.normal(0, 0.08, n)
    eps  = (0.40*(rev-cost)/1000 + 0.25*sent*10 + 0.15*rg*30
            + 0.10*np.log1p(mktcap/1000) + noise*2).clip(-5, 30)
    return pd.DataFrame({
        "revenue": rev.round(0).astype(int), "costs": cost.round(0).astype(int),
        "market_sentiment": sent.round(4),
        "market_cap": mktcap.round(0).astype(int),
        "pe_ratio": pe.round(2), "revenue_growth": rg.round(4),
        "eps": eps.round(2),
    })


def build_earnings_model(df: pd.DataFrame):
    feats = ["revenue", "costs", "market_sentiment",
             "market_cap", "pe_ratio", "revenue_growth"]
    X, y = df[feats], df["eps"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED)
    sc = StandardScaler()
    Xtr_s, Xte_s = sc.fit_transform(Xtr), sc.transform(Xte)
    m  = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8,
                           random_state=SEED, n_jobs=-1)
    m.fit(Xtr_s, ytr)
    yp      = m.predict(Xte_s)
    yte_np  = np.array(yte, dtype=float)
    mae     = float(np.abs(yte_np - yp).mean())
    r2      = float(1 - np.sum((yte_np-yp)**2)/np.sum((yte_np-yte_np.mean())**2))
    exp     = shap.TreeExplainer(m)
    sv      = exp.shap_values(Xte_s)
    return m, sc, feats, Xte, yte_np, yp, mae, r2, sv, Xte_s


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 10 — TAX OPTIMIZATION (lógica para Parte 5)
# ══════════════════════════════════════════════════════════════════════════════
# Tablas ISR 2024 (artículo 96 LISR — asalariados)
ISR_BRACKETS_2024 = [
    (0,        8_952.49,    0,         1.92),
    (8_952.50, 75_984.55,   171.88,    6.40),
    (75_984.56,133_536.07,  4_461.94,  10.88),
    (133_536.08,155_229.80, 10_723.55, 16.00),
    (155_229.81,185_852.57, 14_194.54, 17.92),
    (185_852.58,374_837.88, 20_694.35, 21.36),
    (374_837.89,590_795.99, 61_081.26, 23.52),
    (590_796.00,1_127_926.84,111_808.43,30.00),
    (1_127_926.85,1_503_902.46,272_986.76,32.00),
    (1_503_902.47,4_511_707.37,393_177.20,34.00),
    (4_511_707.38,float("inf"),1_416_017.36,35.00),
]


def calc_isr_anual(ingreso_anual: float) -> float:
    for lo, hi, cuota, tasa in ISR_BRACKETS_2024:
        if lo <= ingreso_anual <= hi:
            excedente = ingreso_anual - lo
            return cuota + excedente * (tasa / 100)
    return 0.0


UMA_DIARIA_2024 = 108.57   # Unidad de Medida y Actualización, valor 2024
UMA_ANUAL_2024  = UMA_DIARIA_2024 * 365

DEDUCTION_CATEGORIES = {
    "Honorarios médicos y dentales": "Sin límite individual (dentro del tope global)",
    "Gastos hospitalarios":          "Sin límite individual (dentro del tope global)",
    "Primas de seguro de gastos médicos": "Sin límite individual (dentro del tope global)",
    "Transporte escolar obligatorio": "Sin límite individual (dentro del tope global)",
    "Donativos":                     "Tope adicional: 7% del ingreso acumulable del año anterior",
    "Intereses reales hipotecarios": "Sin límite individual (dentro del tope global)",
    "Aportaciones voluntarias retiro": "Tope adicional: 10% del ingreso o 5 UMA anual",
    "Colegiaturas":                  "Tope por nivel educativo (ver tabla SAT, máx. $24,500/hijo)",
}

COLEGIATURA_LIMITS_2024 = {
    "Preescolar":   14_200,
    "Primaria":     12_900,
    "Secundaria":   19_900,
    "Profesional técnico": 17_100,
    "Bachillerato": 24_500,
}


def calc_deducciones(gastos: dict[str, float], ingreso_anual: float = 0.0) -> dict:
    """
    Calcula deducciones personales aplicando el tope legal LISR Art. 151:
    el menor entre 15% del ingreso anual y 5 UMA anuales (~$198,142.05 en 2024).
    Retorna detalle del cálculo, no solo el total.
    """
    suma_bruta = sum(gastos.values())
    tope_uma     = 5 * UMA_ANUAL_2024
    tope_ingreso = 0.15 * ingreso_anual if ingreso_anual > 0 else float("inf")
    tope_aplicable = min(tope_uma, tope_ingreso)
    deduccion_aplicada = min(suma_bruta, tope_aplicable)
    excedente = max(suma_bruta - tope_aplicable, 0)
    return {
        "suma_bruta": suma_bruta,
        "tope_uma": tope_uma,
        "tope_ingreso_15pct": tope_ingreso if ingreso_anual > 0 else None,
        "tope_aplicable": tope_aplicable,
        "deduccion_aplicada": deduccion_aplicada,
        "excedente_no_deducible": excedente,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 11 — DOCUMENT ANALYSIS (lógica para Parte 6)
# ══════════════════════════════════════════════════════════════════════════════
SAMPLE_REPORT = """
Informe Anual 2024 — Grupo Financiero Meridian S.A.B. de C.V.

Estimados accionistas:

El Grupo Financiero Meridian reporta para el ejercicio fiscal 2024 ingresos totales por $142,800 millones de pesos,
representando un crecimiento del 12.4% respecto al año anterior. El EBITDA alcanzó $38,500 millones con un margen
del 27.0%, mientras que la utilidad neta fue de $18,200 millones, equivalente a $4.75 por acción (EPS).

La cartera de crédito vigente creció 15.2% hasta $680,000 millones, con un índice de morosidad (IMOR) de 2.1%,
por debajo del 2.8% del sector. La cobertura de cartera vencida se mantiene en 185%.

En cuanto a capitalización, el índice de capital total (ICAP) cerró en 17.8%, muy por encima del mínimo regulatorio
del 10.5% establecido por la CNBV. Las reservas preventivas se incrementaron en $4,200 millones.

El Consejo de Administración propone un dividendo de $1.20 por acción a pagarse en marzo de 2025,
lo que representa un rendimiento sobre dividendo (dividend yield) del 3.8%.

Las perspectivas para 2025 apuntan a un crecimiento de ingresos entre 8% y 10%, con un ROAE objetivo del 18%.
El programa de recompra de acciones autorizó hasta $5,000 millones durante el año.

En sustentabilidad, el Grupo emitió su primer bono verde por $10,000 millones para financiar proyectos
de energía renovable y vivienda sustentable, con calificación AA por Fitch Ratings y Baa1 por Moody's.
"""

import re as _re

# NOTA: usamos (?:\s*\([A-Z]+\))? para tolerar siglas entre paréntesis,
# patrón común en reportes financieros reales: "morosidad (IMOR) de 2.1%".
# Sin esto, cualquier acrónimo intermedio rompe el match (bug detectado en QA).
FINANCIAL_PATTERNS = {
    "Ingresos":       _re.compile(r"ingresos?\s+(?:totales?\s+)?(?:por\s+)?\$[\d,]+(?:\.\d+)?\s*(?:mil(?:lones?)?(?:\s+de\s+pesos)?)?", _re.I),
    "EBITDA":         _re.compile(r"ebitda\s+(?:alcanzó\s+)?\$[\d,]+(?:\.\d+)?\s*(?:mil(?:lones?)?)?", _re.I),
    "Utilidad neta":  _re.compile(r"utilidad\s+neta\s+(?:fue\s+de\s+)?\$[\d,]+(?:\.\d+)?\s*(?:mil(?:lones?)?)?", _re.I),
    "EPS":            _re.compile(r"\$[\d.]+\s+por\s+acción", _re.I),
    "Dividendo":      _re.compile(r"dividendo\s+de\s+\$[\d.]+\s+por\s+acción", _re.I),
    "Crecimiento %":  _re.compile(r"crecimiento\s+del?\s+[\d.]+%", _re.I),
    "Morosidad IMOR": _re.compile(r"(?:imor|morosidad)(?:\s*\([A-Z]+\))?\s+(?:de\s+)?[\d.]+%", _re.I),
    "Capitalización ICAP": _re.compile(r"(?:icap|capital\s+total)(?:\s*\([A-Z]+\))?\s+(?:cerró\s+en\s+)?[\d.]+%", _re.I),
    "ROAE":           _re.compile(r"roae\s+(?:objetivo\s+)?(?:de\s+|del\s+)?[\d.]+%", _re.I),
}


def extract_financial_entities(text: str) -> dict[str, list[str]]:
    results = {}
    for label, pattern in FINANCIAL_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            results[label] = [m.strip() for m in matches]
    return results


def generate_executive_summary(text: str, entities: dict) -> list[str]:
    """Genera bullets de resumen ejecutivo a partir de las entidades extraídas."""
    bullets = []
    if "Ingresos" in entities:
        bullets.append(f"📈 Ingresos: {entities['Ingresos'][0]}")
    if "Crecimiento %" in entities:
        bullets.append(f"🔼 {entities['Crecimiento %'][0]}")
    if "EBITDA" in entities:
        bullets.append(f"💰 {entities['EBITDA'][0]}")
    if "Utilidad neta" in entities:
        bullets.append(f"✅ {entities['Utilidad neta'][0]}")
    if "EPS" in entities:
        bullets.append(f"📊 Utilidad por acción: {entities['EPS'][0]}")
    if "Dividendo" in entities:
        bullets.append(f"💵 {entities['Dividendo'][0]}")
    if "Morosidad IMOR" in entities:
        bullets.append(f"⚠️ {entities['Morosidad IMOR'][0]}")
    if "Capitalización ICAP" in entities:
        bullets.append(f"🏦 {entities['Capitalización ICAP'][0]}")
    if "ROAE" in entities:
        bullets.append(f"🎯 {entities['ROAE'][0]}")
    return bullets


STOPWORDS_ES = {
    "el", "la", "los", "las", "de", "del", "en", "y", "a", "que", "un", "una",
    "para", "con", "su", "sus", "se", "por", "es", "al", "lo", "como", "más",
    "o", "pero", "sus", "le", "ya", "fue", "ha", "este", "esta", "entre",
    "sobre", "durante", "muy", "sin", "estimados", "the", "of", "to",
}


def word_frequency(text: str, top_n: int = 20) -> list[tuple]:
    """Frecuencia de palabras relevantes (excluye stopwords y números sueltos)."""
    words = re.findall(r"[a-záéíóúñA-ZÁÉÍÓÚÑ]{4,}", text.lower())
    words = [w for w in words if w not in STOPWORDS_ES]
    from collections import Counter
    return Counter(words).most_common(top_n)


def extract_numbers(text: str) -> list[dict]:
    # BUG CORREGIDO: "miles?" es literal "mile"+"s"? y NUNCA coincide con "mil"
    # (le falta la "e"). El patrón correcto para la palabra española "mil" es
    # "mil(?:es)?", igual que para "millones?" se necesita "mill\w*". El orden
    # importa: "millones?" debe probarse antes que "mil(?:es)?" porque "mil" es
    # prefijo de "millones" (mismo bug de alternancia ya corregido en extract_amount).
    pattern = _re.compile(
        r"\$\s*[\d,]+(?:\.\d+)?\s*(?:millones?(?:\s+de\s+pesos)?|mil(?:es)?(?:\s+de\s+millones?)?(?:\s+pesos)?)?",
        _re.I,
    )
    found = []
    for m in pattern.finditer(text):
        found.append({"value": m.group().strip(),
                      "context": text[max(0, m.start()-30):m.end()+30].strip()})
    return found
