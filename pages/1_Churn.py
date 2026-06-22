"""
Módulo 1 — Predicción de Churn Bancario
LightGBM + SMOTE · SHAP local/global · plan de retención · explorador de portafolio
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import shap as _shap

from utils.helpers import (
    gen_churn_data, build_churn_model,
    plot_roc, plot_pr, plot_conf_matrix, plot_shap_bar,
    plot_shap_local, plot_gauge,
    extract_shap_local, hex_to_rgba,
    COLORS, SEGMENT_COLORS,
)
from sklearn.metrics import precision_score, recall_score, f1_score

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetric"]{background:#f0f9ff;border:1px solid #bae6fd;
  border-radius:12px;padding:1rem 1.25rem}
.section-title{font-size:1.05rem;font-weight:700;color:#0f172a;
  border-left:4px solid #0ea5e9;padding-left:12px;margin:1.5rem 0 1rem}
.info-box{background:#f0f9ff;border:1px solid #7dd3fc;border-radius:10px;
  padding:11px 15px;font-size:13px;color:#0c4a6e;margin:8px 0}
.warn-box{background:#fffbeb;border:1px solid #fcd34d;border-radius:10px;
  padding:11px 15px;font-size:13px;color:#78350f;margin:8px 0}
.danger-box{background:#fef2f2;border:1px solid #fca5a5;border-radius:10px;
  padding:11px 15px;font-size:13px;color:#7f1d1d;margin:8px 0}
.success-box{background:#f0fdf4;border:1px solid #86efac;border-radius:10px;
  padding:11px 15px;font-size:13px;color:#14532d;margin:8px 0}
</style>
""", unsafe_allow_html=True)

# ── Datos y modelo ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Entrenando modelo de churn (LightGBM + SMOTE)…")
def load():
    df = gen_churn_data(5000)
    return df, *build_churn_model(df)

df, model, scaler, feats, X_te, y_te, y_prob, auc, sv_global, X_te_s = load()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 📉 Predicción de Churn Bancario")
st.markdown(
    "Modelo **LightGBM + SMOTE** que estima la probabilidad de que un cliente abandone "
    "el banco en los próximos 90 días. Incluye explicabilidad SHAP individual y "
    "plan de retención automático basado en los factores de mayor impacto."
)
st.markdown("---")

churn_rate = float(df["churn"].mean())
c1, c2, c3, c4 = st.columns(4)
c1.metric("AUC-ROC",               f"{auc:.3f}",        "LightGBM calibrado")
c2.metric("Tasa de churn",         f"{churn_rate:.1%}", "Dataset sintético")
c3.metric("Clientes en dataset",   f"{len(df):,}")
c4.metric("Balanceo de clases",    "SMOTE",             "Minoría sobremuestreada")

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — Evaluación individual
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">⚙️ Evaluación de Cliente Individual</div>',
            unsafe_allow_html=True)

GEO = {"CDMX": 0, "Guadalajara": 1, "Monterrey": 2, "Otros": 3}
FEAT_LABELS = {
    "age": "Edad", "balance": "Saldo (MXN)", "num_transactions": "Tx últimos 30d",
    "tenure_years": "Antigüedad (años)", "num_products": "Productos",
    "has_credit_card": "Tiene tarjeta crédito", "is_active": "Miembro activo",
    "credit_score": "Score crediticio", "complaints_12m": "Quejas 12 meses",
    "est_salary": "Ingreso anual (MXN)", "geo_enc": "Ciudad",
}

with st.form("churn_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        age    = st.slider("Edad", 18, 75, 42)
        bal    = st.number_input("Saldo promedio (MXN)", 0, 120_000, 15_000, step=500)
        ntx    = st.slider("Transacciones últimos 30 días", 0, 50, 12)
        tenure = st.slider("Años como cliente", 1, 30, 5)
    with c2:
        nprod  = st.slider("Productos contratados", 1, 5, 2)
        has_cc = st.selectbox("¿Tiene tarjeta de crédito?", [1, 0],
                              format_func=lambda x: "Sí" if x else "No")
        active = st.selectbox("¿Miembro activo?", [1, 0],
                              format_func=lambda x: "Sí" if x else "No")
        cscore = st.slider("Score crediticio", 300, 850, 650, step=5)
    with c3:
        compl  = st.slider("Quejas en últimos 12 meses", 0, 6, 1)
        salary = st.number_input("Ingreso anual estimado (MXN)", 8_000, 250_000, 60_000, step=5_000)
        geo    = st.selectbox("Ciudad", list(GEO.keys()))
    submitted = st.form_submit_button(
        "🔍 Calcular riesgo de churn", use_container_width=True, type="primary"
    )

if submitted:
    row = pd.DataFrame([{
        "age": age, "balance": bal, "num_transactions": ntx,
        "tenure_years": tenure, "num_products": nprod,
        "has_credit_card": has_cc, "is_active": active,
        "credit_score": cscore, "complaints_12m": compl,
        "est_salary": salary, "geo_enc": GEO[geo],
    }])
    X_in = scaler.transform(row[feats])
    prob = float(model.predict_proba(X_in)[0, 1])

    risk_label = ("🔴 ALTO"   if prob > 0.65 else
                  "🟡 MEDIO"  if prob > 0.35 else "🟢 BAJO")
    color = (COLORS["danger"]  if prob > 0.65 else
             COLORS["warning"] if prob > 0.35 else COLORS["success"])

    # SHAP local
    exp    = _shap.TreeExplainer(model)
    sv_raw = exp.shap_values(X_in)
    sv_f   = extract_shap_local(sv_raw)

    # Layout resultado
    r1, r2, r3 = st.columns([1, 1, 2])
    with r1:
        st.plotly_chart(
            plot_gauge(prob * 100, "Prob. Churn (%)"),
            use_container_width=True,
        )
    with r2:
        st.markdown(f"""
        <div style='background:{hex_to_rgba(color,.08)};border:2px solid {color};
             border-radius:12px;padding:20px;text-align:center;margin-top:8px'>
          <div style='font-size:2rem;font-weight:800;color:{color}'>{prob:.1%}</div>
          <div style='font-size:0.9rem;font-weight:600;color:{color}'>Probabilidad de Churn</div>
          <div style='font-size:0.8rem;color:#64748b;margin-top:4px'>Riesgo: {risk_label}</div>
        </div>""", unsafe_allow_html=True)

        # ── Plan de retención ─────────────────────────────────────────────────
        st.markdown("<br>**🎯 Plan de retención:**", unsafe_allow_html=True)
        actions = []
        if compl >= 2:
            actions.append("📞 Llamada proactiva — resolver quejas pendientes")
        if not active:
            actions.append("💳 Campaña de reactivación con 2% cashback (90 días)")
        if nprod <= 1:
            actions.append("🎁 Oferta cruzada: seguro de vida o cuenta de inversión")
        if bal < 5_000:
            actions.append("💰 Cuenta de ahorro con tasa preferencial +1.5%")
        if tenure >= 5 and prob > 0.4:
            actions.append("🏅 Upgrade a cliente preferente — gestor dedicado")
        if cscore < 600:
            actions.append("📈 Programa de mejora de score crediticio")
        if not actions:
            actions.append("✅ Cliente estable — monitoreo mensual rutinario")
        for a in actions:
            st.markdown(f"- {a}")

    with r3:
        fig_local = plot_shap_local(sv_f, feats, title="Drivers del churn (SHAP individual)")
        fig_local.update_layout(height=320)
        st.plotly_chart(fig_local, use_container_width=True)

    # ── Comparación con promedios del portafolio ──────────────────────────────
    st.markdown("**Comparativa con promedios del portafolio:**")
    comp_data = {
        "Variable":   ["Saldo", "Tx/mes", "Antigüedad", "Score", "Quejas", "Productos"],
        "Este cliente": [f"${bal:,.0f}", str(ntx), f"{tenure}a", str(cscore), str(compl), str(nprod)],
        "Portafolio":   [
            f"${df['balance'].mean():,.0f}",
            f"{df['num_transactions'].mean():.0f}",
            f"{df['tenure_years'].mean():.0f}a",
            f"{df['credit_score'].mean():.0f}",
            f"{df['complaints_12m'].mean():.1f}",
            f"{df['num_products'].mean():.1f}",
        ],
        "Clientes churn": [
            f"${df.loc[df.churn==1,'balance'].mean():,.0f}",
            f"{df.loc[df.churn==1,'num_transactions'].mean():.0f}",
            f"{df.loc[df.churn==1,'tenure_years'].mean():.0f}a",
            f"{df.loc[df.churn==1,'credit_score'].mean():.0f}",
            f"{df.loc[df.churn==1,'complaints_12m'].mean():.1f}",
            f"{df.loc[df.churn==1,'num_products'].mean():.1f}",
        ],
    }
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — Umbral de decisión y métricas operativas
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">📐 Umbral de Decisión y Métricas Operativas</div>',
            unsafe_allow_html=True)

col_thr, col_info = st.columns([3, 1])
with col_thr:
    threshold = st.slider(
        "Umbral de clasificación (probabilidad de churn)", 0.10, 0.85, 0.40,
        step=0.01, help="Clientes con prob. ≥ umbral se marcan como 'en riesgo'"
    )
with col_info:
    st.markdown("""
    <div class="info-box" style="margin-top:28px">
    Un umbral más bajo captura más churn real (↑ recall) pero genera más falsas alarmas (↓ precisión).
    Ajusta según el costo de contactar vs perder un cliente.
    </div>""", unsafe_allow_html=True)

# Garantizar numpy int/float puro en ambos lados — evita el ValueError
# 'mix of unknown and binary targets' cuando y_te es pandas Series
# con índice no contiguo (train_test_split) en ciertas versiones de sklearn.
y_true_np = np.array(y_te,   dtype=int)
y_prob_np = np.array(y_prob,  dtype=float)
y_pred_t  = (y_prob_np >= threshold).astype(int)

prec    = precision_score(y_true_np, y_pred_t, zero_division=0)
rec     = recall_score(y_true_np,   y_pred_t, zero_division=0)
f1      = f1_score(y_true_np,       y_pred_t, zero_division=0)
flagged = int(y_pred_t.sum())
revenue_at_risk = flagged * df["balance"].mean() * 0.08

tc1, tc2, tc3, tc4, tc5 = st.columns(5)
tc1.metric("Precisión",       f"{prec:.3f}", "TP / (TP+FP)")
tc2.metric("Recall",          f"{rec:.3f}",  "TP / (TP+FN)")
tc3.metric("F1-Score",        f"{f1:.3f}")
tc4.metric("Clientes en riesgo", f"{flagged:,}", f"{flagged/len(y_pred_t):.1%} del test")
tc5.metric("Saldo en riesgo (proxy)", f"${revenue_at_risk/1e6:.1f}M MXN")

tab1, tab2, tab3, tab4 = st.tabs(["📈 Curva ROC", "📉 Prec-Recall", "🔢 Matriz Conf.", "🧠 SHAP Global"])
with tab1:
    st.plotly_chart(plot_roc(y_true_np, y_prob_np, "ROC — Modelo de Churn"), use_container_width=True)
with tab2:
    st.plotly_chart(plot_pr(y_true_np, y_prob_np, "Precisión-Recall — Churn"), use_container_width=True)
    st.markdown('<div class="warn-box">⚠️ Con churn ~22%, la curva Precisión-Recall es más informativa que la ROC para evaluar el modelo en producción.</div>', unsafe_allow_html=True)
with tab3:
    st.plotly_chart(plot_conf_matrix(y_true_np, y_pred_t, ["No Churn", "Churn"]),
                    use_container_width=True)
with tab4:
    st.plotly_chart(
        plot_shap_bar(sv_global, feats, title="Importancia SHAP Global — Churn"),
        use_container_width=True,
    )
    st.markdown('<div class="info-box">SHAP mide la contribución promedio de cada variable al cambio en la predicción. Variables en la cima tienen el mayor efecto en la probabilidad de churn a nivel de portafolio.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — Explorador del portafolio
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">🗂️ Explorador del Portafolio</div>', unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def score_portfolio():
    s = df.sample(min(800, len(df)), random_state=42).copy()
    Xs = scaler.transform(s[feats])
    s["prob_churn"] = model.predict_proba(Xs)[:, 1]
    s["riesgo"] = pd.cut(s["prob_churn"], bins=[0, .35, .65, 1.01],
                          labels=["Bajo", "Medio", "Alto"])
    return s

sample = score_portfolio()

# Filtros
fc1, fc2, fc3, fc4 = st.columns(4)
geo_f  = fc1.multiselect("Ciudad", df["geography"].unique().tolist(),
                          default=df["geography"].unique().tolist())
risk_f = fc2.multiselect("Riesgo", ["Bajo", "Medio", "Alto"],
                          default=["Bajo", "Medio", "Alto"])
minbal = fc3.slider("Saldo mínimo (MXN)", 0, 50_000, 0, step=1_000)
min_tx = fc4.slider("Transacciones mínimas", 0, 30, 0)

filt = sample[
    sample["geography"].isin(geo_f) &
    sample["riesgo"].isin(risk_f) &
    (sample["balance"] >= minbal) &
    (sample["num_transactions"] >= min_tx)
]

ca, cb = st.columns(2)
with ca:
    RISK_COLORS = {"Bajo": COLORS["success"], "Medio": COLORS["warning"], "Alto": COLORS["danger"]}
    fig_sc = px.scatter(
        filt, x="balance", y="prob_churn", color="riesgo",
        color_discrete_map=RISK_COLORS, opacity=0.65,
        labels={"balance": "Saldo (MXN)", "prob_churn": "P(Churn)"},
        title="Saldo vs Probabilidad de Churn",
        hover_data=["age", "num_products", "geography"],
    )
    fig_sc.update_layout(paper_bgcolor="white", plot_bgcolor=COLORS["bg"],
                          margin=dict(t=44, b=44, l=60, r=20))
    st.plotly_chart(fig_sc, use_container_width=True)

with cb:
    grp = (filt.groupby(["geography", "riesgo"], observed=True)
           .size().reset_index(name="n"))
    fig_bar = px.bar(
        grp, x="geography", y="n", color="riesgo",
        color_discrete_map=RISK_COLORS, barmode="stack",
        title="Distribución de Riesgo por Ciudad",
        labels={"geography": "Ciudad", "n": "Clientes"},
    )
    fig_bar.update_layout(paper_bgcolor="white", plot_bgcolor=COLORS["bg"],
                           margin=dict(t=44, b=44, l=60, r=20))
    st.plotly_chart(fig_bar, use_container_width=True)

# KPIs resumen
k1, k2, k3, k4 = st.columns(4)
k1.metric("Clientes filtrados", f"{len(filt):,}")
k2.metric("En riesgo alto",
          f"{(filt['riesgo'] == 'Alto').sum():,}",
          f"{(filt['riesgo'] == 'Alto').mean():.1%}")
k3.metric("Saldo medio — riesgo alto",
          f"${filt.loc[filt['riesgo']=='Alto','balance'].mean():,.0f}")
k4.metric("Churn esperado (umbral actual)",
          f"{(filt['prob_churn'] >= threshold).sum():,}",
          f"umbral {threshold:.0%}")

with st.expander("📋 Top 20 clientes con mayor riesgo de churn"):
    show = filt.nlargest(20, "prob_churn")[[
        "age", "balance", "num_transactions", "tenure_years",
        "num_products", "complaints_12m", "geography", "prob_churn", "riesgo",
    ]].rename(columns={
        "age": "Edad", "balance": "Saldo", "num_transactions": "Tx/mes",
        "tenure_years": "Antigüedad", "num_products": "Prods.",
        "complaints_12m": "Quejas", "geography": "Ciudad",
        "prob_churn": "P(Churn)", "riesgo": "Riesgo",
    })
    st.dataframe(
        show.style
        .format({"Saldo": "${:,.0f}", "P(Churn)": "{:.1%}"})
        .background_gradient(subset=["P(Churn)"], cmap="RdYlGn_r"),
        use_container_width=True, height=400,
    )
