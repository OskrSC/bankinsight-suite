"""
Módulo 9 — Predicción de Ganancias Corporativas
XGBoost Regressor + SHAP · proyección de EPS · sensibilidad a sentimiento de mercado ·
comparativa sectorial
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from utils.helpers import (
    gen_earnings_data, build_earnings_model,
    plot_shap_bar, hex_to_rgba, base_layout, COLORS,
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetric"]{background:#f0f9ff;border:1px solid #bae6fd;
  border-radius:12px;padding:1rem 1.25rem}
.section-title{font-size:1.05rem;font-weight:700;color:#0f172a;
  border-left:4px solid #dc2626;padding-left:12px;margin:1.5rem 0 1rem}
.info-box{background:#f0f9ff;border:1px solid #7dd3fc;border-radius:10px;
  padding:11px 15px;font-size:13px;color:#0c4a6e;margin:8px 0}
</style>
""", unsafe_allow_html=True)

FEAT_LABELS = {
    "revenue": "Ingresos", "costs": "Costos operativos",
    "market_sentiment": "Sentimiento de mercado", "market_cap": "Capitalización",
    "pe_ratio": "Ratio P/E", "revenue_growth": "Crecimiento de ingresos",
}

# ── Datos y modelo ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Entrenando modelo de predicción de ganancias…")
def load():
    df = gen_earnings_data(2000)
    return df, *build_earnings_model(df)

df, model, scaler, feats, X_te, y_te, y_pred, mae, r2, sv_global, X_te_s = load()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 📊 Predicción de Ganancias Corporativas")
st.markdown(
    "Modelo **XGBoost Regressor** que proyecta el **EPS (Earnings Per Share)** de una "
    "empresa a partir de ingresos, costos, sentimiento de mercado y métricas de "
    "valuación. Incluye explicabilidad SHAP y análisis de sensibilidad."
)
st.markdown("---")

c1, c2, c3, c4 = st.columns(4)
c1.metric("R² del modelo",  f"{r2:.3f}",  "XGBoost calibrado")
c2.metric("MAE",            f"${mae:.2f}", "Error absoluto medio en EPS")
c3.metric("EPS promedio",   f"${df['eps'].mean():.2f}")
c4.metric("Empresas en dataset", f"{len(df):,}")

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — Proyección individual
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">🏢 Proyección de EPS por Empresa</div>', unsafe_allow_html=True)

with st.form("earnings_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        revenue = st.number_input("Ingresos (miles USD)", 5_000, 200_000, 60_000, step=1_000)
        costs   = st.number_input("Costos operativos (miles USD)", 2_000, 180_000, 35_000, step=1_000)
    with c2:
        sentiment = st.slider("Sentimiento de mercado (0-1)", 0.0, 1.0, 0.55, step=0.01,
                              help="0 = muy negativo, 1 = muy positivo")
        market_cap = st.number_input("Capitalización de mercado (miles USD)",
                                     50_000, 3_000_000, 800_000, step=10_000)
    with c3:
        pe_ratio = st.slider("Ratio P/E", 4.0, 60.0, 20.0, step=0.5)
        rev_growth = st.slider("Crecimiento de ingresos (%)", -20.0, 40.0, 10.0, step=0.5) / 100
    submitted = st.form_submit_button(
        "📊 Proyectar EPS", use_container_width=True, type="primary"
    )

if submitted:
    row = pd.DataFrame([{
        "revenue": revenue, "costs": costs, "market_sentiment": sentiment,
        "market_cap": market_cap, "pe_ratio": pe_ratio, "revenue_growth": rev_growth,
    }])
    X_in = scaler.transform(row[feats])
    eps_pred = float(model.predict(X_in)[0])

    margin = (revenue - costs) / revenue if revenue > 0 else 0
    implied_price = eps_pred * pe_ratio

    import shap as _shap
    exp_local = _shap.TreeExplainer(model)
    sv_raw = exp_local.shap_values(X_in)
    sv_arr = np.array(sv_raw)
    sv_f = sv_arr[0] if sv_arr.ndim == 2 else sv_arr.reshape(-1)

    r1, r2c, r3 = st.columns([1, 1, 2])
    with r1:
        color = COLORS["success"] if eps_pred > 0 else COLORS["danger"]
        st.markdown(f"""
        <div style='background:{hex_to_rgba(color,.08)};border:2px solid {color};
             border-radius:14px;padding:22px;text-align:center'>
          <div style='font-size:0.8rem;color:#64748b;font-weight:600'>EPS PROYECTADO</div>
          <div style='font-size:2rem;font-weight:800;color:{color};margin:6px 0'>
            ${eps_pred:.2f}
          </div>
          <div style='font-size:0.78rem;color:#64748b'>±${mae:.2f} (MAE del modelo)</div>
        </div>""", unsafe_allow_html=True)
        st.metric("Precio implícito (EPS × P/E)", f"${implied_price:.2f}")
    with r2c:
        st.metric("Margen operativo", f"{margin:.1%}",
                  "Saludable" if margin > 0.30 else "Ajustado")
        st.metric("Utilidad neta estimada (miles)", f"${revenue - costs:,.0f}")
    with r3:
        contrib = [(FEAT_LABELS.get(feats[i], feats[i]), float(sv_f[i]))
                  for i in range(len(feats))]
        contrib.sort(key=lambda x: abs(x[1]), reverse=True)
        labels = [c[0] for c in contrib]
        vals   = [c[1] for c in contrib]
        bcolors = [COLORS["success"] if v > 0 else COLORS["danger"] for v in vals]
        fig_local = go.Figure(go.Bar(
            x=vals, y=labels, orientation="h", marker_color=bcolors,
            hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
        ))
        fig_local.update_layout(
            title="Drivers del EPS (SHAP local)<br><sup>Verde = aumenta EPS · Rojo = reduce EPS</sup>",
            xaxis_title="Contribución al EPS ($)",
            height=300, paper_bgcolor="white", plot_bgcolor=COLORS["bg"],
            font=dict(size=11), margin=dict(t=48, b=20, l=160, r=20),
        )
        st.plotly_chart(fig_local, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — Análisis de sensibilidad al sentimiento de mercado
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">📈 Sensibilidad al Sentimiento de Mercado</div>', unsafe_allow_html=True)
st.markdown("Mantén constantes los fundamentales y observa cómo varía el EPS proyectado solo con el sentimiento del mercado.")

if submitted:
    sent_range = np.linspace(0, 1, 41)
    eps_sens = []
    for s in sent_range:
        r = pd.DataFrame([{
            "revenue": revenue, "costs": costs, "market_sentiment": s,
            "market_cap": market_cap, "pe_ratio": pe_ratio, "revenue_growth": rev_growth,
        }])
        X_s = scaler.transform(r[feats])
        eps_sens.append(float(model.predict(X_s)[0]))

    fig_sens = go.Figure(go.Scatter(
        x=sent_range, y=eps_sens, mode="lines",
        line=dict(color=COLORS["primary"], width=2.5),
        fill="tozeroy", fillcolor=hex_to_rgba(COLORS["primary"], 0.08),
        hovertemplate="Sentimiento: %{x:.2f}<br>EPS: $%{y:.2f}<extra></extra>",
    ))
    fig_sens.add_vline(x=sentiment, line_dash="dash", line_color=COLORS["danger"],
                       annotation_text=f"Tu escenario: {sentiment:.2f}")
    fig_sens.update_layout(
        title="EPS Proyectado vs Sentimiento de Mercado",
        xaxis_title="Sentimiento de mercado (0=negativo, 1=positivo)",
        yaxis_title="EPS proyectado ($)",
        **base_layout(),
    )
    st.plotly_chart(fig_sens, use_container_width=True)
else:
    st.info("Completa el formulario de la sección anterior y presiona 'Proyectar EPS' para ver el análisis de sensibilidad.")

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — Rendimiento del modelo
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">🎯 Rendimiento del Modelo</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📍 Predicho vs Real", "🧠 SHAP Global", "📉 Distribución de errores"])
with tab1:
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=y_te, y=y_pred, mode="markers",
        marker=dict(color=COLORS["primary"], opacity=0.5, size=6),
        hovertemplate="Real: $%{x:.2f}<br>Predicho: $%{y:.2f}<extra></extra>",
    ))
    min_v, max_v = float(y_te.min()), float(y_te.max())
    fig_pred.add_trace(go.Scatter(
        x=[min_v, max_v], y=[min_v, max_v], mode="lines",
        line=dict(color=COLORS["danger"], dash="dash"), name="Predicción perfecta",
    ))
    fig_pred.update_layout(
        title=f"EPS Predicho vs Real (R²={r2:.3f})",
        xaxis_title="EPS real ($)", yaxis_title="EPS predicho ($)",
        **base_layout(),
    )
    st.plotly_chart(fig_pred, use_container_width=True)
with tab2:
    st.plotly_chart(
        plot_shap_bar(sv_global, feats, title="Importancia SHAP Global — EPS"),
        use_container_width=True,
    )
    st.markdown('<div class="info-box">Los ingresos y costos (margen operativo) suelen dominar la predicción de EPS, seguidos por el sentimiento de mercado — un factor "soft" que el modelo aprende a ponderar.</div>', unsafe_allow_html=True)
with tab3:
    errors = y_te - y_pred
    fig_err = go.Figure(go.Histogram(
        x=errors, nbinsx=40, marker_color=COLORS["primary"], opacity=0.8,
    ))
    fig_err.add_vline(x=0, line_dash="dash", line_color="#0f172a")
    fig_err.update_layout(
        title="Distribución de Errores (Real − Predicho)",
        xaxis_title="Error en EPS ($)", yaxis_title="Frecuencia",
        **base_layout(),
    )
    st.plotly_chart(fig_err, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — Explorador comparativo
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">🔍 Explorador de Empresas Simuladas</div>', unsafe_allow_html=True)

color_feat = st.selectbox(
    "Colorear por", ["market_sentiment", "pe_ratio", "revenue_growth"],
    format_func=lambda x: FEAT_LABELS.get(x, x),
)
sample = df.sample(min(800, len(df)), random_state=42)
fig_sc = px.scatter(
    sample, x="revenue", y="eps", color=color_feat,
    color_continuous_scale="RdYlGn",
    labels={"revenue": "Ingresos (miles)", "eps": "EPS ($)",
           color_feat: FEAT_LABELS.get(color_feat, color_feat)},
    title=f"Ingresos vs EPS (color: {FEAT_LABELS.get(color_feat, color_feat)})",
)
fig_sc.update_layout(paper_bgcolor="white", plot_bgcolor=COLORS["bg"],
                      margin=dict(t=44, b=44, l=60, r=20))
st.plotly_chart(fig_sc, use_container_width=True)

with st.expander("📋 Ver muestra del dataset"):
    show = sample.head(100).rename(columns=FEAT_LABELS)
    st.dataframe(
        show.style.format({
            "Ingresos": "${:,.0f}", "Costos operativos": "${:,.0f}",
            "Capitalización": "${:,.0f}", "eps": "${:.2f}",
        }),
        use_container_width=True, height=350,
    )
