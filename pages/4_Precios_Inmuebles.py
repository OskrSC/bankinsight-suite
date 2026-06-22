"""
Módulo 4 — Precios de Inmuebles
XGBoost Regressor + SHAP · valuación individual · mapa de zonas · explorador comparativo
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import shap as _shap

from utils.helpers import (
    gen_realestate_data, build_realestate_model,
    plot_shap_bar, hex_to_rgba, base_layout,
    extract_shap_local, COLORS,
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetric"]{background:#f0f9ff;border:1px solid #bae6fd;
  border-radius:12px;padding:1rem 1.25rem}
.section-title{font-size:1.05rem;font-weight:700;color:#0f172a;
  border-left:4px solid #16a34a;padding-left:12px;margin:1.5rem 0 1rem}
.info-box{background:#f0f9ff;border:1px solid #7dd3fc;border-radius:10px;
  padding:11px 15px;font-size:13px;color:#0c4a6e;margin:8px 0}
.warn-box{background:#fffbeb;border:1px solid #fcd34d;border-radius:10px;
  padding:11px 15px;font-size:13px;color:#78350f;margin:8px 0}
</style>
""", unsafe_allow_html=True)

ZONE_COLORS = {
    "Centro":   "#7c3aed",
    "Norte":    "#0ea5e9",
    "Sur":      "#16a34a",
    "Oriente":  "#d97706",
    "Poniente": "#dc2626",
}
ZONE_MAP   = {"Centro": 0, "Norte": 1, "Sur": 2, "Oriente": 3, "Poniente": 4}
ZONE_DESC  = {
    "Centro":   "Zona premium, alta densidad comercial y conectividad. Plusvalía histórica más alta.",
    "Norte":    "Crecimiento residencial sostenido, buena infraestructura vial y escolar.",
    "Sur":      "Mix residencial-industrial, precios competitivos, en proceso de renovación urbana.",
    "Oriente":  "Zona de expansión reciente, precios accesibles, oferta de vivienda nueva.",
    "Poniente": "Corredor corporativo y residencial de gama media-alta, fuerte demanda de renta.",
}

FEAT_LABELS = {
    "m2": "Tamaño (m²)", "rooms": "Habitaciones", "baths": "Baños",
    "age": "Antigüedad (años)", "parking": "Cajones estac.",
    "floor": "Piso", "amenities": "Amenidades", "dist_metro": "Dist. metro (km)",
    "zone_enc": "Zona",
}

# ── Datos y modelo ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Entrenando modelo de valuación (XGBoost + SHAP)…")
def load():
    df = gen_realestate_data(3000)
    return df, *build_realestate_model(df)

df, model, scaler, feats, X_te, y_te, y_pred, mae, mape, r2, sv_global, X_te_s = load()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🏠 Precios de Inmuebles")
st.markdown(
    "Modelo **XGBoost Regressor** entrenado sobre 3,000 propiedades sintéticas. "
    "Valúa una propiedad nueva en segundos, explica qué factores impulsan el precio "
    "y compara contra el inventario disponible."
)
st.markdown("---")

c1, c2, c3, c4 = st.columns(4)
c1.metric("R² del modelo",        f"{r2:.3f}",          "XGBoost calibrado")
c2.metric("Error absoluto medio", f"${mae:,.0f}",        "MAE en pesos")
c3.metric("Error porcentual",     f"{mape:.1%}",         "MAPE")
c4.metric("Precio medio dataset", f"${df['price'].mean():,.0f}")

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — Valuación individual
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">🧮 Valuación de Propiedad</div>', unsafe_allow_html=True)

with st.form("realestate_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        m2     = st.number_input("Tamaño (m²)", 28, 500, 110, step=5)
        rooms  = st.slider("Habitaciones", 1, 6, 3)
        baths  = st.slider("Baños", 1, 4, 2)
    with c2:
        age    = st.slider("Antigüedad (años)", 0, 50, 8)
        parking= st.slider("Cajones de estacionamiento", 0, 3, 1)
        floor  = st.slider("Piso", 0, 25, 4)
    with c3:
        amenities = st.slider("Amenidades (0-10)", 0, 10, 5,
                              help="Alberca, gym, roof garden, seguridad 24h, etc.")
        dist_metro= st.number_input("Distancia al metro (km)", 0.1, 10.0, 1.2, step=0.1)
        zone      = st.selectbox("Zona", list(ZONE_MAP.keys()))
    submitted = st.form_submit_button(
        "🏠 Valuar propiedad", use_container_width=True, type="primary"
    )

if submitted:
    row = pd.DataFrame([{
        "m2": m2, "rooms": rooms, "baths": baths, "age": age,
        "parking": parking, "floor": floor, "amenities": amenities,
        "dist_metro": dist_metro, "zone_enc": ZONE_MAP[zone],
    }])
    X_in  = scaler.transform(row[feats])
    price = float(model.predict(X_in)[0])

    # Intervalo de confianza aproximado con MAPE
    lo, hi = price * (1 - mape), price * (1 + mape)

    # SHAP local
    exp    = _shap.TreeExplainer(model)
    sv_raw = exp.shap_values(X_in)
    sv_f   = extract_shap_local(sv_raw) if isinstance(sv_raw, list) or np.array(sv_raw).ndim == 3 \
             else np.array(sv_raw).reshape(-1)

    color_z = ZONE_COLORS.get(zone, COLORS["neutral"])

    r1, r2c, r3 = st.columns([1, 1, 2])
    with r1:
        st.markdown(f"""
        <div style='background:{hex_to_rgba(color_z,.08)};border:2px solid {color_z};
             border-radius:14px;padding:22px;text-align:center'>
          <div style='font-size:0.8rem;color:#64748b;font-weight:600'>VALOR ESTIMADO</div>
          <div style='font-size:1.9rem;font-weight:800;color:{color_z};margin:6px 0'>
            ${price:,.0f}
          </div>
          <div style='font-size:0.78rem;color:#64748b'>
            Rango: ${lo:,.0f} – ${hi:,.0f}
          </div>
        </div>""", unsafe_allow_html=True)
        st.metric("Precio por m²", f"${price/m2:,.0f}/m²")
    with r2c:
        st.markdown(f"""
        <div style='background:{hex_to_rgba(color_z,.05)};border-left:4px solid {color_z};
             border-radius:8px;padding:12px 16px;margin-top:8px'>
          <div style='font-size:13px;font-weight:700;color:{color_z}'>📍 Zona {zone}</div>
          <div style='font-size:12px;color:#475569;margin-top:4px'>{ZONE_DESC[zone]}</div>
        </div>""", unsafe_allow_html=True)

        zone_avg = df.loc[df["zone"] == zone, "price"].mean()
        diff_pct = (price / zone_avg - 1)
        st.metric("vs. precio medio de la zona",
                  f"${zone_avg:,.0f}",
                  f"{diff_pct:+.1%}")
    with r3:
        fig_local = go.Figure(go.Bar(
            x=sv_f, y=[FEAT_LABELS.get(f, f) for f in feats],
            orientation="h",
            marker_color=[COLORS["success"] if v > 0 else COLORS["danger"] for v in sv_f],
            hovertemplate="%{y}: %{x:+,.0f}<extra></extra>",
        ))
        fig_local.update_layout(
            title="Factores que impulsan el precio (SHAP)<br><sup>Verde = aumenta valor · Rojo = reduce valor</sup>",
            xaxis_title="Contribución al precio ($)",
            height=320, paper_bgcolor="white", plot_bgcolor=COLORS["bg"],
            font=dict(size=11), margin=dict(t=48, b=20, l=130, r=20),
        )
        st.plotly_chart(fig_local, use_container_width=True)

    # Comparables similares
    st.markdown("**Propiedades comparables en el dataset (misma zona, tamaño ±20%):**")
    comparables = df[
        (df["zone"] == zone) &
        (df["m2"].between(m2 * 0.8, m2 * 1.2))
    ].nsmallest(8, "dist_metro")[
        ["m2", "rooms", "baths", "age", "amenities", "dist_metro", "price"]
    ].rename(columns={
        "m2": "m²", "rooms": "Hab.", "baths": "Baños", "age": "Antigüedad",
        "amenities": "Amenidades", "dist_metro": "Dist. metro", "price": "Precio",
    })
    if len(comparables):
        st.dataframe(
            comparables.style.format({"Precio": "${:,.0f}", "Dist. metro": "{:.2f} km"}),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No se encontraron comparables exactos en el dataset para estos parámetros.")

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — Importancia de variables y rendimiento del modelo
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">📊 Rendimiento del Modelo</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🎯 Predicho vs Real", "🧠 SHAP Global", "📉 Distribución de errores"])
with tab1:
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=y_te, y=y_pred, mode="markers",
        marker=dict(color=COLORS["primary"], opacity=0.5, size=6),
        hovertemplate="Real: $%{x:,.0f}<br>Predicho: $%{y:,.0f}<extra></extra>",
    ))
    min_v, max_v = float(y_te.min()), float(y_te.max())
    fig_pred.add_trace(go.Scatter(
        x=[min_v, max_v], y=[min_v, max_v], mode="lines",
        line=dict(color=COLORS["danger"], dash="dash"),
        name="Predicción perfecta",
    ))
    fig_pred.update_layout(
        title=f"Precio Predicho vs Real (R²={r2:.3f})",
        xaxis_title="Precio real ($)", yaxis_title="Precio predicho ($)",
        **base_layout(),
    )
    st.plotly_chart(fig_pred, use_container_width=True)
with tab2:
    st.plotly_chart(
        plot_shap_bar(sv_global, feats, title="Importancia SHAP Global — Precio de Inmuebles"),
        use_container_width=True,
    )
    st.markdown('<div class="info-box">El tamaño (m²) y la zona suelen ser los predictores más fuertes del precio. La distancia al metro tiene impacto negativo: a mayor distancia, menor precio.</div>', unsafe_allow_html=True)
with tab3:
    errors = y_te - y_pred
    fig_err = go.Figure(go.Histogram(
        x=errors, nbinsx=40, marker_color=COLORS["primary"], opacity=0.8,
    ))
    fig_err.add_vline(x=0, line_dash="dash", line_color="#0f172a")
    fig_err.update_layout(
        title="Distribución de Errores (Real − Predicho)",
        xaxis_title="Error ($)", yaxis_title="Frecuencia",
        **base_layout(),
    )
    st.plotly_chart(fig_err, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — Análisis por zona
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">🗺️ Análisis Comparativo por Zona</div>', unsafe_allow_html=True)

ca, cb = st.columns(2)
with ca:
    zone_stats = df.groupby("zone").agg(
        precio_medio=("price", "mean"),
        precio_m2=("price", lambda x: (x / df.loc[x.index, "m2"]).mean()),
        n=("price", "size"),
    ).reset_index()
    fig_zone = go.Figure(go.Bar(
        x=zone_stats["zone"], y=zone_stats["precio_medio"],
        marker_color=[ZONE_COLORS[z] for z in zone_stats["zone"]],
        text=[f"${v:,.0f}" for v in zone_stats["precio_medio"]],
        textposition="outside",
        hovertemplate="%{x}: $%{y:,.0f}<extra></extra>",
    ))
    fig_zone.update_layout(
        title="Precio Promedio por Zona",
        xaxis_title="Zona", yaxis_title="Precio promedio ($)",
        **base_layout(),
    )
    st.plotly_chart(fig_zone, use_container_width=True)

with cb:
    fig_box = px.box(
        df, x="zone", y="price", color="zone",
        color_discrete_map=ZONE_COLORS,
        title="Distribución de Precios por Zona",
        points="outliers",
        labels={"zone": "Zona", "price": "Precio ($)"},
    )
    fig_box.update_layout(showlegend=False, paper_bgcolor="white",
                          plot_bgcolor=COLORS["bg"],
                          margin=dict(t=44, b=44, l=60, r=20))
    st.plotly_chart(fig_box, use_container_width=True)

# Scatter interactivo m² vs precio
st.markdown("**Relación tamaño vs precio (interactivo):**")
color_feat = st.selectbox(
    "Colorear por", ["zone", "age", "amenities", "dist_metro"],
    format_func=lambda x: FEAT_LABELS.get(x, x.title()),
)
sample = df.sample(min(1000, len(df)), random_state=42)
if color_feat == "zone":
    fig_sc = px.scatter(
        sample, x="m2", y="price", color="zone",
        color_discrete_map=ZONE_COLORS, opacity=0.6,
        labels={"m2": "Tamaño (m²)", "price": "Precio ($)"},
        title="Tamaño vs Precio por Zona",
    )
else:
    fig_sc = px.scatter(
        sample, x="m2", y="price", color=color_feat,
        color_continuous_scale="Viridis", opacity=0.6,
        labels={"m2": "Tamaño (m²)", "price": "Precio ($)"},
        title=f"Tamaño vs Precio (color: {FEAT_LABELS.get(color_feat, color_feat)})",
    )
fig_sc.update_layout(paper_bgcolor="white", plot_bgcolor=COLORS["bg"],
                      margin=dict(t=44, b=44, l=60, r=20))
st.plotly_chart(fig_sc, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — Explorador del inventario
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">🔍 Explorador del Inventario</div>', unsafe_allow_html=True)

fc1, fc2, fc3, fc4 = st.columns(4)
zone_f  = fc1.multiselect("Zona", list(ZONE_MAP.keys()), default=list(ZONE_MAP.keys()))
price_r = fc2.slider("Rango de precio ($)", 300_000, 15_000_000,
                     (300_000, 15_000_000), step=100_000, format="$%d")
m2_min  = fc3.slider("m² mínimo", 28, 500, 28)
rooms_f = fc4.multiselect("Habitaciones", sorted(df["rooms"].unique().tolist()),
                          default=sorted(df["rooms"].unique().tolist()))

filt = df[
    df["zone"].isin(zone_f) &
    df["price"].between(*price_r) &
    (df["m2"] >= m2_min) &
    df["rooms"].isin(rooms_f)
]

k1, k2, k3 = st.columns(3)
k1.metric("Propiedades encontradas", f"{len(filt):,}")
k2.metric("Precio medio del filtro", f"${filt['price'].mean():,.0f}" if len(filt) else "N/A")
k3.metric("Precio/m² medio",
          f"${(filt['price']/filt['m2']).mean():,.0f}/m²" if len(filt) else "N/A")

with st.expander("📋 Ver propiedades filtradas", expanded=False):
    show = filt[["zone", "m2", "rooms", "baths", "age", "amenities",
                "dist_metro", "price"]].rename(columns={
        "zone": "Zona", "m2": "m²", "rooms": "Hab.", "baths": "Baños",
        "age": "Antigüedad", "amenities": "Amenidades",
        "dist_metro": "Dist. metro", "price": "Precio",
    }).sort_values("Precio", ascending=False)
    st.dataframe(
        show.style.format({"Precio": "${:,.0f}", "Dist. metro": "{:.2f} km"})
        .background_gradient(subset=["Precio"], cmap="Greens"),
        use_container_width=True, height=400,
    )
