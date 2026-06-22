"""
Módulo 2 — Segmentación de Clientes Bancarios
K-Means · Elbow Method · perfiles detallados · acciones de marketing · clasificador interactivo
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from utils.helpers import (
    gen_segment_data, build_segment_model,
    hex_to_rgba, base_layout,
    COLORS, SEGMENT_COLORS,
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetric"]{background:#f0f9ff;border:1px solid #bae6fd;
  border-radius:12px;padding:1rem 1.25rem}
.section-title{font-size:1.05rem;font-weight:700;color:#0f172a;
  border-left:4px solid #7c3aed;padding-left:12px;margin:1.5rem 0 1rem}
.info-box{background:#f0f9ff;border:1px solid #7dd3fc;border-radius:10px;
  padding:11px 15px;font-size:13px;color:#0c4a6e;margin:8px 0}
</style>
""", unsafe_allow_html=True)

# ── Acciones de marketing por segmento ───────────────────────────────────────
SEG_ACTIONS = {
    "Premium": [
        ("💎", "Programa de Banca Privada", "Asignación de gestor de patrimonio dedicado y acceso a productos estructurados"),
        ("📈", "Inversiones Alternativas",  "Acceso anticipado a fondos de capital privado, real estate y fibras"),
        ("✈️",  "Beneficios de Viaje",       "Acceso a salas VIP, seguro de viaje premium y concierge 24/7"),
        ("🏆", "Programa de Lealtad Elite", "Puntos triples, upgrades automáticos y invitación a eventos exclusivos"),
    ],
    "Activo": [
        ("💳", "Upgrade de Tarjeta",        "Tarjeta Platinum con 3% cashback y sin comisión anual primer año"),
        ("📱", "Adopción Digital",           "Incentivo por uso de banca móvil: $200 de bono en primeras 5 operaciones"),
        ("🏠", "Crédito Hipotecario",        "Pre-aprobación con tasa preferencial 10.5% y sin gastos de apertura"),
        ("💰", "Ahorro Programado",          "Plan de ahorro automático con rendimiento +1% sobre CETES"),
    ],
    "Ocasional": [
        ("📧", "Campaña Personalizada",     "Email + push con oferta de producto basada en historial de uso"),
        ("🎯", "Producto de Entrada",        "Cuenta ahorro con $500 de bono por apertura y domiciliación de nómina"),
        ("📲", "Activación Digital",         "Kit de bienvenida digital: 6 meses sin comisión en transferencias"),
        ("🤝", "Programa Referidos",         "Bono de $300 por cada amigo que abra cuenta y realice 3 operaciones"),
    ],
    "Inactivo": [
        ("📞", "Reactivación Proactiva",    "Llamada personalizada con oferta exclusiva: 5% de rendimiento 60 días"),
        ("💸", "Incentivo Cashback",         "20% de devolución en primeras 3 compras del mes de reactivación"),
        ("🔔", "Alerta de Valor",            "Notificación 'Tu dinero puede rendir más' con simulador de inversión"),
        ("⚠️",  "Evaluación de Cuenta",      "Revisión a 90 días: si no hay actividad, iniciar proceso de cierre ordenado"),
    ],
}

# ── Carga del modelo ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Parámetros del modelo")
    k = st.slider("Número de segmentos (k)", 2, 8, 4,
                  help="Usa el Elbow Method para encontrar el k óptimo")

@st.cache_data(show_spinner="Ejecutando K-Means clustering…")
def load(k_clusters: int):
    df   = gen_segment_data(4000)
    km, sc, feats, labels, Xs, inertias = build_segment_model(df, k_clusters)
    df["cluster"] = labels
    # Nombrar clusters por saldo promedio descendente
    order = (df.groupby("cluster")["balance"]
               .mean().sort_values(ascending=False).index.tolist())
    names = ["Premium", "Activo", "Ocasional", "Inactivo"][:k_clusters]
    names += [f"Seg-{i}" for i in range(k_clusters - 4)] if k_clusters > 4 else []
    name_map = {cl: names[i] for i, cl in enumerate(order)}
    df["segment"] = df["cluster"].map(name_map)
    return df, km, sc, feats, Xs, inertias, name_map

df, km, sc, feats, Xs, inertias, name_map = load(k)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🎯 Segmentación de Clientes")
st.markdown(
    "**K-Means clustering** sobre variables de comportamiento financiero. "
    "Segmenta la base de clientes en grupos homogéneos para diseñar "
    "estrategias de marketing, retención e inversión diferenciadas."
)
st.markdown("---")

seg_counts = df["segment"].value_counts()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Clientes analizados", f"{len(df):,}")
c2.metric("Segmentos activos",   str(k))
c3.metric("Variables usadas",    str(len(feats)))
c4.metric("Inercia final",       f"{km.inertia_:,.0f}")

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — Elbow + distribución
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">📊 Optimización del Número de Segmentos</div>',
            unsafe_allow_html=True)

ca, cb = st.columns(2)
with ca:
    # Elbow
    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(
        x=list(range(2, 9)), y=inertias,
        mode="lines+markers",
        line=dict(color=COLORS["primary"], width=2.5),
        marker=dict(size=8, color=COLORS["primary"],
                    line=dict(color="white", width=1.5)),
        hovertemplate="k=%{x}<br>Inercia=%{y:,.0f}<extra></extra>",
    ))
    fig_elbow.add_vline(
        x=k, line_dash="dash", line_color=COLORS["danger"],
        annotation_text=f"k={k} seleccionado",
        annotation_font_color=COLORS["danger"],
    )
    fig_elbow.update_layout(
        title="Método del Codo — Inercia vs k",
        xaxis_title="Número de clusters (k)",
        yaxis_title="Inercia (WCSS)",
        **base_layout(),
    )
    st.plotly_chart(fig_elbow, use_container_width=True)

with cb:
    color_list = [SEGMENT_COLORS.get(s, COLORS["neutral"]) for s in seg_counts.index]
    fig_pie = go.Figure(go.Pie(
        labels=seg_counts.index, values=seg_counts.values,
        marker_colors=color_list, hole=0.42,
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:,} clientes (%{percent})<extra></extra>",
    ))
    fig_pie.update_layout(
        title="Distribución de Clientes por Segmento",
        height=370, paper_bgcolor="white",
        margin=dict(t=44, b=20, l=20, r=20),
        legend=dict(orientation="h", y=-0.08),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — Perfiles
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">🧩 Perfil de cada Segmento</div>', unsafe_allow_html=True)

profile = (df.groupby("segment")[feats].mean().round(1)
             .join(df.groupby("segment").size().rename("n_clientes")))
seg_order = [s for s in ["Premium", "Activo", "Ocasional", "Inactivo"]
             if s in profile.index] + \
            [s for s in profile.index if s not in ["Premium","Activo","Ocasional","Inactivo"]]

# Tarjetas de perfil
cols_cards = st.columns(min(k, 4))
for col, seg in zip(cols_cards, seg_order):
    row   = profile.loc[seg]
    color = SEGMENT_COLORS.get(seg, COLORS["neutral"])
    with col:
        st.markdown(f"""
        <div style='background:{hex_to_rgba(color,.07)};border:2px solid {color};
             border-radius:12px;padding:14px;'>
          <div style='font-size:1rem;font-weight:700;color:{color}'>{seg}</div>
          <div style='font-size:11px;color:#64748b;margin-top:2px'>{int(row.n_clientes):,} clientes</div>
          <hr style='border-color:{color};opacity:0.25;margin:8px 0'>
          <div style='font-size:12px;line-height:2'>
            💰 <b>Saldo:</b> ${row.balance:,.0f}<br>
            🔄 <b>Tx/mes:</b> {row.transactions:.0f}<br>
            🏦 <b>Crédito:</b> ${row.loan:,.0f}<br>
            🎂 <b>Edad media:</b> {row.age:.0f}<br>
            📦 <b>Productos:</b> {row.products:.1f}<br>
            ⏸️ <b>Meses inactivo:</b> {row.inactive_months:.0f}
          </div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs de análisis ──────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["🕸️ Radar comparativo", "📊 Box plots", "🗺️ Mapa de dispersión", "📋 Tabla resumen"]
)
with tab1:
    RADAR_FEATS  = ["balance", "transactions", "loan", "products", "inactive_months"]
    RADAR_LABELS = ["Saldo", "Transacciones", "Crédito", "Productos", "Meses inactivo"]
    fig_radar = go.Figure()
    for seg in seg_order:
        row   = profile.loc[seg]
        color = SEGMENT_COLORS.get(seg, COLORS["neutral"])
        vals  = []
        for feat in RADAR_FEATS:
            mx = profile[feat].max()
            v  = float(row[feat]) / mx if mx > 0 else 0
            if feat == "inactive_months":
                v = 1 - v          # invertir: menos inactividad = mejor
            vals.append(round(v, 3))
        vals_c   = vals + [vals[0]]
        labels_c = RADAR_LABELS + [RADAR_LABELS[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_c, theta=labels_c, fill="toself",
            fillcolor=hex_to_rgba(color, 0.12),
            line=dict(color=color, width=2),
            name=seg,
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Perfil Normalizado por Segmento",
        height=440, paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.12),
        margin=dict(t=44, b=70, l=40, r=40),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

with tab2:
    feat_sel = st.selectbox(
        "Variable a comparar",
        feats,
        format_func=lambda x: x.replace("_", " ").title(),
        key="box_feat",
    )
    fig_box = px.box(
        df, x="segment", y=feat_sel, color="segment",
        color_discrete_map=SEGMENT_COLORS,
        category_orders={"segment": seg_order},
        title=f"Distribución de «{feat_sel.replace('_',' ').title()}» por Segmento",
        points="outliers",
    )
    fig_box.update_layout(showlegend=False, paper_bgcolor="white",
                           plot_bgcolor=COLORS["bg"],
                           margin=dict(t=44, b=44, l=60, r=20))
    st.plotly_chart(fig_box, use_container_width=True)

with tab3:
    x_feat = st.selectbox("Eje X", feats, index=0,
                           format_func=lambda x: x.replace("_"," ").title(), key="sx")
    y_feat = st.selectbox("Eje Y", feats, index=2,
                           format_func=lambda x: x.replace("_"," ").title(), key="sy")
    sample = df.sample(min(600, len(df)), random_state=42)
    fig_sp = px.scatter(
        sample, x=x_feat, y=y_feat, color="segment",
        color_discrete_map=SEGMENT_COLORS,
        opacity=0.65, size_max=7,
        title=f"{x_feat.replace('_',' ').title()} vs {y_feat.replace('_',' ').title()}",
        category_orders={"segment": seg_order},
    )
    fig_sp.update_layout(paper_bgcolor="white", plot_bgcolor=COLORS["bg"],
                          margin=dict(t=44, b=44, l=60, r=20))
    st.plotly_chart(fig_sp, use_container_width=True)

with tab4:
    disp = profile.rename(columns={
        "balance": "Saldo", "transactions": "Tx/mes", "loan": "Crédito",
        "age": "Edad", "products": "Productos",
        "inactive_months": "Meses inactivo", "n_clientes": "Clientes",
    })
    st.dataframe(
        disp.style
        .format({"Saldo": "${:,.0f}", "Crédito": "${:,.0f}",
                 "Tx/mes": "{:.0f}", "Edad": "{:.0f}",
                 "Productos": "{:.1f}", "Meses inactivo": "{:.0f}",
                 "Clientes": "{:,}"})
        .background_gradient(subset=["Saldo", "Tx/mes", "Productos"], cmap="Blues"),
        use_container_width=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — Estrategia de marketing
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">📣 Estrategia de Marketing por Segmento</div>',
            unsafe_allow_html=True)

seg_sel = st.selectbox("Segmento a detallar", seg_order, key="seg_mkt")
color_s = SEGMENT_COLORS.get(seg_sel, COLORS["neutral"])
actions = SEG_ACTIONS.get(seg_sel, [])

if actions:
    action_cols = st.columns(min(len(actions), 4))
    for col, (icon, title, desc) in zip(action_cols, actions):
        with col:
            st.markdown(f"""
            <div style='background:{hex_to_rgba(color_s,.06)};
                 border:1px solid {color_s}44;border-radius:10px;padding:12px;height:100%'>
              <div style='font-size:1.3rem'>{icon}</div>
              <div style='font-size:13px;font-weight:600;color:{color_s};margin:4px 0'>{title}</div>
              <div style='font-size:12px;color:#475569'>{desc}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — Clasificador de cliente nuevo
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">🔍 Clasificar Cliente Nuevo</div>', unsafe_allow_html=True)
st.markdown("Ingresa las características de un nuevo cliente para asignarlo automáticamente a un segmento.")

with st.form("seg_form"):
    s1, s2, s3 = st.columns(3)
    with s1:
        n_bal  = st.number_input("Saldo en cuenta (MXN)", 0, 200_000, 10_000, step=500)
        n_tx   = st.slider("Transacciones mensuales", 0, 60, 10)
    with s2:
        n_loan = st.number_input("Crédito vigente (MXN)", 0, 300_000, 20_000, step=1_000)
        n_age  = st.slider("Edad", 18, 80, 38)
    with s3:
        n_prod = st.slider("Número de productos", 1, 5, 2)
        n_inact= st.slider("Meses sin transacciones", 0, 24, 3)
    seg_submit = st.form_submit_button(
        "🔍 Asignar segmento", use_container_width=True, type="primary"
    )

if seg_submit:
    new_X = sc.transform([[n_bal, n_tx, n_loan, n_age, n_prod, n_inact]])
    cl_pred  = int(km.predict(new_X)[0])
    seg_pred = name_map.get(cl_pred, f"Segmento {cl_pred}")
    color_p  = SEGMENT_COLORS.get(seg_pred, COLORS["neutral"])

    # Distancias a todos los centroides
    dists = km.transform(new_X)[0]
    dist_df = pd.DataFrame({
        "Segmento": [name_map.get(i, f"Seg {i}") for i in range(k)],
        "Distancia al centroide": dists.round(3),
    }).sort_values("Distancia al centroide").reset_index(drop=True)

    sp1, sp2 = st.columns([1, 2])
    with sp1:
        st.markdown(f"""
        <div style='background:{hex_to_rgba(color_p,.10)};border:3px solid {color_p};
             border-radius:14px;padding:24px;text-align:center'>
          <div style='font-size:0.85rem;color:#64748b;font-weight:600'>SEGMENTO ASIGNADO</div>
          <div style='font-size:2.4rem;font-weight:900;color:{color_p};margin:8px 0'>{seg_pred}</div>
          <div style='font-size:12px;color:#64748b'>
            k={k} · distancia={dists[cl_pred]:.3f}
          </div>
        </div>""", unsafe_allow_html=True)

        # Barra de confianza (1 - normalización de distancias)
        min_d, max_d = dists.min(), dists.max()
        confidence = (1 - (dists[cl_pred] - min_d) / (max_d - min_d + 1e-9)) * 100
        st.metric("Confianza de asignación", f"{confidence:.1f}%",
                  "100% = más cercano al centroide")

    with sp2:
        # Distancias
        fig_dist = go.Figure(go.Bar(
            x=dist_df["Segmento"], y=dist_df["Distancia al centroide"],
            marker_color=[SEGMENT_COLORS.get(s, COLORS["neutral"])
                          for s in dist_df["Segmento"]],
            hovertemplate="%{x}: %{y:.3f}<extra></extra>",
        ))
        fig_dist.update_layout(
            title="Distancia del cliente a cada centroide (menor = más similar)",
            xaxis_title="Segmento", yaxis_title="Distancia euclidiana",
            paper_bgcolor="white", plot_bgcolor=COLORS["bg"],
            margin=dict(t=44, b=44, l=60, r=20),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        # Acciones recomendadas
        st.markdown(f"**Acciones recomendadas para segmento {seg_pred}:**")
        for icon, title, desc in SEG_ACTIONS.get(seg_pred, []):
            st.markdown(f"- **{icon} {title}** — {desc}")
