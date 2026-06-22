"""
Módulo 3 — Curva de Rendimiento de Bonos
Análisis por escenario · spread histórico 10Y-2Y · calculadora de bono · sensibilidad
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.helpers import gen_yield_data, hex_to_rgba, base_layout, COLORS

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
</style>
""", unsafe_allow_html=True)

CURVE_COLORS = {
    "Normal":      COLORS["success"],
    "Invertida":   COLORS["danger"],
    "Plana":       COLORS["neutral"],
    "Pronunciada": COLORS["primary"],
    "Crisis 2008": COLORS["warning"],
}
CURVE_INFO = {
    "Normal":    {"icon":"🟢","badge":COLORS["success"],
                  "title":"Curva Normal — Pendiente Positiva",
                  "desc":"Los bonos de largo plazo ofrecen mayor rendimiento. Señal de expansión económica. El diferencial positivo 10Y-2Y favorece el margen financiero (NIM) de la banca.",
                  "strategy":"Posicionarse en duración media (5-7 años) para capturar el diferencial sin excesiva sensibilidad a tasas."},
    "Invertida": {"icon":"🔴","badge":COLORS["danger"],
                  "title":"Curva Invertida — Señal de Recesión",
                  "desc":"Los bonos de corto plazo rinden más que los largos. Predictor histórico de recesión en EE.UU. con 12-18 meses de anticipación.",
                  "strategy":"Incrementar duración para anticipar la bajada de tasas. Reducir exposición a crédito de alto rendimiento."},
    "Plana":     {"icon":"🟡","badge":COLORS["warning"],
                  "title":"Curva Plana — Fase de Transición",
                  "desc":"Rendimientos similares en todos los plazos. Señal de incertidumbre. El spread reducido comprime los márgenes bancarios (NIM).",
                  "strategy":"Posicionamiento defensivo: bonos de corto plazo y activos de alta calidad crediticia."},
    "Pronunciada":{"icon":"🔵","badge":COLORS["primary"],
                  "title":"Curva Pronunciada — Expansión Temprana",
                  "desc":"Alto diferencial entre tasas cortas y largas. Beneficia a bancos con captación corta y colocación larga.",
                  "strategy":"Invertir en activos de largo plazo antes de que la curva se aplane. Expandir cartera hipotecaria."},
    "Crisis 2008":{"icon":"⚠️","badge":COLORS["warning"],
                  "title":"Curva de Crisis — Tasas en Mínimos",
                  "desc":"Tasas cercanas a cero (ZIRP). Bancos centrales en emergencia. Curva ascendente pero nivel absoluto muy bajo.",
                  "strategy":"Activos reales, renta variable con dividendo. La renta fija ofrece retornos reales negativos."},
}
MAT_LABELS = ["3M","6M","1A","2A","3A","5A","7A","10A","20A","30A"]

@st.cache_data
def load():
    return gen_yield_data()

data    = load()
mats    = data["maturities"]
curves  = data["curves"]
history = data["history"]

st.markdown("# 📈 Curva de Rendimiento de Bonos")
st.markdown("Análisis interactivo de la **curva yield** por escenario de política monetaria. "
            "Compara escenarios, monitorea el spread 10Y-2Y histórico, calcula precio y duración de bonos.")
st.markdown("---")

cur = curves["Normal"]
sp  = float(cur[7] - cur[3])
c1,c2,c3,c4 = st.columns(4)
c1.metric("Tasa 2 años",   f"{cur[3]:.2f}%")
c2.metric("Tasa 10 años",  f"{cur[7]:.2f}%")
c3.metric("Spread 10Y-2Y", f"{sp:+.2f} pp", "Normal ✓" if sp>0 else "⚠️ Invertida")
c4.metric("Tasa 30 años",  f"{cur[-1]:.2f}%")

# ──────────────────────────────────────────────────────────────────────────────
# S1 — Curvas por escenario
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">🌐 Comparativa de Curvas por Escenario</div>',unsafe_allow_html=True)

sel = st.multiselect("Selecciona escenarios", list(curves.keys()), default=["Normal","Invertida"])
if not sel:
    st.warning("Selecciona al menos un escenario.")
    st.stop()

fig_c = go.Figure()
for name in sel:
    color = CURVE_COLORS.get(name, COLORS["neutral"])
    fig_c.add_trace(go.Scatter(
        x=mats, y=curves[name], mode="lines+markers", name=name,
        line=dict(color=color, width=2.5),
        marker=dict(size=7, line=dict(color="white",width=1.2)),
        hovertemplate=f"<b>{name}</b><br>Plazo: %{{x:.2f}}a<br>Rend: %{{y:.2f}}%<extra></extra>",
    ))
fig_c.update_layout(
    title="Curva de Rendimiento — Comparativa por Escenario",
    xaxis=dict(title="Plazo (años)", tickvals=mats.tolist(), ticktext=MAT_LABELS,
               gridcolor=COLORS["grid"], zeroline=False),
    yaxis=dict(title="Rendimiento (%)", gridcolor=COLORS["grid"], zeroline=False),
    legend=dict(orientation="h", y=-0.18, x=0),
    **{k:v for k,v in base_layout().items() if k not in ("xaxis","yaxis")},
)
st.plotly_chart(fig_c, use_container_width=True)

info = CURVE_INFO.get(sel[0], {})
if info:
    bc = info["badge"]
    st.markdown(f"""
    <div style='background:{hex_to_rgba(bc,.07)};border-left:4px solid {bc};
         border-radius:8px;padding:14px 18px'>
      <div style='font-size:14px;font-weight:700;color:{bc};margin-bottom:6px'>
        {info["icon"]} {info["title"]}
      </div>
      <div style='font-size:13px;color:#0f172a;margin-bottom:8px'>{info["desc"]}</div>
      <div style='font-size:13px;color:{bc};font-weight:600'>💼 Estrategia: {info["strategy"]}</div>
    </div>""", unsafe_allow_html=True)

with st.expander("📋 Tabla de rendimientos por plazo y escenario"):
    tbl = pd.DataFrame({n: curves[n] for n in sel}, index=MAT_LABELS)
    tbl.index.name = "Plazo"
    st.dataframe(tbl.style.format("{:.2f}%").background_gradient(cmap="RdYlGn",axis=None),
                 use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# S2 — Spread histórico 10Y-2Y
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">📉 Evolución Histórica del Spread 10Y-2Y</div>',unsafe_allow_html=True)
st.markdown('<div class="info-box">El spread 10Y-2Y es el indicador de inversión de curva más monitoreado. Cuando cruza cero (negativo), históricamente ha precedido una recesión en EE.UU. con 12-18 meses de anticipación.</div>',unsafe_allow_html=True)

pos_s = history["spread"].clip(lower=0)
neg_s = history["spread"].clip(upper=0)

fig_h = go.Figure()
fig_h.add_trace(go.Scatter(x=history["date"],y=pos_s,fill="tozeroy",mode="none",
    name="Spread ≥ 0 (normal)",fillcolor=hex_to_rgba(COLORS["success"],0.22)))
fig_h.add_trace(go.Scatter(x=history["date"],y=neg_s,fill="tozeroy",mode="none",
    name="Spread < 0 (invertida)",fillcolor=hex_to_rgba(COLORS["danger"],0.22)))
fig_h.add_trace(go.Scatter(x=history["date"],y=history["spread"],mode="lines",
    name="Spread 10Y-2Y",line=dict(color=COLORS["primary"],width=2),
    hovertemplate="%{x|%b %Y}<br>Spread: %{y:+.2f} pp<extra></extra>"))
fig_h.add_hline(y=0,line_dash="dash",line_color="#0f172a",
    annotation_text="Umbral de inversión  ",annotation_position="top right")
fig_h.update_layout(
    title="Spread 10Y-2Y — Serie histórica sintética (últimos 36 meses)",
    xaxis=dict(title="Fecha",gridcolor=COLORS["grid"],zeroline=False),
    yaxis=dict(title="Spread (pp)",gridcolor=COLORS["grid"],zeroline=False),
    legend=dict(orientation="h",y=-0.18),
    **{k:v for k,v in base_layout().items() if k not in ("xaxis","yaxis")},
)
st.plotly_chart(fig_h, use_container_width=True)

last_sp = float(history["spread"].iloc[-1])
min_sp  = float(history["spread"].min())
inv_mo  = int((history["spread"] < 0).sum())
h1,h2,h3,h4 = st.columns(4)
h1.metric("Spread actual",          f"{last_sp:+.2f} pp", "Normal" if last_sp>0 else "⚠️ Invertida")
h2.metric("Mínimo 36 meses",        f"{min_sp:+.2f} pp")
h3.metric("Meses con inversión",    f"{inv_mo}", f"{inv_mo/36:.0%} del período")
h4.metric("Spread promedio",        f"{history['spread'].mean():+.2f} pp")

with st.expander("📊 Ver tasas 10Y y 2Y por separado"):
    fig_r = go.Figure()
    fig_r.add_trace(go.Scatter(x=history["date"],y=history["y10"],mode="lines",
        name="10 años",line=dict(color=COLORS["primary"],width=2)))
    fig_r.add_trace(go.Scatter(x=history["date"],y=history["y2"],mode="lines",
        name="2 años",line=dict(color=COLORS["danger"],width=2)))
    fig_r.update_layout(
        title="Tasas 10Y y 2Y — Serie histórica",
        xaxis=dict(title="Fecha",gridcolor=COLORS["grid"],zeroline=False),
        yaxis=dict(title="Tasa (%)",gridcolor=COLORS["grid"],zeroline=False),
        legend=dict(orientation="h",y=-0.18),
        **{k:v for k,v in base_layout().items() if k not in ("xaxis","yaxis")},
    )
    st.plotly_chart(fig_r, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# S3 — Calculadora de bono
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">🧮 Calculadora de Precio y Duración de Bono</div>',unsafe_allow_html=True)
st.markdown("Calcula precio limpio, duración de Macaulay, duración modificada y convexidad de un bono cupón fijo usando la tasa spot de la curva seleccionada.")

with st.form("bond_form"):
    b1,b2,b3 = st.columns(3)
    with b1:
        face_val   = st.number_input("Valor nominal ($)",1_000,10_000_000,100_000,step=10_000)
        coupon_pct = st.number_input("Tasa cupón anual (%)",0.0,25.0,8.0,step=0.25)
    with b2:
        mat_y = st.selectbox("Plazo del bono",
            [0.25,0.5,1,2,3,5,7,10,20,30], index=7,
            format_func=lambda x: f"{MAT_LABELS[[0.25,0.5,1,2,3,5,7,10,20,30].index(x)]} ({x:.2f}a)")
        freq  = st.selectbox("Frecuencia de cupón",[1,2,4,12],index=1,
            format_func=lambda x:{1:"Anual",2:"Semestral",4:"Trimestral",12:"Mensual"}[x])
    with b3:
        curve_sel = st.selectbox("Escenario de tasa (YTM)",list(curves.keys()))
        show_sens = st.checkbox("Mostrar sensibilidad ±3pp",value=True)
    bond_ok = st.form_submit_button("💰 Calcular precio del bono",use_container_width=True,type="primary")

if bond_ok:
    ytm = float(np.interp(mat_y, mats, curves[curve_sel])) / 100
    n   = max(1, int(round(mat_y * freq)))
    r   = ytm / freq
    cf  = coupon_pct / 100 * face_val / freq
    ts  = np.arange(1, n + 1, dtype=float)

    pv_cfs  = cf / (1 + r) ** ts
    pv_face = face_val / (1 + r) ** n
    price   = float(pv_cfs.sum() + pv_face)

    # Duración Macaulay
    all_pv  = np.append(pv_cfs, pv_face)
    all_t   = np.append(ts / freq, float(mat_y))
    mac_dur = float((all_pv * all_t).sum() / price)
    mod_dur = mac_dur / (1 + r)

    # Convexidad
    convexity = float(
        (all_pv * all_t * (all_t + 1/freq)).sum()
        / (price * (1 + ytm)**2)
    )

    premium = price / face_val - 1
    dp_up   = (-mod_dur*0.01  + 0.5*convexity*0.01**2)  * price
    dp_down = (-mod_dur*(-0.01)+ 0.5*convexity*0.01**2) * price

    pr1,pr2,pr3,pr4 = st.columns(4)
    pr1.metric("Precio del bono",     f"${price:,.2f}",    f"{'Prima' if premium>0 else 'Descuento'} {premium:+.2%}")
    pr2.metric("YTM de mercado",      f"{ytm*100:.2f}%",   f"Curva {curve_sel} — {mat_y:.1f}a")
    pr3.metric("Duración Modificada", f"{mod_dur:.3f} a",  "Sensibilidad Δ1pp")
    pr4.metric("Convexidad",          f"{convexity:.2f}",  "Ajuste cuadrático")

    e1,e2 = st.columns(2)
    e1.metric("Δ Precio si tasa ↑ 1pp", f"${dp_up:,.2f}",   f"{dp_up/price:.2%}")
    e2.metric("Δ Precio si tasa ↓ 1pp", f"${dp_down:,.2f}", f"{dp_down/price:.2%}")

    st.markdown(f'<div class="info-box">📐 Con duración modificada <strong>{mod_dur:.2f}</strong>, un incremento de 1pp en tasas reduce el precio aprox. <strong>${abs(dp_up):,.0f}</strong>. La convexidad ({convexity:.2f}) reduce la pérdida real — cuanto mayor la convexidad, más favorable para el tenedor.</div>',unsafe_allow_html=True)

    if show_sens:
        dr_range  = np.linspace(-0.03, 0.03, 61)
        prices_real = []
        prices_lin  = []
        prices_quad = []
        for dr in dr_range:
            r2 = max(ytm + dr, 0.001) / freq
            prices_real.append(float((cf/(1+r2)**ts).sum() + face_val/(1+r2)**n))
            prices_lin.append(price + (-mod_dur*dr)*price)
            prices_quad.append(price + (-mod_dur*dr + 0.5*convexity*dr**2)*price)

        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=dr_range*100,y=prices_real,mode="lines",
            name="Precio real",line=dict(color=COLORS["primary"],width=2.5),
            hovertemplate="Δ tasa: %{x:+.2f}pp<br>Precio: $%{y:,.2f}<extra></extra>"))
        fig_s.add_trace(go.Scatter(x=dr_range*100,y=prices_lin,mode="lines",
            name="Aprox. lineal (solo duración)",line=dict(color=COLORS["danger"],width=1.5,dash="dash")))
        fig_s.add_trace(go.Scatter(x=dr_range*100,y=prices_quad,mode="lines",
            name="Aprox. cuadrática (dur.+convexidad)",line=dict(color=COLORS["success"],width=1.5,dash="dot")))
        fig_s.add_vline(x=0,line_dash="dash",line_color="#0f172a",annotation_text="Tasa base")
        fig_s.add_hline(y=face_val,line_dash="dot",line_color=COLORS["neutral"],annotation_text="Valor nominal")
        fig_s.update_layout(
            title="Sensibilidad del Precio a Cambios en Tasa ±3pp",
            xaxis=dict(title="Cambio en tasa (pp)",gridcolor=COLORS["grid"],zeroline=False),
            yaxis=dict(title="Precio ($)",gridcolor=COLORS["grid"],zeroline=False),
            legend=dict(orientation="h",y=-0.18),
            **{k:v for k,v in base_layout().items() if k not in ("xaxis","yaxis")},
        )
        st.plotly_chart(fig_s, use_container_width=True)
        st.markdown('<div class="info-box">La diferencia entre la línea azul (precio real) y la roja (duración lineal) es el efecto de la <strong>convexidad</strong>. El bono siempre vale más de lo que predice la duración sola — la convexidad es siempre positiva para bonos estándar.</div>',unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# S4 — Diferencial entre escenarios
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">⚖️ Diferencial entre Escenarios por Plazo</div>',unsafe_allow_html=True)

dc1,dc2 = st.columns(2)
with dc1:
    sel_a = st.selectbox("Escenario A",list(curves.keys()),index=0,key="ca")
with dc2:
    sel_b = st.selectbox("Escenario B",list(curves.keys()),index=1,key="cb")

diff = np.array(curves[sel_a]) - np.array(curves[sel_b])
bar_colors = [COLORS["success"] if v>=0 else COLORS["danger"] for v in diff]

fig_d = go.Figure(go.Bar(
    x=MAT_LABELS, y=diff, marker_color=bar_colors,
    text=[f"{v:+.2f}" for v in diff], textposition="outside",
    hovertemplate="%{x}: %{y:+.2f} pp<extra></extra>",
))
fig_d.add_hline(y=0,line_dash="dash",line_color="#0f172a")
fig_d.update_layout(
    title=f"Diferencial por Plazo: {sel_a} − {sel_b}",
    xaxis=dict(title="Plazo",gridcolor=COLORS["grid"],zeroline=False),
    yaxis=dict(title="Diferencial (pp)",gridcolor=COLORS["grid"],zeroline=False),
    **{k:v for k,v in base_layout().items() if k not in ("xaxis","yaxis")},
)
st.plotly_chart(fig_d, use_container_width=True)

st.markdown("**Todos los escenarios — tabla comparativa:**")
all_df = pd.DataFrame(curves, index=MAT_LABELS)
all_df.index.name = "Plazo"
st.dataframe(
    all_df.style.format("{:.2f}%")
    .background_gradient(cmap="RdYlGn",axis=None)
    .highlight_max(axis=1,color="#dcfce7")
    .highlight_min(axis=1,color="#fef2f2"),
    use_container_width=True,
)
