"""
Módulo 5 — Sentimiento Financiero
Léxico financiero en español (palabra clave + reglas de contexto) ·
análisis individual · monitor de tendencias · nube de palabras
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from utils.helpers import (
    NEWS_HEADLINES, analyze_financial_sentiment, analyze_news_batch,
    hex_to_rgba, base_layout, COLORS,
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
.warn-box{background:#fffbeb;border:1px solid #fcd34d;border-radius:10px;
  padding:11px 15px;font-size:13px;color:#78350f;margin:8px 0}
</style>
""", unsafe_allow_html=True)

SENT_COLORS = {"Positivo": COLORS["success"], "Negativo": COLORS["danger"], "Neutral": COLORS["neutral"]}

# ── Análisis del dataset base ─────────────────────────────────────────────────
@st.cache_data
def load():
    headlines = [h for h, _ in NEWS_HEADLINES]
    expected  = [e for _, e in NEWS_HEADLINES]
    results   = analyze_news_batch(headlines)
    df = pd.DataFrame(results)
    df["esperado"] = [e.capitalize() for e in expected]
    df["acierto"]  = df["label"] == df["esperado"]
    return df

df = load()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 📰 Sentimiento Financiero")
st.markdown(
    "Motor de análisis de sentimiento con **léxico financiero en español** "
    "(palabra clave ponderada + reglas de contexto). Clasifica noticias del "
    "mercado como positivas, negativas o neutrales."
)

st.markdown(
    '<div class="warn-box">⚠️ <strong>Nota técnica:</strong> librerías como VADER '
    'o TextBlob están entrenadas en inglés y devuelven 0.0 (neutral) para '
    'cualquier texto en español — se verificó empíricamente antes de construir '
    'este módulo. Por eso se implementó un léxico financiero específico de dominio '
    'con precisión validada empíricamente sobre el dataset de muestra (ver métrica abajo).</div>',
    unsafe_allow_html=True,
)
st.markdown("---")

acc = df["acierto"].mean()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Precisión del léxico",   f"{acc:.0%}",  "vs etiquetas esperadas")
c2.metric("Noticias analizadas",    f"{len(df)}")
c3.metric("% Positivas",            f"{(df['label']=='Positivo').mean():.0%}")
c4.metric("% Negativas",            f"{(df['label']=='Negativo').mean():.0%}")

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — Analizar texto propio
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">✍️ Analizar Titular o Texto Propio</div>', unsafe_allow_html=True)

user_text = st.text_area(
    "Escribe o pega un titular financiero en español",
    "El Banco de México reporta crecimiento sólido en reservas internacionales",
    height=80,
)
analyze_btn = st.button("🔍 Analizar sentimiento", type="primary", use_container_width=True)

if analyze_btn and user_text.strip():
    result = analyze_financial_sentiment(user_text)
    color  = SENT_COLORS.get(result["label"], COLORS["neutral"])

    r1, r2 = st.columns([1, 2])
    with r1:
        st.markdown(f"""
        <div style='background:{hex_to_rgba(color,.08)};border:2px solid {color};
             border-radius:14px;padding:22px;text-align:center'>
          <div style='font-size:0.8rem;color:#64748b;font-weight:600'>SENTIMIENTO DETECTADO</div>
          <div style='font-size:1.8rem;font-weight:800;color:{color};margin:8px 0'>
            {result["label"]}
          </div>
          <div style='font-size:0.85rem;color:#64748b'>Score: {result["score"]:+.2f} (rango −1 a +1)</div>
        </div>""", unsafe_allow_html=True)
    with r2:
        st.markdown("**Palabras clave detectadas:**")
        if result["pos_words"]:
            st.markdown(
                "🟢 **Positivas:** " +
                ", ".join(f"`{w}`" for w in result["pos_words"])
            )
        if result["neg_words"]:
            st.markdown(
                "🔴 **Negativas:** " +
                ", ".join(f"`{w}`" for w in result["neg_words"])
            )
        if not result["pos_words"] and not result["neg_words"]:
            st.markdown("_No se detectaron palabras clave financieras — texto clasificado como Neutral._")

        # Barra de score visual
        fig_score = go.Figure(go.Bar(
            x=[result["score"]], y=["Score"], orientation="h",
            marker_color=color,
            hovertemplate="%{x:+.2f}<extra></extra>",
        ))
        fig_score.update_layout(
            xaxis=dict(range=[-1, 1], title="Negativo ← → Positivo",
                      gridcolor=COLORS["grid"], zeroline=True, zerolinecolor="#0f172a"),
            yaxis=dict(visible=False),
            height=120, paper_bgcolor="white", plot_bgcolor=COLORS["bg"],
            margin=dict(t=10, b=30, l=20, r=20),
        )
        st.plotly_chart(fig_score, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — Monitor de noticias del mercado
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">📊 Monitor de Noticias del Mercado</div>', unsafe_allow_html=True)

ca, cb = st.columns([1, 1])
with ca:
    sent_counts = df["label"].value_counts().reindex(["Positivo", "Neutral", "Negativo"], fill_value=0)
    fig_pie = go.Figure(go.Pie(
        labels=sent_counts.index, values=sent_counts.values,
        marker_colors=[SENT_COLORS[l] for l in sent_counts.index],
        hole=0.45, textinfo="label+percent",
    ))
    fig_pie.update_layout(
        title="Distribución de Sentimiento — Noticias del Mercado",
        height=360, paper_bgcolor="white",
        margin=dict(t=44, b=20, l=20, r=20),
    )
    st.plotly_chart(fig_pie, use_container_width=True)
with cb:
    fig_score = go.Figure(go.Bar(
        x=df["score"], y=[t[:40] + "…" if len(t) > 40 else t for t in df["text"]],
        orientation="h",
        marker_color=[SENT_COLORS[l] for l in df["label"]],
        hovertemplate="%{y}<br>Score: %{x:+.2f}<extra></extra>",
    ))
    fig_score.update_layout(
        title="Score por Titular",
        xaxis_title="Score (−1 a +1)",
        height=360, paper_bgcolor="white", plot_bgcolor=COLORS["bg"],
        margin=dict(t=44, b=44, l=260, r=20),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_score, use_container_width=True)

# Validación del modelo (comparación con etiqueta esperada)
with st.expander("🧪 Ver validación del léxico vs etiquetas esperadas"):
    show = df[["text", "label", "esperado", "score", "acierto"]].rename(columns={
        "text": "Titular", "label": "Predicho", "esperado": "Esperado",
        "score": "Score", "acierto": "Acierto",
    })
    st.dataframe(
        show.style
        .format({"Score": "{:+.2f}"})
        .apply(lambda row: ['background-color: #f0fdf4' if row["Acierto"]
                            else 'background-color: #fef2f2' for _ in row], axis=1),
        use_container_width=True, height=420,
    )
    st.markdown(f'<div class="info-box">El léxico clasificó correctamente <strong>{df["acierto"].sum()} de {len(df)}</strong> titulares ({acc:.0%} de precisión). Los errores suelen ocurrir en frases con ironía, doble negación o contexto que requiere conocimiento externo (ej. "nivel mínimo" puede ser bueno o malo según qué se mida).</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — Nube de palabras por sentimiento
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">☁️ Palabras Clave por Sentimiento</div>', unsafe_allow_html=True)

sent_filter = st.radio("Ver palabras clave de:", ["Positivo", "Negativo"], horizontal=True)
word_col = "pos_words" if sent_filter == "Positivo" else "neg_words"
all_words = [w for words in df[word_col] for w in words]

if all_words:
    word_freq = pd.Series(all_words).value_counts().reset_index()
    word_freq.columns = ["palabra", "frecuencia"]
    color_w = SENT_COLORS[sent_filter]
    fig_words = go.Figure(go.Bar(
        x=word_freq["frecuencia"], y=word_freq["palabra"],
        orientation="h", marker_color=color_w,
        hovertemplate="%{y}: %{x}<extra></extra>",
    ))
    fig_words.update_layout(
        title=f"Frecuencia de palabras clave — {sent_filter}",
        xaxis_title="Frecuencia", height=max(300, len(word_freq) * 28),
        paper_bgcolor="white", plot_bgcolor=COLORS["bg"],
        margin=dict(t=44, b=44, l=120, r=20),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_words, use_container_width=True)
else:
    st.info(f"No se detectaron palabras clave {sent_filter.lower()}s en el dataset actual.")

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — Tendencia simulada (serie temporal)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">📈 Tendencia de Sentimiento (Simulada)</div>', unsafe_allow_html=True)
st.markdown("Simulación de cómo evolucionaría el sentimiento promedio del mercado si las noticias llegaran en orden cronológico.")

# Usar índice como proxy de tiempo + media móvil
df_trend = df.reset_index().rename(columns={"index": "orden"})
df_trend["score_ma3"] = df_trend["score"].rolling(3, min_periods=1).mean()

fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(
    x=df_trend["orden"], y=df_trend["score"], mode="markers",
    marker=dict(color=[SENT_COLORS[l] for l in df_trend["label"]], size=9),
    name="Score individual",
    hovertemplate="Score: %{y:+.2f}<extra></extra>",
))
fig_trend.add_trace(go.Scatter(
    x=df_trend["orden"], y=df_trend["score_ma3"], mode="lines",
    line=dict(color=COLORS["primary"], width=2.5, dash="dash"),
    name="Media móvil (3 noticias)",
))
fig_trend.add_hline(y=0, line_dash="dot", line_color="#0f172a")
fig_trend.update_layout(
    title="Evolución del Sentimiento del Mercado",
    xaxis_title="Orden de publicación", yaxis_title="Score de sentimiento",
    legend=dict(orientation="h", y=-0.18),
    **base_layout(),
)
st.plotly_chart(fig_trend, use_container_width=True)

avg_recent = df_trend["score"].tail(5).mean()
trend_label = "🟢 Optimista" if avg_recent > 0.1 else "🔴 Pesimista" if avg_recent < -0.1 else "🟡 Neutral"
st.markdown(
    f'<div class="info-box">📍 <strong>Sentimiento reciente (últimas 5 noticias): {trend_label}</strong> '
    f'(score promedio {avg_recent:+.2f}). En un sistema real, este indicador se actualizaría '
    f'continuamente con feeds de noticias en vivo (NewsAPI, RSS de Reuters/Bloomberg).</div>',
    unsafe_allow_html=True,
)
