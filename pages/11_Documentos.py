"""
Módulo 11 — Análisis de Documentos Financieros
NLP + regex · extracción de entidades financieras · resumen ejecutivo automático ·
nube de palabras · análisis de texto propio
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.helpers import (
    SAMPLE_REPORT, FINANCIAL_PATTERNS,
    extract_financial_entities, extract_numbers,
    generate_executive_summary, word_frequency,
    hex_to_rgba, base_layout, COLORS,
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetric"]{background:#f0f9ff;border:1px solid #bae6fd;
  border-radius:12px;padding:1rem 1.25rem}
.section-title{font-size:1.05rem;font-weight:700;color:#0f172a;
  border-left:4px solid #0891b2;padding-left:12px;margin:1.5rem 0 1rem}
.info-box{background:#f0f9ff;border:1px solid #7dd3fc;border-radius:10px;
  padding:11px 15px;font-size:13px;color:#0c4a6e;margin:8px 0}
.entity-tag{display:inline-block;font-size:12px;font-weight:600;padding:4px 12px;
  border-radius:20px;margin:3px;background:#f0f9ff;color:#0c4a6e;border:1px solid #7dd3fc}
.summary-bullet{font-size:14px;padding:8px 14px;margin:5px 0;background:#f8fafc;
  border-left:3px solid #0891b2;border-radius:6px}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 📄 Análisis de Documentos Financieros")
st.markdown(
    "Motor de **NLP + expresiones regulares** que extrae KPIs financieros, genera "
    "un resumen ejecutivo automático y analiza la frecuencia de términos en "
    "reportes anuales, informes trimestrales o comunicados a inversionistas."
)
st.markdown("---")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Patrones de extracción", str(len(FINANCIAL_PATTERNS)))
c2.metric("Categorías de KPI",      "9", "Ingresos, EBITDA, EPS, etc.")
c3.metric("Motor",                  "Regex + NLP", "100% local")
c4.metric("Idioma",                 "Español", "Optimizado para reportes LATAM")

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — Entrada del documento
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">📥 Documento a Analizar</div>', unsafe_allow_html=True)

source = st.radio(
    "Fuente del texto", ["Usar reporte de ejemplo", "Pegar mi propio texto"],
    horizontal=True,
)

if source == "Usar reporte de ejemplo":
    text_input = SAMPLE_REPORT
    with st.expander("📄 Ver texto completo del reporte de ejemplo"):
        st.text(text_input)
else:
    text_input = st.text_area(
        "Pega el texto del documento financiero",
        height=250,
        placeholder="Pega aquí el contenido de un reporte anual, informe trimestral, "
                    "comunicado de prensa financiero, etc.",
    )

analyze_btn = st.button("🔍 Analizar documento", use_container_width=True, type="primary")

if analyze_btn and text_input.strip():
    st.session_state["doc_analyzed"] = True
    st.session_state["doc_text"] = text_input
elif analyze_btn and not text_input.strip():
    st.warning("Por favor ingresa o selecciona un texto para analizar.")

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — Resultados del análisis
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.get("doc_analyzed") and st.session_state.get("doc_text"):
    text = st.session_state["doc_text"]
    entities = extract_financial_entities(text)
    numbers  = extract_numbers(text)
    summary  = generate_executive_summary(text, entities)
    freq     = word_frequency(text, top_n=20)

    st.markdown("---")
    st.markdown('<div class="section-title">📋 Resumen Ejecutivo Automático</div>',
                unsafe_allow_html=True)

    if summary:
        for bullet in summary:
            st.markdown(f'<div class="summary-bullet">{bullet}</div>', unsafe_allow_html=True)
    else:
        st.info("No se detectaron KPIs estructurados en el texto. Prueba con un reporte que incluya cifras financieras explícitas (ingresos, EBITDA, EPS, etc.)")

    # ──────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-title">🏷️ Entidades Financieras Extraídas</div>',
                unsafe_allow_html=True)

    ec1, ec2 = st.columns([1, 1])
    with ec1:
        st.markdown("**Por categoría:**")
        if entities:
            for label, matches in entities.items():
                tags_html = "".join(f'<span class="entity-tag">{m}</span>' for m in matches)
                st.markdown(f"**{label}**<br>{tags_html}", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.info("No se encontraron entidades con los patrones definidos.")
    with ec2:
        cat_counts = pd.Series({k: len(v) for k, v in entities.items()})
        if len(cat_counts):
            fig_cat = go.Figure(go.Bar(
                x=cat_counts.values, y=cat_counts.index, orientation="h",
                marker_color=COLORS["primary"],
                hovertemplate="%{y}: %{x} coincidencias<extra></extra>",
            ))
            fig_cat.update_layout(
                title="Coincidencias por Categoría",
                xaxis_title="Número de menciones",
                height=320, paper_bgcolor="white", plot_bgcolor=COLORS["bg"],
                margin=dict(t=44, b=44, l=160, r=20),
            )
            st.plotly_chart(fig_cat, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-title">💲 Todas las Cifras Monetarias Detectadas</div>',
                unsafe_allow_html=True)

    if numbers:
        num_df = pd.DataFrame(numbers).rename(columns={"value": "Cifra", "context": "Contexto"})
        st.dataframe(num_df, use_container_width=True, hide_index=True, height=min(400, 45 + 35*len(num_df)))
    else:
        st.info("No se detectaron cifras monetarias con formato $X en el texto.")

    # ──────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-title">☁️ Frecuencia de Términos Clave</div>',
                unsafe_allow_html=True)

    if freq:
        freq_df = pd.DataFrame(freq, columns=["Palabra", "Frecuencia"])
        fig_freq = go.Figure(go.Bar(
            x=freq_df["Frecuencia"], y=freq_df["Palabra"], orientation="h",
            marker_color=COLORS["purple"],
            hovertemplate="%{y}: %{x} veces<extra></extra>",
        ))
        fig_freq.update_layout(
            title="Top 20 Palabras Más Frecuentes (excluyendo stopwords)",
            xaxis_title="Frecuencia",
            height=500, paper_bgcolor="white", plot_bgcolor=COLORS["bg"],
            yaxis=dict(autorange="reversed"),
            margin=dict(t=44, b=44, l=120, r=20),
        )
        st.plotly_chart(fig_freq, use_container_width=True)
    else:
        st.info("Texto insuficiente para generar análisis de frecuencia.")

    st.markdown(
        '<div class="info-box">💡 <strong>Cómo funciona:</strong> el motor usa expresiones '
        'regulares especializadas para detectar patrones financieros comunes en español '
        '(ingresos, EBITDA, EPS, morosidad, etc.), incluyendo tolerancia a siglas entre '
        'paréntesis como "(IMOR)" o "(ICAP)". En producción esto se sustituiría por modelos '
        'NER especializados como FinBERT para mayor cobertura y precisión.</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown("---")
    st.info("👆 Selecciona o pega un texto y presiona 'Analizar documento' para ver los resultados.")

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — Patrones de extracción (referencia técnica)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">🔧 Patrones de Extracción Configurados</div>',
            unsafe_allow_html=True)

with st.expander("Ver los patrones regex utilizados"):
    pattern_rows = []
    for label, pattern in FINANCIAL_PATTERNS.items():
        pattern_rows.append({
            "Categoría": label,
            "Patrón regex": pattern.pattern[:90] + ("…" if len(pattern.pattern) > 90 else ""),
        })
    st.dataframe(pd.DataFrame(pattern_rows), use_container_width=True, hide_index=True)
    st.markdown(
        '<div class="info-box">Todos los patrones toleran mayúsculas/minúsculas (flag '
        '<code>re.I</code>) y siglas regulatorias entre paréntesis, un caso muy común en '
        'reportes bancarios reales (IMOR, ICAP, ROAE, etc.).</div>',
        unsafe_allow_html=True,
    )
