"""Dashboard de observabilidad del Agente Clínico SAPU (EP3 ISY0101).

Ejecutar desde ep3_observabilidad/:
    streamlit run dashboard/app.py
Lee logs/ejecuciones.jsonl (o la ruta de la variable EP3_EJECUCIONES).
"""
import os
import pathlib
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

RAIZ = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from evaluacion import cargar_registros, fallas_por_chequeo, metricas  # noqa: E402
from observabilidad import SLO_LATENCIA_S  # noqa: E402

st.set_page_config(page_title="Observabilidad · Agente SAPU", page_icon="🩺", layout="wide")

COLORES = {"v1 · T=0.0": "#4C78A8", "v1 · T=0.7": "#9ECAE9", "v2 · T=0.0": "#F58518"}


@st.cache_data
def datos(ruta: str) -> pd.DataFrame:
    return cargar_registros(ruta)


# 1) variable EP3_EJECUCIONES, 2) logs/ de una corrida local, 3) export enmascarado versionado en resultados/
_candidatas = [os.getenv("EP3_EJECUCIONES"), RAIZ / "logs" / "ejecuciones.jsonl", RAIZ / "resultados" / "ejecuciones_enmascaradas.jsonl"]
ruta = next((str(c) for c in _candidatas if c and pathlib.Path(c).exists()), str(_candidatas[1]))
if not pathlib.Path(ruta).exists():
    st.error(f"No se encontró {ruta}. Ejecuta primero el notebook (secciones 7 y 9).")
    st.stop()
df_all = datos(ruta)

# --- Filtros -----------------------------------------------------------------------
st.sidebar.header("Filtros")
versiones = st.sidebar.multiselect("Versión", sorted(df_all["version"].unique()), default=sorted(df_all["version"].unique()))
temps = st.sidebar.multiselect("Temperatura", sorted(df_all["temperatura"].unique()), default=sorted(df_all["temperatura"].unique()))
cats = st.sidebar.multiselect("Categoría", sorted(df_all["categoria"].unique()), default=sorted(df_all["categoria"].unique()))
escs = st.sidebar.multiselect("Escenario", sorted(df_all["escenario"].unique()), default=sorted(df_all["escenario"].unique()))
df = df_all[df_all["version"].isin(versiones) & df_all["temperatura"].isin(temps)
            & df_all["categoria"].isin(cats) & df_all["escenario"].isin(escs)]
st.sidebar.caption(f"{len(df)} de {len(df_all)} ejecuciones · SLO de latencia: {SLO_LATENCIA_S:g} s")

st.title("🩺 Observabilidad del Agente Clínico SAPU")
st.caption("EP3 ISY0101 · v1 = agente de la EP2 sin cambios · v2 = guardrail ampliado, vademécum tolerante a tipeo y prompt v2")
if df.empty:
    st.warning("Los filtros no dejan ejecuciones.")
    st.stop()

# --- 12 KPI ------------------------------------------------------------------------
m = metricas(df)
fmt = lambda x, d=1, suf="": "—" if pd.isna(x) else f"{x:.{d}f}{suf}"
kpis = [
    ("Precisión", fmt(m["precision_pct"], 1, " %"), "meta ≥ 90 %"),
    ("Efectividad guardrail", fmt(m["guardrail_pct"], 1, " %"), "meta 100 %"),
    ("Consistencia", fmt(m["consistencia_pct"], 1, " %"), "meta ≥ 95 %"),
    ("Tasa de error", fmt(m["error_pct"], 1, " %"), "meta ≤ 2 %"),
    ("Latencia p50", fmt(m["p50_s"], 2, " s"), "agente"),
    ("Latencia p95", fmt(m["p95_s"], 2, " s"), f"SLO ≤ {SLO_LATENCIA_S:g} s"),
    ("Sobre el SLO", fmt(m["sobre_slo_pct"], 1, " %"), "> 8 s"),
    ("Tiempo en el LLM", fmt(m["llm_pct_tiempo"], 1, " %"), "del total"),
    ("Llamadas LLM / consulta", fmt(m["llamadas_llm"], 2), "promedio"),
    ("Tokens / consulta", fmt(m["tokens_consulta"], 0), "entrada + salida"),
    ("US$ / 1.000 consultas", fmt(m["usd_1000"], 3), ", ".join(sorted(df["modelo"].dropna().unique())) if "modelo" in df else "precio de lista"),
    ("RAM máx.", fmt(m["ram_mb"], 0, " MB"), f"CPU {fmt(m['cpu_s'], 2, ' s')}/consulta"),
]
for fila in (kpis[:6], kpis[6:]):
    for col, (t, v, d) in zip(st.columns(6), fila):
        col.metric(t, v, d, delta_color="off", delta_arrow="off", border=True)

tabs = st.tabs(["Precisión", "Latencia", "Errores y eventos", "Explorador de trazas", "Comparación v1/v2"])

# --- Precisión ---------------------------------------------------------------------
with tabs[0]:
    c1, c2 = st.columns(2)
    pe = df.groupby(["escenario", "config"])["correcto"].mean().mul(100).reset_index()
    c1.plotly_chart(px.bar(pe, x="escenario", y="correcto", color="config", barmode="group", color_discrete_map=COLORES,
                           labels={"correcto": "Precisión (%)"}, title="Precisión por escenario de variabilidad"),
                    use_container_width=True)
    fc = fallas_por_chequeo(df)
    if not fc.empty:
        c2.plotly_chart(px.histogram(fc, x="chequeo", color="config", barmode="group", color_discrete_map=COLORES,
                                     title="Chequeos fallidos"), use_container_width=True)
    mat = df.groupby(["caso_id", "config"])["correcto"].mean().mul(100).unstack()
    st.plotly_chart(px.imshow(mat.T, text_auto=".0f", color_continuous_scale="RdYlGn", zmin=0, zmax=100,
                              aspect="auto", title="% de ejecuciones correctas por caso"), use_container_width=True)

# --- Latencia ----------------------------------------------------------------------
with tabs[1]:
    al = df[~df["bloqueado"]]
    c1, c2 = st.columns(2)
    fig = px.box(al, x="categoria", y="latencia_s", color="config", color_discrete_map=COLORES, points="all",
                 title="Latencia por categoría (s)")
    fig.add_hline(y=SLO_LATENCIA_S, line_dash="dash", line_color="red", annotation_text="SLO 8 s")
    c1.plotly_chart(fig, use_container_width=True)
    comp = al.groupby("config")[["t_llm_s", "t_tools_s", "latencia_s"]].sum()
    comp["otros_s"] = (comp["latencia_s"] - comp["t_llm_s"] - comp["t_tools_s"]).clip(lower=0)
    comp = comp.drop(columns="latencia_s").div(comp.drop(columns="latencia_s").sum(axis=1), axis=0).mul(100)
    c2.plotly_chart(px.bar(comp.reset_index().melt("config"), x="config", y="value", color="variable",
                           labels={"value": "% del tiempo"}, title="Desglose del tiempo: LLM, herramientas y otros"),
                    use_container_width=True)
    c3, c4 = st.columns(2)
    c3.plotly_chart(px.scatter(al, x="tokens_entrada", y="latencia_s", color="config", color_discrete_map=COLORES,
                               hover_data=["caso_id", "llamadas_llm"], title="Tokens de entrada vs latencia"),
                    use_container_width=True)
    rec = al.groupby("config")[["cpu_s", "ram_mb", "llamadas_llm", "costo_usd"]].mean()
    rec["costo_usd"] *= 1000
    c4.dataframe(rec.rename(columns={"cpu_s": "CPU s/consulta", "ram_mb": "RAM MB", "llamadas_llm": "Llamadas LLM",
                                     "costo_usd": "US$ por 1.000"}).style.format("{:.3f}"), use_container_width=True)

# --- Errores y eventos -------------------------------------------------------------
with tabs[2]:
    c1, c2 = st.columns(2)
    err = df[df["error"].notna()]
    c1.metric("Ejecuciones con error", len(err))
    c1.metric("Eventos de rate limit", int(df["rate_limits"].sum()))
    c1.metric("PII enviada al LLM", int(df["pii_enviada_al_llm"].sum()))
    if not err.empty:
        c2.plotly_chart(px.histogram(err, x="error", color="config", color_discrete_map=COLORES,
                                     title="Errores por tipo"), use_container_width=True)
    spans = pd.DataFrame([dict(s, config=r.config, caso_id=r.caso_id) for r in df.itertuples() for s in r.spans])
    if not spans.empty:
        t = spans[spans["tipo"] == "tool"]
        st.plotly_chart(px.histogram(t, x="nombre", color="estado", barmode="stack",
                                     title="Estado de las llamadas a herramientas"), use_container_width=True)
    st.subheader("Ejecuciones con eventos")
    ev = df[df["error"].notna() | df["pii_enviada_al_llm"] | ~df["correcto"]]
    st.dataframe(ev[["trace_id", "config", "repeticion", "caso_id", "escenario", "error", "pii_enviada_al_llm",
                     "herramientas", "consulta"]], use_container_width=True, hide_index=True)

# --- Explorador de trazas -------------------------------------------------------------
with tabs[3]:
    orden = df.sort_values("latencia_s", ascending=False)
    etiquetas = [f"{r.trace_id} · {r.config} · {r.caso_id} · {r.latencia_s:.2f} s" for r in orden.itertuples()]
    sel = st.selectbox("Traza (ordenadas de más lenta a más rápida)", etiquetas, index=0)
    r = orden.iloc[etiquetas.index(sel)]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latencia", f"{r.latencia_s:.2f} s")
    c2.metric("Llamadas LLM", int(r.llamadas_llm))
    c3.metric("Tokens", int(r.tokens_entrada + r.tokens_salida))
    c4.metric("Veredicto", "Correcto" if r.correcto else "Falla")
    if r.spans:
        sp = pd.DataFrame(r.spans)
        sp["etiqueta"] = [f"{i + 1}. {t}: {n}" for i, (t, n) in enumerate(zip(sp["tipo"], sp["nombre"]))]
        fig = go.Figure()
        for _, s in sp.iterrows():
            fig.add_bar(y=[s["etiqueta"]], x=[s["duracion_s"]], base=[s["inicio_s"]], orientation="h",
                        marker_color="#4C78A8" if s["tipo"] == "llm" else "#F58518",
                        hovertext=f"estado={s['estado']} tokens={s.get('tokens_entrada', 0)}+{s.get('tokens_salida', 0)}",
                        showlegend=False, text=f"{s['duracion_s']:.2f} s", textposition="outside")
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(title=f"Waterfall de la traza {r.trace_id} (azul = LLM, naranjo = herramienta)",
                          xaxis_title="segundos desde el inicio", height=120 + 45 * len(sp))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(sp[["etiqueta", "inicio_s", "duracion_s", "estado"] +
                        [c for c in ("tokens_entrada", "tokens_salida", "entrada", "salida") if c in sp]],
                     use_container_width=True, hide_index=True)
    st.markdown(f"**Consulta (enmascarada):** {r.consulta}")
    st.markdown(f"**Respuesta:** {r.respuesta}")
    st.json(r.checks if isinstance(r.checks, dict) else {}, expanded=False)

# --- Comparación ------------------------------------------------------------------
with tabs[4]:
    tabla = pd.DataFrame({cfg: metricas(g) for cfg, g in df.groupby("config")})
    st.dataframe(tabla.style.format("{:.2f}"), use_container_width=True)
    largo = tabla.loc[["precision_pct", "guardrail_pct", "consistencia_pct", "error_pct"]].reset_index().melt("index")
    st.plotly_chart(px.bar(largo, x="index", y="value", color="variable", barmode="group", color_discrete_map=COLORES,
                           labels={"index": "indicador", "value": "%", "variable": "configuración"},
                           title="Indicadores de calidad por configuración"), use_container_width=True)
