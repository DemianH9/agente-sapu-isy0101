"""Calcula, desde logs/ejecuciones.jsonl, los valores que cita el informe (tablas de las secciones 4 a 6).

Uso (desde ep3_observabilidad/):  python valores_informe.py logs/ejecuciones.jsonl resultados
Escribe resultados/valores_informe.json y figuras/fig1_precision_escenario.png e imprime el JSON.
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import evaluacion as ev

ruta, out = sys.argv[1], pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
df = ev.cargar_registros(ruta)
CFG = {"v1_0": "v1 · T=0.0", "v1_7": "v1 · T=0.7", "v2_0": "v2 · T=0.0"}
g = {k: df[df.config == c] for k, c in CFG.items()}
m = {k: ev.metricas(x) for k, x in g.items()}
comp = ev.tabla_comparacion(df)
f = lambda x, d=1: f"{x:.{d}f}".replace(".", ",")

v1 = df[df.version == "v1"]
c05 = v1[v1.caso_id == "C05"]
sin_dato = sum(any(s["nombre"] == "consultar_vademecum" and s["estado"] == "sin_dato" for s in r.spans) for r in c05.itertuples())
c06 = v1[v1.caso_id == "C06"]
calc_c06 = sum("calcular_dosis" in r.herramientas for r in c06.itertuples())
dosis = g["v1_0"][(g["v1_0"].categoria == "dosis") & ~g["v1_0"].bloqueado]
pe = ev.precision_por_escenario(df)
lenta = df.sort_values("latencia_s", ascending=False).iloc[0]

CONSULTAS_TURNO, TURNOS_MES = 50, 90
usd = m["v2_0"]["usd_1000"]
valores = {
    "comparacion": {k: dict(zip(["prec_guard", "consistencia", "error", "latencia", "llm", "tokens_costo"],
                                comp[CFG[k]].tolist())) for k in CFG},
    "hallazgos": {
        "H2": [sin_dato, len(c05)],
        "H3_evidencia": [f(m["v1_0"]["llm_pct_tiempo"]), f(dosis.llamadas_llm.mean(), 1)],
        "H3_impacto": [f(dosis.latencia_s.quantile(0.95), 2)],
        "H4": [calc_c06, len(c06)],
        "H5": [f(m["v1_0"]["consistencia_pct"]), f(m["v1_7"]["consistencia_pct"]),
               int(df.rate_limits.sum()), int((df.error == "max_iteraciones").sum())],
    },
    "recomendaciones": {
        "R1": [f(m["v1_0"]["guardrail_pct"], 0), f(m["v2_0"]["guardrail_pct"], 0)],
        "R2": [f(pe.loc["ruido", CFG["v1_0"]], 0), f(pe.loc["ruido", CFG["v2_0"]], 0)],
        "R3": [f(pe.loc["dato_faltante", CFG["v1_0"]], 0), f(pe.loc["dato_faltante", CFG["v2_0"]], 0)],
    },
    "costo": [f(usd, 3), CONSULTAS_TURNO, f(usd / 1000 * CONSULTAS_TURNO * TURNOS_MES, 2)],
    "extra": {
        "metricas": {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in mm.items()} for k, mm in m.items()},
        "precision_por_escenario": pe.round(1).to_dict(),
        "traza_mas_lenta": {"trace_id": lenta.trace_id, "config": lenta.config, "caso": lenta.caso_id,
                            "latencia_s": round(lenta.latencia_s, 2), "spans": len(lenta.spans)},
        "fallas": ev.fallas_por_chequeo(df).groupby(["config", "caso_id", "chequeo"]).size().reset_index().to_dict("records"),
        "pii_enviada_v1": int(v1.pii_enviada_al_llm.sum()),
        "casos_pii_enviada": sorted(v1[v1.pii_enviada_al_llm].caso_id.unique().tolist()),
        "n_por_config": df.groupby("config").size().to_dict(),
        "prompt_injection": [pe.loc["prompt_injection", CFG["v1_0"]], pe.loc["prompt_injection", CFG["v2_0"]]],
    },
}
(out / "valores_informe.json").write_text(json.dumps(valores, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

# Figura 3 (misma que la sección 11 del notebook)
orden = ["base", "parafraseo", "ruido", "limite", "dato_faltante", "farmaco_ausente", "rut_formatos", "nombre_sin_rut", "prompt_injection"]
etiquetas = ["base", "parafraseo", "ruido\n(tipeo)", "límite", "dato\nfaltante", "fármaco\nausente", "formatos\nde RUT", "nombre\nsin RUT", "prompt\ninjection"]
p = pe.reindex(orden).rename(index=dict(zip(orden, etiquetas)))[[CFG["v1_0"], CFG["v2_0"]]]
ax = p.plot.bar(figsize=(9, 3.1), color=["#4C78A8", "#F58518"], width=0.75)
ax.set_ylabel("Precisión (%)"); ax.set_xlabel(""); ax.set_ylim(0, 105); ax.axhline(90, ls="--", c="gray", lw=1)
ax.set_title("Precisión por escenario, v1 vs v2 (T = 0)"); ax.legend(title=None, loc="lower left", fontsize=9)
plt.xticks(rotation=0, fontsize=8); plt.tight_layout(); pathlib.Path("figuras").mkdir(exist_ok=True); plt.savefig("figuras/fig1_precision_escenario.png", dpi=200)
print("@@@" + json.dumps(valores, ensure_ascii=False, default=str) + "@@@")
