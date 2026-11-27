"""Evaluación contra casos con respuesta esperada y cálculo de las métricas IE1/IE2."""
from __future__ import annotations

import json
import pathlib
import re
import time

import pandas as pd

from agente import proveedor_actual
from observabilidad import SLO_LATENCIA_S, LimiteDiarioAlcanzado, ejecutar_observado

CITA = re.compile(r"\bp\.?\s?\d+|p[áa]g(?:ina|\.)?\s*\d+", re.IGNORECASE)


def cargar_casos(ruta="casos_prueba.json") -> list[dict]:
    return json.loads(pathlib.Path(ruta).read_text(encoding="utf-8"))["casos"]


def evaluar(caso: dict, reg: dict) -> dict:
    """Chequeos del caso sobre el registro (texto sin enmascarar)."""
    esp = caso["esperado"]
    if esp.get("bloqueado"):
        # Casos con dato personal: lo único correcto es bloquear antes del LLM
        return {"guardrail": reg["bloqueado"]}
    if reg["bloqueado"]:
        return {"guardrail": False}
    resp = reg.get("respuesta") or ""
    usadas = set(reg["herramientas"])
    herr = (set(esp.get("requeridas", [])) <= usadas
            and (not esp.get("alguna") or bool(usadas & set(esp["alguna"])))
            and not (usadas & set(esp.get("prohibidas", []))))
    return {
        "guardrail": True,
        "herramientas": herr,
        "contenido": all(re.search(p, resp, re.IGNORECASE) for p in esp.get("contenido", [])),
        "cita": bool(CITA.search(resp)) if esp.get("cita") else True,
        "sin_alucinacion": not any(re.search(p, resp, re.IGNORECASE) for p in esp.get("prohibido", [])),
        "sin_error": reg["error"] is None,
    }


def correr(executor, casos, *, version, temperatura, repeticiones, guardrail, logger, ruta_jsonl,
           pausa_s: float = 1.0, verbose=True) -> list[dict]:
    regs = []
    n = len(casos) * repeticiones
    # Reanudación: si Colab se desconecta, no se repiten las ejecuciones ya registradas
    hechos = set()
    if pathlib.Path(ruta_jsonl).exists():
        for l in pathlib.Path(ruta_jsonl).read_text(encoding="utf-8").splitlines():
            if l.strip():
                d = json.loads(l)
                hechos.add((d["version"], d["temperatura"], d["repeticion"], d["caso_id"]))
    for rep in range(1, repeticiones + 1):
        for caso in casos:
            if (version, temperatura, rep, caso["id"]) in hechos:
                continue
            meta = {"temperatura": temperatura, "repeticion": rep, "caso_id": caso["id"],
                    "escenario": caso["escenario"], "categoria": caso["categoria"]}
            try:
                r = ejecutar_observado(executor, caso["consulta"], version=version, guardrail=guardrail, meta=meta,
                                       logger=logger, evaluar=lambda reg, c=caso: evaluar(c, reg),
                                       ruta_jsonl=ruta_jsonl)
            except LimiteDiarioAlcanzado as e:
                print(f"⏸️ Cuota diaria del proveedor agotada ({e}). Vuelve a ejecutar más tarde: se reanuda donde quedó.")
                raise
            regs.append(r)
            if verbose:
                print(f"[{len(regs):>3}/{n}] {version} T={temperatura} rep{rep} {caso['id']:<4} "
                      f"{'OK ' if r['correcto'] else 'FALLA'} {r['latencia_s']:5.2f}s "
                      f"tools={','.join(r['herramientas']) or '-'}{' ERR=' + r['error'] if r['error'] else ''}")
            if not r["bloqueado"]:
                # Ritmo según el límite de tokens por minuto del plan gratuito (75 % del tope, para dejar margen)
                tpm = proveedor_actual().get("tpm")
                tokens = r["tokens_entrada"] + r["tokens_salida"]
                time.sleep(max(pausa_s, 60 * tokens / (0.75 * tpm)) if tpm else pausa_s)
    return regs


def cargar_registros(ruta) -> pd.DataFrame:
    filas = [json.loads(l) for l in pathlib.Path(ruta).read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(filas)
    df["config"] = df["version"] + " · T=" + df["temperatura"].map(lambda t: f"{t:.1f}")
    df["herr_firma"] = df["herramientas"].map(lambda h: ",".join(sorted(set(h))))
    df["pii_caso"] = df["categoria"].eq("privacidad")
    return df


def metricas(df: pd.DataFrame) -> dict:
    """Métricas de una configuración (versión + temperatura)."""
    al_llm = df[~df["bloqueado"]]
    pii = df[df["pii_caso"]]
    cons = df.groupby("caso_id").agg(v=("correcto", "nunique"), h=("herr_firma", "nunique"))
    lat = al_llm["latencia_s"]
    return {
        "ejecuciones": len(df),
        "precision_pct": 100 * df["correcto"].mean(),
        "guardrail_pct": 100 * pii["bloqueado"].mean() if len(pii) else float("nan"),
        "consistencia_pct": 100 * ((cons["v"] == 1) & (cons["h"] == 1)).mean(),
        "error_pct": 100 * df["error"].notna().mean(),
        "rate_limits": int(df["rate_limits"].sum()),
        "max_iteraciones": int((df["error"] == "max_iteraciones").sum()),
        "p50_s": lat.quantile(0.50),
        "p95_s": lat.quantile(0.95),
        "sobre_slo_pct": 100 * (lat > SLO_LATENCIA_S).mean(),
        "llm_pct_tiempo": 100 * al_llm["t_llm_s"].sum() / max(lat.sum(), 1e-9),
        "tools_pct_tiempo": 100 * al_llm["t_tools_s"].sum() / max(lat.sum(), 1e-9),
        "llamadas_llm": al_llm["llamadas_llm"].mean(),
        "tokens_consulta": (df["tokens_entrada"] + df["tokens_salida"]).mean(),
        "usd_1000": 1000 * df["costo_usd"].mean(),
        "cpu_s": al_llm["cpu_s"].mean(),
        "ram_mb": df["ram_mb"].max(),
    }


def tabla_metricas(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({cfg: metricas(g) for cfg, g in df.groupby("config")})


def tabla_comparacion(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla 'comparacion' con el formato de la sección 4 del informe."""
    m = tabla_metricas(df)
    f = lambda x, d=1: f"{x:.{d}f}".replace(".", ",")
    filas = {
        "Precisión / guardrail": lambda c: f"{f(c['precision_pct'])} % / {f(c['guardrail_pct'])} %",
        "Consistencia de veredicto": lambda c: f"{f(c['consistencia_pct'])} %",
        "Tasa de error": lambda c: f"{f(c['error_pct'])} %",
        "Latencia p50 / p95 (s)": lambda c: f"{f(c['p50_s'], 2)} / {f(c['p95_s'], 2)}",
        "% del tiempo en el LLM": lambda c: f"{f(c['llm_pct_tiempo'])} %",
        "Tokens por consulta / US$ por 1.000": lambda c: f"{c['tokens_consulta']:.0f} / {f(c['usd_1000'], 3)}",
    }
    return pd.DataFrame({cfg: {k: fn(m[cfg]) for k, fn in filas.items()} for cfg in m.columns})


def precision_por_escenario(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["escenario", "config"])["correcto"].mean().mul(100).unstack("config"))


def fallas_por_chequeo(df: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for _, r in df.iterrows():
        for k, v in (r["checks"] or {}).items():
            if not v:
                filas.append({"config": r["config"], "caso_id": r["caso_id"], "escenario": r["escenario"], "chequeo": k})
    return pd.DataFrame(filas)
