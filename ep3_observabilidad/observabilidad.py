"""Observabilidad del agente SAPU: trazas por span, métricas por ejecución y logs enmascarados.

- TrazaHandler: callback de LangChain que registra un span por cada llamada al LLM y a cada
  herramienta (duración, tokens, estado), sin modificar la lógica del agente.
- ejecutar_observado(): ejecuta una consulta, aplica el guardrail de la versión, mide latencia,
  CPU, RAM y costo, reintenta ante rate limit y escribe logs/ejecuciones.jsonl y logs/agente.log
  después de enmascarar los datos personales.
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import time
import uuid
from datetime import datetime, timezone

from langchain_core.callbacks import BaseCallbackHandler

from agente import contiene_pii_v2, enmascarar

from agente import proveedor_actual


def precios() -> tuple[float, float]:
    """US$ por token (entrada, salida) del proveedor configurado (costo equivalente de lista)."""
    ent, sal = proveedor_actual()["precio"]
    return ent / 1_000_000, sal / 1_000_000

SLO_LATENCIA_S = 8.0

LOG_DIR = pathlib.Path(os.getenv("EP3_LOG_DIR", "logs"))


def configurar_logging(log_dir: pathlib.Path = LOG_DIR) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("agente_sapu")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_dir / "agente.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.WARNING)
    sh.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def _modelo(kwargs) -> str:
    ip = kwargs.get("invocation_params") or {}
    return ip.get("model") or ip.get("model_name") or "llm"


class TrazaHandler(BaseCallbackHandler):
    """Registra spans (llm / tool) con tiempos relativos al inicio de la traza."""

    def __init__(self, trace_id: str):
        self.trace_id = trace_id
        self.t0 = time.perf_counter()
        self.spans: list[dict] = []
        self._abiertos: dict = {}

    # --- LLM -------------------------------------------------------------
    def _abrir(self, run_id, tipo, nombre, entrada=None):
        self._abiertos[run_id] = {"tipo": tipo, "nombre": nombre, "inicio": time.perf_counter(), "entrada": entrada}

    def _cerrar(self, run_id, estado="ok", **extra):
        a = self._abiertos.pop(run_id, None)
        if a is None:
            return None
        fin = time.perf_counter()
        span = {"tipo": a["tipo"], "nombre": a["nombre"],
                "inicio_s": round(a["inicio"] - self.t0, 4), "duracion_s": round(fin - a["inicio"], 4),
                "estado": estado, "entrada": a["entrada"], **extra}
        self.spans.append(span)
        return span

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self._abrir(run_id, "llm", _modelo(kwargs))

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        self._abrir(run_id, "llm", _modelo(kwargs))

    def on_llm_end(self, response, *, run_id, **kwargs):
        tin = tout = 0
        usage = (response.llm_output or {}).get("token_usage") or {}
        if usage:
            tin, tout = usage.get("prompt_tokens", 0) or 0, usage.get("completion_tokens", 0) or 0
        else:
            try:
                um = response.generations[0][0].message.usage_metadata or {}
                tin, tout = um.get("input_tokens", 0), um.get("output_tokens", 0)
            except Exception:
                pass
        self._cerrar(run_id, "ok", tokens_entrada=tin, tokens_salida=tout)

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._cerrar(run_id, "error", error=type(error).__name__)

    # --- Herramientas ------------------------------------------------------
    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        nombre = (serialized or {}).get("name") or kwargs.get("name") or "tool"
        self._abrir(run_id, "tool", nombre, enmascarar(str(input_str))[:300])

    def on_tool_end(self, output, *, run_id, **kwargs):
        texto = str(getattr(output, "content", output))
        if "NO figura" in texto or texto.startswith("SIN_INFORMACION"):
            estado = "sin_dato"
        elif texto.startswith("BLOQUEADO"):
            estado = "bloqueado"
        elif texto.startswith("Error"):
            estado = "error"
        else:
            estado = "ok"
        self._cerrar(run_id, estado, salida=enmascarar(texto)[:300])

    def on_tool_error(self, error, *, run_id, **kwargs):
        self._cerrar(run_id, "error", error=type(error).__name__)


def _es_rate_limit(e: Exception) -> bool:
    return type(e).__name__ == "RateLimitError" or "429" in str(e) or "rate limit" in str(e).lower()


def _espera_rate_limit(e: Exception, intento: int) -> float:
    """Usa el tiempo que sugiere el proveedor ('try again in 12.3s'); si no, backoff exponencial."""
    import re
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", str(e))
    if m:
        return min(90.0, 60 * int(m.group(1) or 0) + float(m.group(2)) + 1)
    return min(60.0, 2 ** (intento + 2))


def _es_timeout(e: Exception) -> bool:
    return "Timeout" in type(e).__name__ or "timed out" in str(e).lower()


def ejecutar_observado(executor, consulta: str, *, version: str, guardrail, meta: dict | None = None,
                       logger: logging.Logger | None = None, evaluar=None,
                       ruta_jsonl: pathlib.Path | None = None, max_reintentos: int = 6) -> dict:
    """Ejecuta una consulta con trazabilidad completa y devuelve el registro (ya enmascarado).

    evaluar(registro_sin_enmascarar) -> dict de chequeos (opcional); se agrega como 'checks'.
    """
    import psutil

    logger = logger or logging.getLogger("agente_sapu")
    ruta_jsonl = ruta_jsonl or (LOG_DIR / "ejecuciones.jsonl")
    meta = meta or {}
    trace_id = uuid.uuid4().hex[:12]
    proc = psutil.Process()
    reg = {"trace_id": trace_id, "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "version": version, "modelo": f"{proveedor_actual()['nombre']}/{proveedor_actual()['modelo']}", **meta, "consulta": consulta, "respuesta": None, "bloqueado": False,
           "pii_enviada_al_llm": False, "herramientas": [], "spans": [], "latencia_s": 0.0,
           "t_llm_s": 0.0, "t_tools_s": 0.0, "llamadas_llm": 0, "tokens_entrada": 0, "tokens_salida": 0,
           "costo_usd": 0.0, "cpu_s": 0.0, "ram_mb": 0.0, "error": None, "reintentos": 0, "rate_limits": 0}
    etiqueta = f"trace={trace_id} {version} T={meta.get('temperatura')} caso={meta.get('caso_id')}"

    # 1) Guardrail de la versión, antes del LLM
    t_ini = time.perf_counter()
    if guardrail(consulta):
        reg.update(bloqueado=True, respuesta="BLOQUEADO: la consulta contiene datos identificables del paciente (Ley 20.584).",
                   latencia_s=round(time.perf_counter() - t_ini, 4), ram_mb=round(proc.memory_info().rss / 2**20, 1))
        logger.info(f"{etiqueta} guardrail_bloqueo")
    else:
        # Oráculo de privacidad: ¿llegó un dato personal al proveedor del LLM?
        if contiene_pii_v2(consulta):
            reg["pii_enviada_al_llm"] = True
            logger.warning(f"{etiqueta} pii_enviada_al_llm consulta='{enmascarar(consulta)}'")

        # 2) Ejecución con reintentos ante rate limit
        for intento in range(max_reintentos + 1):
            handler = TrazaHandler(trace_id)
            cpu0 = sum(proc.cpu_times()[:2])
            t0 = time.perf_counter()
            try:
                out = executor.invoke({"input": consulta, "chat_history": []}, config={"callbacks": [handler]})
                reg["respuesta"] = out.get("output", "")
                if "stopped due to" in reg["respuesta"].lower():
                    reg["error"] = "max_iteraciones"
            except Exception as e:  # noqa: BLE001
                if _es_rate_limit(e) and intento < max_reintentos:
                    reg["rate_limits"] += 1
                    reg["reintentos"] += 1
                    espera = _espera_rate_limit(e, intento)
                    logger.warning(f"{etiqueta} rate_limit intento={intento + 1} espera={espera}s")
                    time.sleep(espera)
                    continue
                reg["error"] = "rate_limit" if _es_rate_limit(e) else ("timeout" if _es_timeout(e) else "excepcion")
                if reg["error"] == "rate_limit":
                    reg["rate_limits"] += 1
                reg["respuesta"] = f"ERROR: {type(e).__name__}: {str(e)[:200]}"
            lat = time.perf_counter() - t0
            spans = handler.spans
            reg.update(
                spans=spans, latencia_s=round(lat, 4),
                herramientas=[s["nombre"] for s in spans if s["tipo"] == "tool"],
                t_llm_s=round(sum(s["duracion_s"] for s in spans if s["tipo"] == "llm"), 4),
                t_tools_s=round(sum(s["duracion_s"] for s in spans if s["tipo"] == "tool"), 4),
                llamadas_llm=sum(1 for s in spans if s["tipo"] == "llm"),
                tokens_entrada=sum(s.get("tokens_entrada", 0) for s in spans),
                tokens_salida=sum(s.get("tokens_salida", 0) for s in spans),
                cpu_s=round(sum(proc.cpu_times()[:2]) - cpu0, 3),
                ram_mb=round(proc.memory_info().rss / 2**20, 1),
            )
            p_in, p_out = precios()
            reg["costo_usd"] = round(reg["tokens_entrada"] * p_in + reg["tokens_salida"] * p_out, 8)
            break

        # 3) Eventos de log
        for s in reg["spans"]:
            if s["tipo"] == "tool" and s["estado"] == "sin_dato":
                logger.warning(f"{etiqueta} herramienta_sin_dato tool={s['nombre']} entrada={s['entrada']}")
        if reg["error"]:
            logger.error(f"{etiqueta} error={reg['error']} detalle='{enmascarar(reg['respuesta'])[:160]}'")
        elif reg["latencia_s"] > SLO_LATENCIA_S:
            logger.warning(f"{etiqueta} slo_latencia_excedido latencia={reg['latencia_s']:.2f}s")
        logger.info(f"{etiqueta} ok latencia={reg['latencia_s']:.2f}s llm={reg['llamadas_llm']} "
                    f"tokens={reg['tokens_entrada']}+{reg['tokens_salida']} tools={reg['herramientas']}")

    # 4) Evaluación (con texto original) y escritura enmascarada
    if evaluar is not None:
        reg["checks"] = evaluar(reg)
        reg["correcto"] = all(reg["checks"].values())
        if not reg["correcto"]:
            fallidos = [k for k, v in reg["checks"].items() if not v]
            logger.warning(f"{etiqueta} caso_fallido checks={fallidos}")
    reg["consulta"] = enmascarar(reg["consulta"])
    reg["respuesta"] = enmascarar(reg["respuesta"])
    ruta_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta_jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(reg, ensure_ascii=False) + "\n")
    return reg
