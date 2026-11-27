"""Pruebas deterministas (sin LLM ni API key). Confirman H1 y H2 y validan la instrumentación.

Ejecutar desde ep3_observabilidad/:  python -m pytest -q tests
"""
import json
import sys
import pathlib

import pytest

RAIZ = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from langchain_core.embeddings import DeterministicFakeEmbedding  # noqa: E402

import agente  # noqa: E402
import evaluacion  # noqa: E402
import observabilidad as obs  # noqa: E402

CASOS = evaluacion.cargar_casos(RAIZ / "casos_prueba.json")
POR_ID = {c["id"]: c for c in CASOS}


@pytest.fixture(scope="module")
def vector_db():
    return agente.crear_vector_db(DeterministicFakeEmbedding(size=64))


def _tool(tools, nombre):
    return next(t for t in tools if t.name == nombre)


# --- Set de casos -------------------------------------------------------------
def test_set_de_casos_20_en_9_escenarios():
    assert len(CASOS) == 20
    assert len({c["id"] for c in CASOS}) == 20
    assert len({c["escenario"] for c in CASOS}) == 9


# --- H1: guardrail v1 deja pasar RUT sin guion y nombres ----------------------
@pytest.mark.parametrize("cid", ["C13", "C14", "C15"])
def test_h1_v1_bloquea_rut_con_guion(cid):
    assert agente.contiene_datos_paciente_v1(POR_ID[cid]["consulta"])


@pytest.mark.parametrize("cid", ["C17", "C18"])
def test_h1_v1_deja_pasar_rut_sin_guion_y_nombre(cid):
    assert not agente.contiene_datos_paciente_v1(POR_ID[cid]["consulta"])


@pytest.mark.parametrize("cid", ["C13", "C14", "C15", "C17", "C18"])
def test_r1_v2_bloquea_todos_los_formatos(cid):
    assert agente.contiene_pii_v2(POR_ID[cid]["consulta"])


def test_r1_v2_sin_falsos_positivos():
    for c in CASOS:
        if c["categoria"] != "privacidad":
            assert not agente.contiene_pii_v2(c["consulta"]), c["id"]


# --- H2: error de tipeo oculta un fármaco existente ---------------------------
def test_h2_v1_hidrocortizona_no_figura(vector_db):
    out = _tool(agente.crear_herramientas(vector_db, "v1"), "consultar_vademecum").invoke({"farmaco": "hidrocortizona"})
    assert "NO figura" in out


def test_r2_v2_tolera_tipeo(vector_db):
    vad = _tool(agente.crear_herramientas(vector_db, "v2"), "consultar_vademecum")
    assert "5 mg/kg" in vad.invoke({"farmaco": "hidrocortizona"})
    assert "100 mcg" in vad.invoke({"farmaco": "salbutamool"})


@pytest.mark.parametrize("farmaco", ["zorbalix", "ketorolaco", "amoxicilina"])
def test_r2_v2_no_inventa_farmacos_ausentes(vector_db, farmaco):
    vad = _tool(agente.crear_herramientas(vector_db, "v2"), "consultar_vademecum")
    assert "NO figura" in vad.invoke({"farmaco": farmaco})


# --- Herramienta de razonamiento ----------------------------------------------
@pytest.mark.parametrize("peso,esperado,tope", [(15, "75 mg", False), (20, "100 mg", False), (25, "100 mg", True)])
def test_calcular_dosis_y_tope(vector_db, peso, esperado, tope):
    calc = _tool(agente.crear_herramientas(vector_db, "v1"), "calcular_dosis")
    out = calc.invoke({"peso_kg": peso, "dosis_mg_por_kg": 5, "max_mg": 100})
    assert esperado in out and (("tope" in out) == tope)


# --- Privacidad de los registros ----------------------------------------------
def test_enmascarar_no_deja_datos_personales():
    for cid in ["C13", "C14", "C15", "C17", "C18"]:
        m = agente.enmascarar(POR_ID[cid]["consulta"])
        assert not agente.contiene_pii_v2(m), m


# --- Instrumentación -----------------------------------------------------------
def test_traza_handler_registra_span_de_herramienta(vector_db):
    vad = _tool(agente.crear_herramientas(vector_db, "v1"), "consultar_vademecum")
    h = obs.TrazaHandler("t1")
    vad.invoke({"farmaco": "hidrocortizona"}, config={"callbacks": [h]})
    assert len(h.spans) == 1
    s = h.spans[0]
    assert s["tipo"] == "tool" and s["nombre"] == "consultar_vademecum" and s["estado"] == "sin_dato"
    assert s["duracion_s"] >= 0


class _ExecutorNoDebeLlamarse:
    def invoke(self, *a, **k):
        raise AssertionError("el guardrail debió bloquear antes del LLM")


def test_ejecutar_observado_bloquea_y_enmascara(tmp_path):
    ruta = tmp_path / "e.jsonl"
    caso = POR_ID["C17"]
    reg = obs.ejecutar_observado(_ExecutorNoDebeLlamarse(), caso["consulta"], version="v2",
                                 guardrail=agente.contiene_pii_v2, meta={"caso_id": "C17"},
                                 evaluar=lambda r: evaluacion.evaluar(caso, r), ruta_jsonl=ruta)
    assert reg["bloqueado"] and reg["correcto"] and reg["llamadas_llm"] == 0
    escrito = ruta.read_text(encoding="utf-8")
    assert "123456785" not in escrito and "[RUT]" in escrito


class RateLimitError(Exception):
    pass


class _ExecutorConRateLimit:
    def __init__(self):
        self.n = 0

    def invoke(self, entrada, config=None):
        self.n += 1
        if self.n == 1:
            raise RateLimitError("Error code: 429 - rate limit exceeded")
        return {"output": "Dosis calculada: 75 mg [Vademécum SAPU (DEMO), p.4]"}


def test_ejecutar_observado_reintenta_rate_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(obs.time, "sleep", lambda s: None)
    reg = obs.ejecutar_observado(_ExecutorConRateLimit(), "consulta sin datos personales", version="v1",
                                 guardrail=agente.contiene_datos_paciente_v1, ruta_jsonl=tmp_path / "e.jsonl")
    assert reg["reintentos"] == 1 and reg["rate_limits"] == 1 and reg["error"] is None
    assert "75 mg" in reg["respuesta"]


def test_evaluar_detecta_alucinacion_y_herramientas():
    caso = POR_ID["C09"]
    reg = {"bloqueado": False, "error": None, "herramientas": ["consultar_vademecum", "calcular_dosis"],
           "respuesta": "Ketorolaco: 30 mg"}
    chk = evaluacion.evaluar(caso, reg)
    assert not chk["herramientas"] and not chk["sin_alucinacion"] and not chk["contenido"]
