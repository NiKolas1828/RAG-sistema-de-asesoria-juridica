#!/usr/bin/env python3
"""
test_integracion.py
===================
Tests de integración del flujo completo:
    RAGPipeline → ResponseGenerator → respuesta final

Cubren el contrato entre módulos usando mocks, sin necesidad
de base de datos vectorial ni API keys reales.

Ejecutar con: python test_integracion.py
"""

import sys
from unittest.mock import patch, MagicMock
from datetime import datetime

# ─── Helpers de output ───────────────────────────────────────

def ok(msg):  print(f"  [OK]   {msg}")
def fail(msg, detail=""):
    print(f"  [FAIL] {msg}")
    if detail:
        print(f"         → {detail}")
    sys.exit(1)

# ─── Fixtures ────────────────────────────────────────────────

def _rag_output_con_contexto(score: float = 0.20, query: str = "¿Cuál es la multa por no usar casco?") -> dict:
    """
    Simula el output REAL de RAGPipeline.run().
    Usa las claves exactas que devuelve ContextBuilder:
      - contexto_formateado, documentos_usados, score_promedio
    """
    return {
        "query_original": query,
        "status": "éxito",
        "mensaje": "Prompt contextualizado generado correctamente",
        "contexto": {
            "contexto_formateado": (
                "[1] Resolución Única Compilatoria | Art. 7.4.4\n"
                "Los conductores y acompañantes de motocicletas deben usar "
                "obligatoriamente el casco protector homologado.\n\n"
                "[2] Resolución 3027 de 2010 | Art. 8\n"
                "D.06. Conducir vehículo sin usar el casco protector. "
                "Sanción: multa equivalente a 15 SMLDV."
            ),
            "documentos_usados": 2,
            "tokens_contexto": 120,
            "tokens_disponibles": 1880,
            "score_promedio": score,
        },
        "prompt": {
            "prompt": (
                "[SYSTEM]\nEres un asistente jurídico experto...\n\n"
                "[CONTEXT]\n[1] Art. 7.4.4...\n\n"
                f"[QUESTION]\n{query}\n"
            ),
            "tokens_prompt": 280,
            "tokens_available_for_response": 232,
        },
        "search_results": {
            "query_original": query,
            "total_encontrados": 10,
            "resultados": [],
            "status": "éxito",
        },
    }


def _rag_output_sin_contexto(query: str = "holq") -> dict:
    """Simula un output de RAGPipeline sin contexto relevante (score bajo)."""
    return {
        "query_original": query,
        "status": "éxito",
        "contexto": {
            "contexto_formateado": "",
            "documentos_usados": 0,
            "tokens_contexto": 0,
            "tokens_disponibles": 2000,
            "score_promedio": 0.01,  # debajo del umbral de 0.10
        },
        "prompt": {
            "prompt": "",
            "tokens_prompt": 0,
            "tokens_available_for_response": 0,
        },
        "search_results": {"total_encontrados": 0, "resultados": [], "status": "éxito"},
    }


# ─── Tests ───────────────────────────────────────────────────

def test_flujo_exitoso_con_gemini():
    """
    TEST 1: Flujo completo exitoso.
    RAGPipeline devuelve contexto relevante → ResponseGenerator
    llama a Gemini → retorna respuesta estructurada.
    """
    from src.generation.response_generator import ResponseGenerator

    respuesta_esperada = (
        "**Respuesta directa:** La multa por no usar casco es de 15 SMLDV.\n\n"
        "**Detalle normativo:** Según el Art. 7.4.4 [1]...\n\n"
        "**Fuentes:** [1] Resolución Única Compilatoria"
    )

    rag_output = _rag_output_con_contexto(score=0.20)

    with patch("src.generation.response_generator.GeminiClient") as MockGemini:
        mock_instance = MagicMock()
        mock_instance.generate.return_value = respuesta_esperada
        MockGemini.return_value = mock_instance

        generator = ResponseGenerator()
        resultado = generator.generate(rag_output)

    assert resultado["status"] == "éxito", \
        f"Status inesperado: {resultado['status']}"
    assert resultado["respuesta"] == respuesta_esperada, \
        "La respuesta no coincide con la generada por el mock"
    assert "gemini" in resultado["modelo_usado"].lower(), \
        f"Modelo inesperado: {resultado['modelo_usado']}"
    assert mock_instance.generate.called, \
        "Gemini no fue llamado aunque había contexto suficiente"

    ok("Flujo exitoso con Gemini — contexto → LLM → respuesta estructurada")


def test_sin_contexto_no_llama_al_llm():
    """
    TEST 2: Anti-alucinación.
    Si score_promedio < umbral (0.10), el sistema retorna
    RESPUESTA_SIN_CONTEXTO sin llamar al LLM.
    """
    from src.generation.response_generator import ResponseGenerator, RESPUESTA_SIN_CONTEXTO

    rag_output = _rag_output_sin_contexto()

    with patch("src.generation.response_generator.GeminiClient") as MockGemini, \
         patch("src.generation.response_generator.GroqClient") as MockGroq:

        mock_gemini = MagicMock()
        mock_groq = MagicMock()
        MockGemini.return_value = mock_gemini
        MockGroq.return_value = mock_groq

        generator = ResponseGenerator()
        resultado = generator.generate(rag_output)

    assert resultado["status"] == "sin_contexto", \
        f"Status inesperado: {resultado['status']} (debería ser 'sin_contexto')"
    assert resultado["respuesta"] == RESPUESTA_SIN_CONTEXTO, \
        "La respuesta no es el mensaje estándar de sin contexto"
    assert resultado["modelo_usado"] == "sin_llm", \
        f"Se usó un modelo cuando no debería: {resultado['modelo_usado']}"
    assert not mock_gemini.generate.called, \
        "Gemini fue llamado aunque no había contexto suficiente (alucinación)"
    assert not mock_groq.generate.called, \
        "Groq fue llamado aunque no había contexto suficiente (alucinación)"

    ok("Anti-alucinación — score bajo → RESPUESTA_SIN_CONTEXTO, LLM no llamado")


def test_fallback_gemini_a_groq():
    """
    TEST 3: Fallback automático.
    Gemini lanza GeminiRateLimitError → ResponseGenerator
    activa Groq automáticamente y retorna su respuesta.
    """
    from src.generation.response_generator import ResponseGenerator
    from src.generation.gemini_client import GeminiRateLimitError

    respuesta_groq = (
        "**Respuesta directa:** La multa es de 15 SMLDV.\n\n"
        "**Fuentes:** [1] Resolución Única Compilatoria | Art. 7.4.4"
    )

    rag_output = _rag_output_con_contexto(score=0.20)

    with patch("src.generation.response_generator.GeminiClient") as MockGemini, \
         patch("src.generation.response_generator.GroqClient") as MockGroq:

        mock_gemini = MagicMock()
        mock_gemini.generate.side_effect = GeminiRateLimitError("429 rate limit")
        MockGemini.return_value = mock_gemini

        mock_groq = MagicMock()
        mock_groq.generate.return_value = respuesta_groq
        MockGroq.return_value = mock_groq

        generator = ResponseGenerator()
        resultado = generator.generate(rag_output)

    assert resultado["status"] == "éxito", \
        f"Status inesperado tras fallback: {resultado['status']}"
    assert resultado["respuesta"] == respuesta_groq, \
        "La respuesta del fallback Groq no es la esperada"
    assert "groq" in resultado["modelo_usado"].lower(), \
        f"Modelo inesperado tras fallback: {resultado['modelo_usado']}"
    assert mock_gemini.generate.called, "Gemini no fue intentado primero"
    assert mock_groq.generate.called, "Groq no fue activado como fallback"

    ok("Fallback Gemini→Groq — rate limit detectado, Groq responde correctamente")


def test_estructura_completa_del_resultado():
    """
    TEST 4: Contrato de la respuesta.
    El dict resultado debe contener siempre los campos requeridos
    con los tipos correctos, independientemente del caso.
    """
    from src.generation.response_generator import ResponseGenerator

    CAMPOS_REQUERIDOS = {
        "query_original": str,
        "respuesta": str,
        "modelo_usado": str,
        "tokens_prompt": int,
        "status": str,
        "timestamp": str,
        "error": (str, type(None)),
    }

    casos = [
        ("con_contexto", _rag_output_con_contexto(score=0.20)),
        ("sin_contexto", _rag_output_sin_contexto()),
    ]

    for nombre, rag_output in casos:
        with patch("src.generation.response_generator.GeminiClient") as MockGemini:
            mock_instance = MagicMock()
            mock_instance.generate.return_value = "Respuesta de prueba del LLM."
            MockGemini.return_value = mock_instance

            generator = ResponseGenerator()
            resultado = generator.generate(rag_output)

        for campo, tipo in CAMPOS_REQUERIDOS.items():
            assert campo in resultado, \
                f"[{nombre}] Campo '{campo}' ausente en el resultado"
            if isinstance(tipo, tuple):
                assert isinstance(resultado[campo], tipo), \
                    f"[{nombre}] Campo '{campo}' tiene tipo {type(resultado[campo])}, esperado {tipo}"
            else:
                assert isinstance(resultado[campo], tipo), \
                    f"[{nombre}] Campo '{campo}' tiene tipo {type(resultado[campo])}, esperado {tipo}"

        # Verificar que timestamp es un ISO datetime válido
        try:
            datetime.fromisoformat(resultado["timestamp"])
        except ValueError:
            fail(f"[{nombre}] timestamp no es ISO format: {resultado['timestamp']}")

    ok("Estructura del resultado — todos los campos presentes con tipos correctos")


# ─── Runner ──────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("=" * 70)
    print("  TESTS DE INTEGRACIÓN — RAGPipeline → ResponseGenerator")
    print("=" * 70)

    tests = [
        ("Flujo exitoso con Gemini",           test_flujo_exitoso_con_gemini),
        ("Sin contexto no llama al LLM",       test_sin_contexto_no_llama_al_llm),
        ("Fallback automático Gemini → Groq",  test_fallback_gemini_a_groq),
        ("Estructura completa del resultado",  test_estructura_completa_del_resultado),
    ]

    pasaron = 0
    for nombre, test_fn in tests:
        try:
            test_fn()
            pasaron += 1
        except AssertionError as e:
            fail(nombre, str(e))
        except Exception as e:
            fail(nombre, f"Error inesperado: {e}")

    print("=" * 70)
    print(f"  Resultado: {pasaron}/{len(tests)} pruebas pasaron")
    print("=" * 70)
    print()
