# test_generation.py
# ============================================================
# Pruebas de generación — Historia de usuario: configurar LLM
# Cubre los 3 criterios de aceptación:
#   1. Ajuste de parámetros (temperature, max_tokens, etc.)
#   2. Respuestas coherentes y formales
#   3. Se evita alucinación en lo posible
# ============================================================
#
# Ejecución:
#   python -m pytest tests/unit/test_generation.py
#
# Requiere .env con GEMINI_API_KEY y/o GROQ_API_KEY
# ============================================================

import sys
import json
import logging
from unittest.mock import MagicMock, patch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Colores para la consola ─────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"

def ok(msg):  logger.info(f"{GREEN}✓ {msg}{RESET}")
def fail(msg): logger.error(f"{RED}✗ {msg}{RESET}")
def info(msg): logger.info(f"{CYAN}→ {msg}{RESET}")


# ─── Helpers ────────────────────────────────────────────────

def _mock_rag_output(query: str, similitud: float = 0.85) -> dict:
    """Simula el output de RAGPipeline para tests sin base de datos.

    Usa el formato real que devuelve ContextBuilder:
      - contexto_formateado, documentos_usados, score_promedio
    (NO chunks_seleccionados, que no existe en el output real)
    """
    return {
        "query_original": query,
        "status": "éxito",
        "contexto": {
            "contexto_formateado": (
                "[1] Art. 106 — Ley 769 de 2002\n"
                "Todo conductor de motocicleta, motoneta y bicicleta motorizada, "
                "debe portar casco de seguridad homologado. El incumplimiento de "
                "esta norma acarrea multa equivalente a quince (15) salarios mínimos "
                "legales diarios vigentes."
            ),
            "documentos_usados": 1,
            "tokens_contexto": 80,
            "tokens_disponibles": 1920,
            "score_promedio": similitud,
        },
        "prompt": {
            "prompt": (
                "[SYSTEM]\nEres un asistente...\n\n"
                "[CONTEXT]\n[1] Art. 106...\n\n"
                f"[QUESTION]\n{query}\n"
            ),
            "tokens_prompt": 220,
            "tokens_available_for_response": 292,
        },
    }


def _mock_rag_output_sin_contexto(query: str) -> dict:
    """Simula un RAG output sin chunks relevantes (similitud muy baja)."""
    return {
        "query_original": query,
        "status": "éxito",
        "contexto": {
            "contexto_formateado": "",
            "documentos_usados": 0,
            "tokens_contexto": 0,
            "tokens_disponibles": 2000,
            "score_promedio": 0.0,
        },
        "prompt": {"prompt": "", "tokens_prompt": 0, "tokens_available_for_response": 0},
    }


# ─── TEST 1: Parámetros del modelo ───────────────────────────

def test_parametros_llm_config():
    """CA1: Verifica que los parámetros están configurados correctamente."""
    info("TEST 1: Verificación de parámetros LLM")

    from src.generation.llm_config import GeminiConfig, GroqConfig, GenerationConfig

    gcfg = GeminiConfig()
    assert 0.0 <= gcfg.temperature <= 0.3, (
        f"Temperature debe ser baja para dominio legal. Valor: {gcfg.temperature}"
    )
    assert gcfg.max_output_tokens <= 8192, (
        f"max_output_tokens supera el límite del free tier. Valor: {gcfg.max_output_tokens}"
    )
    assert gcfg.model_name == "gemini-2.0-flash", (
        f"Modelo incorrecto: {gcfg.model_name}"
    )
    assert "ÚNICAMENTE" in gcfg.system_instruction, (
        "System instruction debe restringir al contexto (palabra clave: ÚNICAMENTE)"
    )
    assert "cita" in gcfg.system_instruction.lower(), (
        "System instruction debe exigir citas de artículos"
    )

    rcfg = GroqConfig()
    assert rcfg.temperature == gcfg.temperature, (
        "Temperature de Groq debe coincidir con Gemini para respuestas consistentes"
    )

    cfg = GenerationConfig()
    assert 0.0 <= cfg.min_similarity_to_generate <= 1.0, (
        f"Umbral de similitud fuera de rango válido (0-1): {cfg.min_similarity_to_generate}"
    )

    ok("Parámetros LLM correctamente configurados")


# ─── TEST 2: Respuesta coherente y formal (con mock de Gemini) ─

def test_respuesta_formal_con_contexto():
    """CA2: La respuesta debe ser formal, citar artículo y no estar vacía."""
    info("TEST 2: Coherencia y formalidad de la respuesta")

    # Mock: simula que Gemini responde correctamente
    respuesta_simulada = (
        "Según el Artículo 106 del Código Nacional de Tránsito (Ley 769 de 2002), "
        "todo conductor de motocicleta debe portar casco de seguridad homologado. "
        "El incumplimiento acarrea una multa de 15 salarios mínimos legales diarios vigentes.\n\n"
        "Referencias: Art. 106, Ley 769 de 2002."
    )

    with patch("src.generation.gemini_client.GeminiClient.generate", return_value=respuesta_simulada):
        from src.generation.response_generator import ResponseGenerator
        generator = ResponseGenerator()

        rag_output = _mock_rag_output("¿Cuál es la multa por no usar casco?")
        resultado = generator.generate(rag_output)

    assert resultado["status"] == "éxito", f"Status inesperado: {resultado['status']}"
    assert resultado["respuesta"], "La respuesta no debe estar vacía"
    assert len(resultado["respuesta"]) > 50, "Respuesta demasiado corta para ser útil"

    respuesta = resultado["respuesta"].lower()
    assert any(kw in respuesta for kw in ["artículo", "art.", "ley", "resolución", "norma"]), (
        "La respuesta debe citar la fuente normativa"
    )
    assert resultado["modelo_usado"] == "gemini-2.0-flash", (
        f"Modelo incorrecto: {resultado['modelo_usado']}"
    )

    ok(f"Respuesta formal recibida ({len(resultado['respuesta'])} chars)")
    info(f"Fragmento: {resultado['respuesta'][:120]}...")


# ─── TEST 3: Anti-alucinación — sin contexto suficiente ──────

def test_sin_contexto_no_alucina():
    """CA3: Sin contexto relevante, NO debe inventar respuesta."""
    info("TEST 3: Anti-alucinación con contexto insuficiente")

    from src.generation.response_generator import ResponseGenerator, RESPUESTA_SIN_CONTEXTO
    generator = ResponseGenerator()

    rag_output = _mock_rag_output_sin_contexto("¿Cuánto cuesta la revisión técnico-mecánica?")
    resultado = generator.generate(rag_output)

    assert resultado["status"] == "sin_contexto", (
        f"Debería ser 'sin_contexto', fue: {resultado['status']}"
    )
    assert resultado["modelo_usado"] == "sin_llm", (
        "No debe llamar al LLM si no hay contexto suficiente"
    )
    assert resultado["respuesta"] == RESPUESTA_SIN_CONTEXTO, (
        "Debe retornar el mensaje estándar de 'no encontré información'"
    )

    ok("Sistema rechazó generar respuesta sin contexto suficiente (anti-alucinación ✓)")


# ─── TEST 4: Fallback automático Gemini → Groq ───────────────

def test_fallback_groq_ante_rate_limit():
    """CA1 + CA2: Al superar rate limit de Gemini, activa Groq automáticamente."""
    info("TEST 4: Fallback automático Gemini → Groq")

    respuesta_groq = (
        "De acuerdo con el Artículo 106 de la Ley 769 de 2002, "
        "el uso del casco es obligatorio. La multa es de 15 SMLDV."
    )

    from src.generation.gemini_client import GeminiRateLimitError

    with patch("src.generation.gemini_client.GeminiClient.generate",
               side_effect=GeminiRateLimitError("429 rate limit")):
        with patch("src.generation.groq_client.GroqClient.generate",
                   return_value=respuesta_groq):
            from src.generation.response_generator import ResponseGenerator
            generator = ResponseGenerator()
            rag_output = _mock_rag_output("¿Cuál es la multa por no usar casco?")
            resultado = generator.generate(rag_output)

    assert resultado["status"] == "éxito", f"Status inesperado: {resultado['status']}"
    assert "groq" in resultado["modelo_usado"].lower(), (
        f"Debería usar Groq como fallback, usó: {resultado['modelo_usado']}"
    )
    assert resultado["respuesta"] == respuesta_groq

    ok(f"Fallback a Groq funcionó correctamente (modelo: {resultado['modelo_usado']})")


# ─── TEST 5: Estructura del resultado ────────────────────────

def test_estructura_resultado():
    """Verifica que el resultado siempre tiene los campos requeridos."""
    info("TEST 5: Estructura del resultado de generación")

    campos_requeridos = [
        "query_original", "timestamp", "modelo_usado",
        "respuesta", "tokens_prompt", "status", "error"
    ]

    from src.generation.response_generator import ResponseGenerator
    generator = ResponseGenerator()

    rag_output = _mock_rag_output_sin_contexto("pregunta sin contexto")
    resultado = generator.generate(rag_output)

    for campo in campos_requeridos:
        assert campo in resultado, f"Campo requerido ausente en el resultado: '{campo}'"

    ok("Estructura del resultado correcta — todos los campos presentes")


# ─── Ejecución ───────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_parametros_llm_config,
        test_respuesta_formal_con_contexto,
        test_sin_contexto_no_alucina,
        test_fallback_groq_ante_rate_limit,
        test_estructura_resultado,
    ]

    passed = 0
    failed = 0

    print(f"\n{CYAN}{'='*60}")
    print("  PRUEBAS DE GENERACIÓN — RAG Normas de Tránsito")
    print(f"{'='*60}{RESET}\n")

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            fail(f"{test.__name__}: {e}")
            failed += 1
        except Exception as e:
            fail(f"{test.__name__}: Error inesperado — {e}")
            failed += 1
        print()

    print(f"{CYAN}{'='*60}{RESET}")
    print(f"  Resultado: {GREEN}{passed} pasaron{RESET}  |  {RED}{failed} fallaron{RESET}")
    print(f"{CYAN}{'='*60}{RESET}\n")

    sys.exit(0 if failed == 0 else 1)
