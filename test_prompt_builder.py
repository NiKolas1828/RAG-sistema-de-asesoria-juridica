#!/usr/bin/env python3
"""
test_prompt_builder.py
============================================================
Pruebas unitarias para PromptBuilder.

Valida que el prompt RAG ensamblado cumpla con los criterios
de aceptación del feature "Diseño de prompts documentado":
  - Contiene los 4 bloques obligatorios
  - El system prompt incluye las 4 reglas anti-alucinación
  - La instrucción de formato exige la estructura tripartita
  - La estimación de tokens retorna un entero positivo
  - El contexto formateado queda incluido en el prompt
  - Las instrucciones adicionales se incorporan correctamente

Ejecución:
    python test_prompt_builder.py
============================================================
"""

from src.context_builder.prompt_builder import PromptBuilder, SYSTEM_PROMPT, INSTRUCTION_BLOCK


# ─── Helpers ─────────────────────────────────────────────────

def assert_true(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


CONTEXTO_EJEMPLO = (
    "[1] Código Nacional de Tránsito (Ley 769 de 2002) | Art. 106 | Relevancia: 88.5%\n"
    "------------------------------------------------------------\n"
    "Todo conductor de motocicleta debe portar casco de seguridad homologado. "
    "El incumplimiento acarrea multa de 15 salarios mínimos legales diarios vigentes."
)

PREGUNTA_EJEMPLO = "¿Cuál es la multa por no usar casco en moto?"


# ─── TEST 1: Los 4 bloques obligatorios están presentes ──────

def test_prompt_contiene_cuatro_bloques():
    """El prompt RAG debe incluir siempre los 4 bloques estructurales."""
    builder = PromptBuilder()
    resultado = builder.build_prompt(CONTEXTO_EJEMPLO, PREGUNTA_EJEMPLO)
    prompt = resultado["prompt"]

    assert_true("[SYSTEM]" in prompt,      "Falta el bloque [SYSTEM] en el prompt")
    assert_true("[INSTRUCTION]" in prompt, "Falta el bloque [INSTRUCTION] en el prompt")
    assert_true("[CONTEXT]" in prompt,     "Falta el bloque [CONTEXT] en el prompt")
    assert_true("[QUESTION]" in prompt,    "Falta el bloque [QUESTION] en el prompt")


# ─── TEST 2: El SYSTEM_PROMPT contiene las 4 reglas ──────────

def test_system_prompt_contiene_reglas_anti_alucinacion():
    """Las 4 reglas obligatorias del system prompt deben estar presentes."""

    # Regla 1: fidelidad al contexto RAG
    assert_true(
        "ÚNICAMENTE" in SYSTEM_PROMPT,
        "Regla 1 ausente: el system prompt debe restringir la respuesta al [CONTEXT]"
    )
    # Regla 2: cita obligatoria de fuente normativa
    assert_true(
        "Cita siempre" in SYSTEM_PROMPT,
        "Regla 2 ausente: el system prompt debe exigir citar normas y artículos"
    )
    # Regla 3: tono accesible para ciudadanos
    assert_true(
        "ciudadano" in SYSTEM_PROMPT,
        "Regla 3 ausente: el system prompt debe indicar el tono para el ciudadano"
    )
    # Regla 4: prohibición de inventar leyes
    assert_true(
        "inventes" in SYSTEM_PROMPT,
        "Regla 4 ausente: el system prompt debe prohibir inventar artículos o leyes"
    )


# ─── TEST 3: La instrucción de formato exige estructura tripartita ─

def test_instruction_block_exige_estructura_tripartita():
    """El bloque de instrucciones debe exigir Respuesta directa, Detalle y Fuentes."""
    assert_true(
        "Respuesta directa" in INSTRUCTION_BLOCK,
        "Falta 'Respuesta directa' en el bloque de instrucciones"
    )
    assert_true(
        "Detalle normativo" in INSTRUCTION_BLOCK,
        "Falta 'Detalle normativo' en el bloque de instrucciones"
    )
    assert_true(
        "Fuentes" in INSTRUCTION_BLOCK,
        "Falta 'Fuentes' en el bloque de instrucciones"
    )


# ─── TEST 4: La estimación de tokens retorna entero positivo ─

def test_tokens_prompt_es_entero_positivo():
    """tokens_prompt debe ser un entero mayor que cero."""
    builder = PromptBuilder()
    resultado = builder.build_prompt(CONTEXTO_EJEMPLO, PREGUNTA_EJEMPLO)

    tokens = resultado["tokens_prompt"]
    assert_true(isinstance(tokens, int), f"tokens_prompt debe ser int, fue: {type(tokens)}")
    assert_true(tokens > 0, f"tokens_prompt debe ser mayor que 0, fue: {tokens}")


# ─── TEST 5: El contexto recuperado queda incluido en el prompt ─

def test_contexto_queda_en_el_prompt():
    """El texto del contexto RAG debe aparecer en el prompt final."""
    builder = PromptBuilder()
    resultado = builder.build_prompt(CONTEXTO_EJEMPLO, PREGUNTA_EJEMPLO)
    prompt = resultado["prompt"]

    assert_true(
        "Art. 106" in prompt,
        "El artículo del contexto recuperado no aparece en el prompt"
    )
    assert_true(
        "casco de seguridad" in prompt,
        "El contenido del contexto recuperado no aparece en el prompt"
    )


# ─── TEST 6: La pregunta del ciudadano queda en el prompt ────

def test_pregunta_queda_en_el_prompt():
    """La pregunta original del ciudadano debe aparecer en el bloque [QUESTION]."""
    builder = PromptBuilder()
    resultado = builder.build_prompt(CONTEXTO_EJEMPLO, PREGUNTA_EJEMPLO)
    prompt = resultado["prompt"]

    assert_true(
        PREGUNTA_EJEMPLO in prompt,
        "La pregunta del ciudadano no aparece en el bloque [QUESTION]"
    )


# ─── TEST 7: Las instrucciones adicionales se incorporan ─────

def test_instrucciones_adicionales_se_incorporan():
    """Si se pasan additional_instructions, deben aparecer en el bloque [SYSTEM]."""
    instruccion_extra = "Responde en menos de 3 oraciones."
    builder = PromptBuilder()
    resultado = builder.build_prompt(
        CONTEXTO_EJEMPLO,
        PREGUNTA_EJEMPLO,
        additional_instructions=instruccion_extra,
    )
    prompt = resultado["prompt"]

    assert_true(
        instruccion_extra in prompt,
        "Las instrucciones adicionales no aparecen en el prompt"
    )


# ─── TEST 8: Prompt vacío sin contexto retorna estructura válida ─

def test_prompt_sin_contexto_tiene_estructura():
    """Incluso sin contexto, el prompt debe tener los 4 bloques (el RAG puede estar vacío)."""
    builder = PromptBuilder()
    resultado = builder.build_prompt(contexto_formateado="", query=PREGUNTA_EJEMPLO)
    prompt = resultado["prompt"]

    assert_true("[SYSTEM]" in prompt,      "Falta [SYSTEM] incluso sin contexto")
    assert_true("[INSTRUCTION]" in prompt, "Falta [INSTRUCTION] incluso sin contexto")
    assert_true("[CONTEXT]" in prompt,     "Falta [CONTEXT] incluso sin contexto")
    assert_true("[QUESTION]" in prompt,    "Falta [QUESTION] incluso sin contexto")


# ─── Ejecución ───────────────────────────────────────────────

def main():
    tests = [
        ("4 bloques obligatorios presentes",             test_prompt_contiene_cuatro_bloques),
        ("4 reglas anti-alucinación en SYSTEM_PROMPT",   test_system_prompt_contiene_reglas_anti_alucinacion),
        ("Estructura tripartita en INSTRUCTION_BLOCK",   test_instruction_block_exige_estructura_tripartita),
        ("tokens_prompt es entero positivo",             test_tokens_prompt_es_entero_positivo),
        ("Contexto RAG incluido en el prompt",           test_contexto_queda_en_el_prompt),
        ("Pregunta del ciudadano en [QUESTION]",         test_pregunta_queda_en_el_prompt),
        ("Instrucciones adicionales incorporadas",       test_instrucciones_adicionales_se_incorporan),
        ("Estructura válida sin contexto",               test_prompt_sin_contexto_tiene_estructura),
    ]

    print("=" * 70)
    print("  PRUEBAS DE PROMPT BUILDER — Diseño de Prompts RAG")
    print("=" * 70)

    passed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"  [OK]   {name}")
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {name}")
            print(f"         → {e}")

    print("=" * 70)
    print(f"  Resultado: {passed}/{len(tests)} pruebas pasaron")
    print("=" * 70)

    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
