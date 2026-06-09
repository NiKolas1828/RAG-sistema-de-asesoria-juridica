#!/usr/bin/env python3
"""
Pruebas directas para ContextBuilder.
Ejecuta: python -m pytest tests/unit/test_context_builder.py
"""

from src.context_builder.context_builder import ContextBuilder


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def test_empty_input():
    builder = ContextBuilder()
    result = builder.build_context(None)

    assert_true(result["contexto_formateado"] == "", "El contexto vacío debe devolver texto vacío")
    assert_true(result["documentos_usados"] == 0, "No deben usarse documentos en entrada vacía")
    assert_true(result["tokens_contexto"] == 0, "Los tokens deben ser 0 en entrada vacía")
    assert_true(result["score_promedio"] == 0.0, "El score promedio debe ser 0.0 en entrada vacía")


def test_filter_and_sort_by_score():
    builder = ContextBuilder(min_similarity=0.7)
    results = [
        {"texto": "Bajo", "metadata": {"articulo": "1", "fuente": "A"}, "similitud": 0.55},
        {"texto": "Alto", "metadata": {"articulo": "2", "fuente": "B"}, "similitud": 0.92},
        {"texto": "Medio", "metadata": {"articulo": "3", "fuente": "C"}, "similitud": 0.76},
    ]

    filtered = builder._filter_by_score(results)

    assert_true(len(filtered) == 2, "Solo deben pasar los documentos con similitud >= 0.7")
    assert_true(filtered[0]["similitud"] >= filtered[1]["similitud"], "Los resultados deben ordenarse de mayor a menor")
    assert_true(filtered[0]["texto"] == "Alto", "El resultado más relevante debe quedar primero")


def test_filter_fallback_when_none_meet_threshold():
    builder = ContextBuilder(min_similarity=0.95)
    results = [
        {"texto": "Uno", "metadata": {}, "similitud": 0.62},
        {"texto": "Dos", "metadata": {}, "similitud": 0.71},
    ]

    filtered = builder._filter_by_score(results)

    assert_true(len(filtered) == 2, "Si ninguno supera el umbral, debe devolverse la lista ordenada")
    assert_true(filtered[0]["similitud"] == 0.71, "El fallback debe seguir ordenando por similitud")


def test_format_document():
    builder = ContextBuilder()
    document = {
        "texto": "Linea 1\nLinea 2\n\nLinea 3",
        "metadata": {"articulo": "Art. 10", "fuente": "Resolución 160"},
        "similitud": 0.8123,
    }

    formatted = builder._format_document(document, 1)

    assert_true("[1] Resolución 160 | Art. 10 | Relevancia: 81.23%" in formatted, "El encabezado debe incluir fuente, artículo y relevancia")
    assert_true("Linea 1 Linea 2 Linea 3" in formatted, "El texto debe normalizar saltos de línea")


def test_build_context_basic():
    builder = ContextBuilder(max_tokens=2000, min_similarity=0.7)
    search_results = {
        "resultados": [
            {
                "texto": "Texto relevante A",
                "metadata": {"articulo": "Art. 1", "fuente": "Norma A"},
                "similitud": 0.93,
            },
            {
                "texto": "Texto relevante B",
                "metadata": {"articulo": "Art. 2", "fuente": "Norma B"},
                "similitud": 0.88,
            },
            {
                "texto": "Texto irrelevante",
                "metadata": {"articulo": "Art. 3", "fuente": "Norma C"},
                "similitud": 0.41,
            },
        ]
    }

    result = builder.build_context(search_results, max_documents=3)

    assert_true(result["documentos_usados"] == 2, "Solo deben entrar los documentos por encima del umbral")
    assert_true("Texto relevante A" in result["contexto_formateado"], "Debe incluir el documento más relevante")
    assert_true("Texto relevante B" in result["contexto_formateado"], "Debe incluir el segundo documento relevante")
    assert_true("Texto irrelevante" not in result["contexto_formateado"], "No debe incluir documentos bajo el umbral")
    assert_true(abs(result["score_promedio"] - 0.905) < 0.01, "El score promedio debe calcularse correctamente")


def test_token_limit_enforced():
    builder = ContextBuilder(max_tokens=35, min_similarity=0.0)
    long_text = " ".join(["palabra"] * 300)
    search_results = {
        "resultados": [
            {
                "texto": long_text,
                "metadata": {"articulo": "Art. 99", "fuente": "Norma Larga"},
                "similitud": 0.99,
            }
        ]
    }

    result = builder.build_context(search_results, max_documents=1)

    assert_true(result["tokens_contexto"] <= 35, "El contexto final debe respetar el límite de tokens")
    assert_true(result["contexto_formateado"] != "", "Debe devolver un contexto aunque tenga que truncar")
    assert_true(result["documentos_usados"] == 1, "Debe seguir contando el documento como usado")


def main():
    tests = [
        ("Entrada vacía", test_empty_input),
        ("Filtrado y orden por score", test_filter_and_sort_by_score),
        ("Fallback sin umbral", test_filter_fallback_when_none_meet_threshold),
        ("Formato de documento", test_format_document),
        ("Construcción básica", test_build_context_basic),
        ("Límite de tokens", test_token_limit_enforced),
    ]

    print("=" * 70)
    print("PRUEBAS DE CONTEXT BUILDER")
    print("=" * 70)

    passed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"[OK] {name}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")

    print("=" * 70)
    print(f"Resultado: {passed}/{len(tests)} pruebas pasaron")
    print("=" * 70)

    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
