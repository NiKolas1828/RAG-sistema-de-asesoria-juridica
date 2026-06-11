# src/context_builder/question_classifier.py
# ============================================================
# Clasificador de Intención de Pregunta (sin LLM, reglas léxicas)
#
# Detecta el TIPO de consulta del usuario para que el prompt
# builder construya el formato de respuesta más adecuado.
#
# Categorías:
#   MULTA       → "¿cuánto es la multa...?", "valor infracción..."
#   REQUISITOS  → "¿qué documentos...?", "requisitos para..."
#   USO_CORRECTO→ "¿cómo debo...?", "¿cómo se usa...?"
#   COMPARATIVO → "diferencia entre...", "vs", "mejor..."
#   PROCEDIMIENTO → "¿qué hago si...?", "cómo tramitar..."
#   INFRACCION  → "me pusieron un comparendo", "me multaron..."
#   GENERAL     → todo lo que no encaja en las anteriores
# ============================================================

import re
from enum import Enum


class QuestionType(Enum):
    MULTA        = "multa"
    REQUISITOS   = "requisitos"
    USO_CORRECTO = "uso_correcto"
    COMPARATIVO  = "comparativo"
    PROCEDIMIENTO = "procedimiento"
    INFRACCION   = "infraccion"
    GENERAL      = "general"


# ─── Patrones léxicos (orden = prioridad) ─────────────────────────────────────
# Los patrones DEBEN estar sin tildes porque la función classify() remueve las
# tildes antes de evaluar las expresiones regulares.
_PATTERNS = [
    (QuestionType.MULTA, [
        r"cuanto.*multa",
        r"valor.*multa",
        r"cuanto.*comparendo",
        r"multa.*cuanto",
        r"cuanto.*infraccion",
        r"precio.*multa",
        r"valor.*infraccion",
        r"cuanto.*pagar",
        r"monto.*multa",
        r"cuanto.*cuesta",
        r"costo.*multa",
        r"costo.*comparendo",
        r"costo.*infraccion",
        r"\bsmmlv\b",
        r"\buvt\b.*multa",
    ]),
    (QuestionType.REQUISITOS, [
        r"que documentos",
        r"que papeles",
        r"que se (necesita|requiere)",
        r"requisitos (para|de)",
        r"documentos (necesarios|requeridos)",
        r"que (debo|tengo) (llevar|presentar|tener)",
        r"cuales son los requisitos",
    ]),
    (QuestionType.USO_CORRECTO, [
        r"como (debo|debe|se debe|hay que) (usar|llevar|portar|colocar|ajustar)",
        r"como (se usa|se porta|se coloca)",
        r"manera correcta de",
        r"forma correcta de",
        r"instrucciones para usar",
        r"bien (puesto|colocado|ajustado)",
    ]),
    (QuestionType.COMPARATIVO, [
        r"diferencia.*entre",
        r"versus|\bvs\b",
        r"mejor.*entre",
        r"comparar|comparacion",
        r"que tan diferente",
        r"mas (seguro|exigente|costoso).*que",
    ]),
    (QuestionType.INFRACCION, [
        r"me (pusieron|impusieron|levantaron).*comparendo",
        r"me (multaron|detectaron|atraparon)",
        r"tengo un comparendo",
        r"recibi una multa",
        r"me puso.*policia",
        r"comparendo.*que (hago|puedo|debo)",
        r"impugnar.*comparendo",
        r"recurso de reposicion",
    ]),
    (QuestionType.PROCEDIMIENTO, [
        r"como (tramitar|obtener|sacar|renovar|solicitar)",
        r"donde (tramitar|ir|pagar|reportar)",
        r"proceso para (obtener|sacar|renovar)",
        r"pasos para",
        r"que (hago|debo hacer) (si|para)",
        r"como pagar",
        r"plazo.*pagar",
        r"donde pago",
    ]),
]


class QuestionClassifier:
    """
    Clasifica la pregunta del usuario en una categoría de intención.

    Usa reglas léxicas (regex) sobre el texto normalizado.
    No consume cuota de API ni requiere el LLM.

    Uso:
        clf = QuestionClassifier()
        tipo = clf.classify("¿cuánto es la multa por no usar casco?")
        # → QuestionType.MULTA
    """

    def classify(self, query: str) -> QuestionType:
        """
        Retorna el tipo de pregunta detectado.

        Args:
            query: Pregunta original del usuario.

        Returns:
            QuestionType correspondiente, o GENERAL si no hay match.
        """
        normalized = query.lower().strip()
        # Remover tildes para mayor tolerancia
        normalized = self._remove_accents(normalized)

        for question_type, patterns in _PATTERNS:
            for pattern in patterns:
                if re.search(pattern, normalized):
                    return question_type

        return QuestionType.GENERAL

    def _remove_accents(self, text: str) -> str:
        replacements = {
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
            'ü': 'u', 'ñ': 'n',
        }
        for accented, plain in replacements.items():
            text = text.replace(accented, plain)
        return text
