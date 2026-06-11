# src/retrieval/domain_guard.py
# ============================================================
# Domain Guard — Filtro de Consultas Fuera del Dominio
#
# Impide que el agente intente responder preguntas que NO tienen
# que ver con normas de tránsito colombianas.
#
# Estrategia de dos capas:
#   1. Capa léxica (0 ms): lista negra de temas claramente fuera del dominio.
#   2. Capa semántica (basada en el score del top resultado RAG):
#      si el mejor chunk tiene similitud < MIN_SIMILARITY_THRESHOLD,
#      la pregunta está fuera del dominio de los datos indexados.
#
# La capa léxica se aplica ANTES del retrieval (ahorra tiempo).
# La capa semántica se aplica DESPUÉS del retrieval (usa resultados ya calculados).
# ============================================================

import re
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# ─── Configuración ────────────────────────────────────────────

# Umbral de similitud semántica mínima para considerar que el
# resultado es relevante. Si el top-1 está por debajo → fuera de dominio.
# Calibrado en 0.20 para reducir falsos positivos.
MIN_SIMILARITY_THRESHOLD = 0.10

# Palabras clave que indican temas CLARAMENTE fuera del dominio
# de normas de tránsito colombianas.
_TEMAS_FUERA_DOMINIO = [
    # Cocina / recetas
    r"\b(receta|cocinar|ingrediente|plato|comida|restaurante)\b",
    # Medicina / salud
    r"\b(medicamento|enfermedad|síntoma|doctor|hospital|diagnóstico|pastilla|vacuna)\b",
    # Tecnología general
    r"\b(programar|código fuente|javascript|python|sql|base de datos|app|software)\b",
    # Deportes
    r"\b(fútbol|baloncesto|béisbol|tenis|natación|gol|partido)\b",
    # Entretenimiento
    r"\b(película|pelicula|serie|canción|artista|actor|actriz|concierto)\b",
    # Geografía / política / historia general
    r"\b(capital de|presidente de|historia de|geografía de|geografia de)\b",
    # ¿Cuánto mide? / ¿Cuánto pesa? sin contexto de tránsito
    r"\b(cuánto mide|cuanto mide|cuánto pesa|cuanto pesa)\b(?!.*(?:vehiculo|moto|carro|camion))",
    # Finanzas
    r"\b(inversión|bolsa de valores|cripto|bitcoin|acción bursátil)\b",
    # Educación general
    r"\b(matemáticas|química|física|biología|literatura)\b",
]

# Palabras clave positivas del dominio — si están presentes, NO aplicar
# la capa léxica aunque haya coincidencia con _TEMAS_FUERA_DOMINIO
_PALABRAS_DOMINIO = [
    "tránsito", "transito", "moto", "vehículo", "vehiculo", "licencia",
    "conducir", "casco", "multa", "comparendo", "infracción", "infracion",
    "soat", "runt", "carretera", "vía", "via", "semáforo", "semafor",
    "peatón", "peaton", "autopista", "velocidad", "alcohol", "borracho",
    "embriaguez", "conductor", "ciclista", "bicicleta", "circular",
    "carnet", "registro", "placa", "revisión técnica", "policia",
    "código nacional de tránsito", "ley 769",
]

_COMPILED_BLACKLIST = [re.compile(p, re.IGNORECASE) for p in _TEMAS_FUERA_DOMINIO]
_COMPILED_WHITELIST = [re.compile(r'\b' + w + r'\b', re.IGNORECASE) for w in _PALABRAS_DOMINIO]

# Respuesta estándar fuera de dominio
RESPUESTA_FUERA_DOMINIO = (
    "🚦 Esta consulta no está relacionada con las normas de tránsito colombianas "
    "que tengo indexadas.\n\n"
    "Puedo ayudarte con preguntas como:\n"
    "• ¿Cuánto es la multa por no usar casco?\n"
    "• ¿Qué documentos necesito para matricular un vehículo?\n"
    "• ¿Qué dice la ley sobre el SOAT?\n"
    "• Me pusieron un comparendo, ¿qué hago?\n\n"
    "Por favor reformula tu consulta sobre normas de tránsito."
)


class DomainGuard:
    """
    Filtro de dominio de dos capas para el agente RAG de tránsito colombiano.

    Uso típico (integrado en RAGPipeline):

        guard = DomainGuard()

        # Capa 1: antes del retrieval
        es_valida, motivo = guard.check_lexical(query)
        if not es_valida:
            return fuera_de_dominio_response

        # ... hacer retrieval ...

        # Capa 2: después del retrieval
        es_valida, motivo = guard.check_semantic(search_results)
        if not es_valida:
            return fuera_de_dominio_response
    """

    def check_lexical(self, query: str) -> Tuple[bool, str]:
        """
        Capa 1 — Verificación léxica (sin costo computacional).

        Retorna (True, "") si la pregunta es del dominio.
        Retorna (False, motivo) si está claramente fuera del dominio.
        """
        query_lower = query.lower()

        # Si tiene palabras del dominio, siempre es válida
        for pattern in _COMPILED_WHITELIST:
            if pattern.search(query_lower):
                return True, ""

        # Verificar lista negra
        for pattern in _COMPILED_BLACKLIST:
            m = pattern.search(query_lower)
            if m:
                motivo = f"Tema detectado fuera del dominio: '{m.group()}'"
                logger.info(f"[DomainGuard] Consulta rechazada léxicamente. {motivo}")
                return False, motivo

        return True, ""

    def check_semantic(self, search_results: dict) -> Tuple[bool, str]:
        """
        Capa 2 — Verificación semántica basada en el score del top resultado.

        Retorna (True, "") si hay chunks suficientemente relevantes.
        Retorna (False, motivo) si todos los resultados tienen baja similitud.
        """
        resultados = search_results.get("resultados", [])
        if not resultados:
            motivo = "Sin resultados en la búsqueda semántica."
            logger.info(f"[DomainGuard] {motivo}")
            return False, motivo

        top_score = resultados[0].get("similitud", 0.0)
        if top_score < MIN_SIMILARITY_THRESHOLD:
            motivo = (
                f"Similitud máxima ({top_score:.3f}) "
                f"< umbral ({MIN_SIMILARITY_THRESHOLD}). "
                "Consulta probablemente fuera del dominio."
            )
            logger.info(f"[DomainGuard] Consulta rechazada semánticamente. {motivo}")
            return False, motivo

        return True, ""


# Instancia singleton
domain_guard = DomainGuard()
