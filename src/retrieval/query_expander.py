# src/retrieval/query_expander.py
# ============================================================
# Expansor de Consultas (Multi-Query Retrieval)
# Usa Groq para reescribir la pregunta original en variaciones
# semánticas con vocabulario legal para mejorar el retrieval.
# ============================================================

import logging
from typing import List

from src.generation.groq_client import GroqClient
from src.generation.llm_config import GroqConfig

logger = logging.getLogger(__name__)

EXPANDER_SYSTEM_PROMPT = """
Eres un experto en tránsito y transporte de Colombia.
Tu objetivo es reescribir la pregunta del usuario en 3 variaciones distintas para usarlas en un motor de búsqueda vectorial.
Debes usar vocabulario jurídico o técnico (ej: "comparendo", "infracción", "SMLDV", "UVT", "SOAT", "licencia", "caducidad").

REGLAS:
1. Retorna ÚNICAMENTE las 3 variaciones, separadas por saltos de línea.
2. NO incluyas números de viñeta (1., 2., 3.).
3. NO incluyas saludos ni explicaciones.
4. Mantén las variaciones cortas y directas.
5. MUY IMPORTANTE: Si la pregunta trata sobre una multa/infracción específica y conoces su código exacto en Colombia según la Resolución 3027 (ej. C.24 para casco, D.02 para SOAT, D.04 para semáforo, C.14 para pico y placa), INCLUYE el código exacto en al menos una de las variaciones.

Ejemplo de entrada:
"cuánto vale la multa por andar sin casco"

Ejemplo de salida:
valor infracción C.24 transitar sin casco protector motocicleta
sanción económica comparendo no portar casco
multa código de infracción casco conductor pasajero moto
"""

class QueryExpander:
    def __init__(self):
        # Usar Groq con un prompt de sistema específico para expansión
        config = GroqConfig(
            system_instruction=EXPANDER_SYSTEM_PROMPT,
            temperature=0.3, # Baja temperatura para que sea predecible
            max_tokens=150
        )
        self.client = GroqClient(config=config)

    def expand_query(self, original_query: str) -> List[str]:
        """
        Toma la pregunta original y devuelve una lista de queries
        incluyendo la original y 3 variaciones generadas por el LLM.
        """
        queries = [original_query]
        
        try:
            logger.debug(f"[QueryExpander] Expandiendo consulta: '{original_query}'")
            respuesta = self.client.generate(prompt=original_query)
            
            # Limpiar y parsear las líneas
            variaciones = [line.strip() for line in respuesta.split("\n") if line.strip()]
            
            # Evitar que el LLM meta viñetas como "1. " o "- "
            for i in range(len(variaciones)):
                var = variaciones[i]
                if var[0].isdigit() and var[1] in [".", ")"]:
                    var = var[2:].strip()
                elif var.startswith("- ") or var.startswith("* "):
                    var = var[2:].strip()
                variaciones[i] = var
                
            # Agregar solo las primeras 3 válidas
            queries.extend(variaciones[:3])
            logger.debug(f"[QueryExpander] Variaciones generadas: {variaciones[:3]}")
            
        except Exception as e:
            logger.warning(f"[QueryExpander] Error expandiendo consulta, se usará solo la original: {e}")
            
        return queries
