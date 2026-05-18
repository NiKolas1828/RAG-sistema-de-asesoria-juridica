from typing import Dict


class PromptBuilder:
    def __init__(self, system_instructions: str = None, response_buffer: int = 128):
        self.system_instructions = (
            system_instructions
            or (
                "Eres un asistente jurídico experto. Responde con precisión basándote SOLO en la"
                " información provista en el bloque de contexto. Cita las fuentes usando los índices"
                " provistos y, si la información no está en el contexto, indícalo explícitamente."
            )
        )
        self.response_buffer = response_buffer

    def build_prompt(
        self,
        contexto_formateado: str,
        query: str,
        additional_instructions: str = "",
        max_prompt_tokens: int = 2000,
    ) -> Dict:
        """Arma el prompt final para el LLM y devuelve conteo estimado de tokens.

        No realiza llamadas al LLM; solo concatena y estima tokens.
        """
        system_block = f"[SYSTEM]\n{self.system_instructions}\n"
        if additional_instructions:
            system_block += f"{additional_instructions}\n"

        instruction_block = (
            "Responde de forma concisa y estructurada. Incluye: respuesta breve, explicación y referencias."
        )

        context_block = "[CONTEXT]\n" + (contexto_formateado or "") + "\n"
        question_block = f"[QUESTION]\n{query}\n"

        prompt = "\n".join([system_block, instruction_block, context_block, question_block])

        tokens_prompt = self._estimate_tokens(prompt)
        tokens_available_for_response = max(0, max_prompt_tokens - tokens_prompt - self.response_buffer)

        return {
            "prompt": prompt,
            "tokens_prompt": tokens_prompt,
            "tokens_available_for_response": tokens_available_for_response,
        }

    def _estimate_tokens(self, text: str) -> int:
        try:
            import tiktoken  # type: ignore[import-not-found]

            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            words = len(text.split())
            return int(words / 0.75)
