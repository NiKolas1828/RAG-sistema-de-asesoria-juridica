import statistics
from typing import List, Dict, Any

class ContextBuilder:
    def __init__(self, max_tokens: int = 2000, min_similarity: float = 0.7):
        self.max_tokens = max_tokens
        self.min_similarity = min_similarity

    def build_context(self, search_results: Dict[str, Any], max_documents: int = 3) -> Dict[str, Any]:
        resultados = []
        if not search_results:
            return {
                "contexto_formateado": "",
                "documentos_usados": 0,
                "tokens_contexto": 0,
                "tokens_disponibles": self.max_tokens,
                "score_promedio": 0.0,
            }

        # Extraer lista de resultados
        if isinstance(search_results, dict) and "resultados" in search_results:
            resultados = search_results.get("resultados", [])
        elif isinstance(search_results, list):
            resultados = search_results
        else:
            raise ValueError("Formato de `search_results` no reconocido")

        # Filtrar y ordenar
        filtered = self._filter_by_score(resultados)
        # Tomar top según max_documents
        top_docs = filtered[:max_documents]

        # Formatear documentos
        formatted_docs = [self._format_document(d, i + 1) for i, d in enumerate(top_docs)]

        header = "NORMATIVAS JURÍDICAS RELEVANTES\n" + ("═" * 60) + "\n\n"
        cuerpo = "\n\n".join(formatted_docs)
        contexto = header + cuerpo

        # Ajustar por límite de tokens (puede truncar texto o eliminar documentos)
        contexto_ajustado, tokens_used = self._validate_token_limit(contexto, top_docs)

        scores = [d.get("similitud", 0.0) for d in top_docs]
        score_prom = round(float(statistics.mean(scores)) if scores else 0.0, 4)

        return {
            "contexto_formateado": contexto_ajustado,
            "documentos_usados": len(top_docs),
            "tokens_contexto": tokens_used,
            "tokens_disponibles": max(0, self.max_tokens - tokens_used),
            "score_promedio": score_prom,
            "chunks_seleccionados": top_docs,
        }


    def _filter_by_score(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not results:
            return []

        # Asegurar que cada item tenga la clave 'similitud'
        for r in results:
            if "similitud" not in r:
                r.setdefault("similitud", 0.0)

        ordenados = sorted(results, key=lambda x: x.get("similitud", 0.0), reverse=True)
        filtrados = [r for r in ordenados if r.get("similitud", 0.0) >= self.min_similarity]

        # Fallback: si no hay resultados por encima del umbral, devolver los top por similitud
        if not filtrados:
            return ordenados
        return filtrados

    def _format_document(self, document: Dict[str, Any], index: int) -> str:
        texto = document.get("texto", "")
        metadata = document.get("metadata", {}) or {}
        similitud = document.get("similitud", document.get("score", 0.0))

        articulo = metadata.get("articulo") if isinstance(metadata, dict) else None
        fuente = metadata.get("fuente") if isinstance(metadata, dict) else None

        header = f"[{index}] {fuente or 'Fuente desconocida'} | {articulo or 'Artículo/Sección'} | Relevancia: {round(similitud * 100, 2)}%"
        separator = "-" * 60

        # Normalizar texto: eliminar saltos de línea múltiples
        cuerpo = " ".join(texto.split())

        # No truncamos aquí; la validación de tokens se encargará si es necesario
        formatted = f"{header}\n{separator}\n{cuerpo}"
        return formatted

    def _estimate_tokens(self, text: str) -> int:
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            # fallback: aproximación simple (1 token ~ 0.75 palabras)
            words = len(text.split())
            return int(words / 0.75)

    def _validate_token_limit(self, contexto: str, top_docs: List[Dict[str, Any]]):
        
        tokens = self._estimate_tokens(contexto)
        if tokens <= self.max_tokens:
            return contexto, tokens

        # Convertir cada documento a su texto para poder truncar por separado
        docs_texts = [d.get("texto", "") for d in top_docs]

        # Intentar truncar textos proporcionalmente hasta cumplir el límite
        # Empezamos permitiendo 100% y reducimos en pasos
        allowed_fraction = 1.0
        min_fraction = 0.2

        while tokens > self.max_tokens and allowed_fraction >= min_fraction:
            parts = []
            for i, d in enumerate(top_docs):
                text = docs_texts[i]
                keep_len = max(50, int(len(text) * allowed_fraction))
                truncated = text[:keep_len]
                meta = d.get("metadata", {}) or {}
                articulo = meta.get("articulo") if isinstance(meta, dict) else None
                fuente = meta.get("fuente") if isinstance(meta, dict) else None
                header = f"[{i+1}] {fuente or 'Fuente desconocida'} | {articulo or 'Artículo/Sección'} | Relevancia: {round(d.get('similitud',0.0)*100,2)}%"
                separator = "-" * 60
                parts.append(f"{header}\n{separator}\n{' '.join(truncated.split())}")

            new_context = "NORMATIVAS JURÍDICAS RELEVANTES\n" + ("═" * 60) + "\n\n" + "\n\n".join(parts)
            tokens = self._estimate_tokens(new_context)
            if tokens <= self.max_tokens:
                return new_context, tokens

            allowed_fraction -= 0.15

        # Si truncando no alcanza, eliminar documentos menos relevantes uno por uno
        remaining = top_docs.copy()
        while tokens > self.max_tokens and remaining:
            remaining.pop()  # elimina el menos relevante (último)
            parts = []
            for i, d in enumerate(remaining):
                text = d.get("texto", "")
                truncated = text[: max(100, int(len(text) * 0.5))]
                meta = d.get("metadata", {}) or {}
                articulo = meta.get("articulo") if isinstance(meta, dict) else None
                fuente = meta.get("fuente") if isinstance(meta, dict) else None
                header = f"[{i+1}] {fuente or 'Fuente desconocida'} | {articulo or 'Artículo/Sección'} | Relevancia: {round(d.get('similitud',0.0)*100,2)}%"
                separator = "-" * 60
                parts.append(f"{header}\n{separator}\n{' '.join(truncated.split())}")

            new_context = "NORMATIVAS JURÍDICAS RELEVANTES\n" + ("═" * 60) + "\n\n" + "\n\n".join(parts)
            tokens = self._estimate_tokens(new_context)
            if tokens <= self.max_tokens:
                return new_context, tokens

        # Si aún así no cabe, devolver el texto más relevante truncado severamente
        if top_docs:
            top = top_docs[0]
            text = top.get("texto", "")
            truncated = text[:1000]
            meta = top.get("metadata", {}) or {}
            articulo = meta.get("articulo") if isinstance(meta, dict) else None
            fuente = meta.get("fuente") if isinstance(meta, dict) else None
            header = f"[1] {fuente or 'Fuente desconocida'} | {articulo or 'Artículo/Sección'} | Relevancia: {round(top.get('similitud',0.0)*100,2)}%"
            separator = "-" * 60
            final = header + "\n" + separator + "\n" + " ".join(truncated.split())
            final_context = "NORMATIVAS JURÍDICAS RELEVANTES\n" + ("═" * 60) + "\n\n" + final
            tokens = self._estimate_tokens(final_context)
            return final_context, tokens

        return "", 0