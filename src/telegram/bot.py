# src/telegram/bot.py
# ============================================================
# Bot de Telegram — Sistema RAG de Normas de Tránsito Colombia
#
# Soporta dos modos de ejecución:
#   - Polling  (desarrollo local):  WEBHOOK_URL no definida en .env
#   - Webhook  (producción/Render): WEBHOOK_URL=https://tu-app.onrender.com
#
# Características:
#   - Memoria multi-turno por chat_id (últimos 5 turnos)
#   - Indicador "escribiendo..." mientras procesa
#   - División automática de respuestas largas (límite 4096 chars)
#   - Comandos: /start, /ayuda, /limpiar
# ============================================================

import os
import logging
import asyncio
from typing import Dict

from dotenv import load_dotenv
from telegram import Update, BotCommand
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest

from src.retrieval.rag_pipeline import RAGPipeline
from src.generation.response_generator import ResponseGenerator
from src.memory.conversation_memory import ConversationMemory

load_dotenv()

logger = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────
TOKEN       = os.getenv("TELEGRAM_BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")        # Vacío → polling local
WEBHOOK_PORT = int(os.getenv("PORT", "8443"))      # Render asigna PORT

# Máx turnos de memoria por usuario (par pregunta/respuesta)
MAX_MEMORY_TURNS = 5

# Límite de caracteres por mensaje de Telegram
TG_MAX_CHARS = 4096

# ─── Estado global (en memoria RAM por proceso) ───────────────
# Mapa: chat_id → ConversationMemory
_memorias: Dict[int, ConversationMemory] = {}

# Pipeline y generador compartidos (singleton por proceso)
_pipeline:  RAGPipeline        = None
_generator: ResponseGenerator  = None


def _get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        logger.info("[Bot] Inicializando RAGPipeline...")
        _pipeline = RAGPipeline()
    return _pipeline


def _get_generator() -> ResponseGenerator:
    global _generator
    if _generator is None:
        logger.info("[Bot] Inicializando ResponseGenerator...")
        _generator = ResponseGenerator()
    return _generator


def _get_memory(chat_id: int) -> ConversationMemory:
    """Retorna la memoria del usuario, creándola si no existe."""
    if chat_id not in _memorias:
        _memorias[chat_id] = ConversationMemory(max_turns=MAX_MEMORY_TURNS)
    return _memorias[chat_id]


def _split_message(text: str, max_len: int = TG_MAX_CHARS) -> list[str]:
    """
    Divide un texto largo en partes respetando el límite de Telegram.
    Intenta cortar en saltos de línea para no partir frases a la mitad.
    """
    if len(text) <= max_len:
        return [text]

    parts = []
    while text:
        if len(text) <= max_len:
            parts.append(text)
            break
        # Buscar el último salto de línea dentro del límite
        cut = text.rfind("\n", 0, max_len)
        if cut == -1:
            cut = max_len
        parts.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return parts


async def _send_long_message(update: Update, text: str) -> None:
    """Envía un mensaje, partiéndolo si supera el límite de Telegram."""
    partes = _split_message(text)
    for i, parte in enumerate(partes):
        try:
            await update.message.reply_text(parte)
        except BadRequest as e:
            logger.warning(f"[Bot] Error enviando parte {i+1}: {e}")
            await update.message.reply_text(
                "⚠️ Hubo un error enviando parte de la respuesta."
            )


# ─── Handlers ────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Mensaje de bienvenida."""
    nombre = update.effective_user.first_name or "ciudadano"
    texto = (
        f"👋 Hola, *{nombre}*\\. Soy el asistente de normas de tránsito de Colombia\\.\n\n"
        "Puedo responder preguntas como:\n"
        "• *¿Cuánto es la multa por no usar casco?*\n"
        "• *¿Qué documentos necesito para renovar la licencia?*\n"
        "• *¿Qué dice la ley sobre el SOAT?*\n\n"
        "Simplemente escríbeme tu pregunta\\. 🚦\n\n"
        "Comandos disponibles:\n"
        "/ayuda — Ver esta ayuda\n"
        "/limpiar — Reiniciar el historial de conversación"
    )
    await update.message.reply_text(texto, parse_mode=ParseMode.MARKDOWN_V2)


async def cmd_ayuda(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Muestra la ayuda."""
    await cmd_start(update, context)


async def cmd_limpiar(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Limpia el historial de conversación del usuario."""
    chat_id = update.effective_chat.id
    memoria = _get_memory(chat_id)
    memoria.clear()
    await update.message.reply_text(
        "🧹 Historial limpiado\\. Empezamos de cero\\.",
        parse_mode=ParseMode.MARKDOWN_V2,
    )


async def handle_pregunta(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handler principal: procesa la pregunta del usuario con el pipeline RAG.
    """
    chat_id  = update.effective_chat.id
    pregunta = update.message.text.strip()

    if not pregunta:
        return

    # Mostrar indicador "escribiendo..." mientras procesa
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    memoria   = _get_memory(chat_id)
    pipeline  = _get_pipeline()
    generator = _get_generator()

    try:
        # Ejecutar el pipeline RAG en un thread pool para no bloquear el event loop
        loop = asyncio.get_event_loop()

        rag_output = await loop.run_in_executor(
            None,
            lambda: pipeline.run(pregunta, k=10, max_documents=8)
        )

        historial = memoria.get_history() if not memoria.is_empty() else None

        resultado = await loop.run_in_executor(
            None,
            lambda: generator.generate(rag_output, history=historial)
        )

        respuesta = resultado.get("respuesta", "").strip()
        status    = resultado.get("status", "error")

        if status == "éxito" and respuesta:
            # Guardar en memoria del usuario
            memoria.add_turn(pregunta, respuesta)

            # Añadir pie con modelo usado (opcional, para debug)
            modelo = resultado.get("modelo_usado", "")
            if memoria.turn_count() > 1:
                respuesta += f"\n\n_🧠 Contexto: {memoria.turn_count()} turnos_"

            await _send_long_message(update, respuesta)

        else:
            await update.message.reply_text(
                "No encontré información suficiente sobre eso en las normas consultadas\\. "
                "Te recomiendo contactar al organismo de tránsito de tu municipio\\.",
                parse_mode=ParseMode.MARKDOWN_V2,
            )

    except Exception as e:
        logger.error(f"[Bot] Error procesando pregunta de {chat_id}: {e}", exc_info=True)
        await update.message.reply_text(
            "⚠️ Ocurrió un error interno\\. Por favor intenta de nuevo en unos segundos\\.",
            parse_mode=ParseMode.MARKDOWN_V2,
        )


# ─── Entrypoint ──────────────────────────────────────────────

def main() -> None:
    if not TOKEN:
        raise EnvironmentError(
            "TELEGRAM_BOT_TOKEN no encontrado en .env. "
            "Crea el bot en @BotFather y configura la variable de entorno."
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    app = ApplicationBuilder().token(TOKEN).build()

    # Registrar comandos
    app.add_handler(CommandHandler("start",   cmd_start))
    app.add_handler(CommandHandler("ayuda",   cmd_ayuda))
    app.add_handler(CommandHandler("limpiar", cmd_limpiar))

    # Handler de mensajes de texto
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_pregunta)
    )

    if WEBHOOK_URL:
        # ── Modo Webhook (producción en Render.com) ──────────────
        logger.info(f"[Bot] Iniciando en modo WEBHOOK → {WEBHOOK_URL}")
        app.run_webhook(
            listen="0.0.0.0",
            port=WEBHOOK_PORT,
            webhook_url=f"{WEBHOOK_URL}/webhook/{TOKEN}",
            url_path=f"/webhook/{TOKEN}",
        )
    else:
        # ── Modo Polling (desarrollo local) ──────────────────────
        logger.info("[Bot] Iniciando en modo POLLING (local)...")
        app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()