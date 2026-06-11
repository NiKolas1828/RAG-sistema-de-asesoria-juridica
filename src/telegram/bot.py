# src/telegram/bot.py
# ============================================================
# Bot de Telegram — Sistema RAG de Normas de Tránsito Colombia
#
# Soporta dos modos de ejecución:
#   - Polling  (desarrollo local):  WEBHOOK_URL no definida en .env
#   - Webhook  (producción/Render): WEBHOOK_URL=https://tu-app.onrender.com
#
# Características:
#   ✅ Memoria persistente por usuario (SQLite — sobrevive reinicios)
#   ✅ Botones de feedback 👍/👎 en cada respuesta
#   ✅ Sugerencias de preguntas relacionadas al final de cada respuesta
#   ✅ Indicador "escribiendo..." mientras procesa
#   ✅ División automática de respuestas largas (límite 4096 chars)
#   ✅ Formatos adaptativos (tabla multas, checklist, comparativo, etc.)
#   ✅ Comandos: /start, /ayuda, /limpiar, /historial, /stats
# ============================================================

import os
import logging
import asyncio
import re
import json
from typing import Dict, Optional

from dotenv import load_dotenv
from telegram import Update, BotCommand, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest

from src.retrieval.rag_pipeline import RAGPipeline
from src.generation.response_generator import ResponseGenerator
from src.memory.persistent_memory import (
    PersistentMemory,
    init_db,
    save_feedback,
    get_feedback_stats,
)

load_dotenv()

logger = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────
TOKEN        = os.getenv("TELEGRAM_BOT_TOKEN")
WEBHOOK_URL  = os.getenv("WEBHOOK_URL", "")       # Vacío → polling local
WEBHOOK_PORT = int(os.getenv("PORT", "8443"))      # Render asigna PORT

# Límite de caracteres por mensaje de Telegram
TG_MAX_CHARS = 4096

# Pipeline y generador compartidos (singleton por proceso)
_pipeline:  Optional[RAGPipeline]       = None
_generator: Optional[ResponseGenerator] = None


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


# ─── Sugerencias de preguntas relacionadas ───────────────────
# Mapa tipo de pregunta → lista de sugerencias contextualmente relevantes
_SUGERENCIAS = {
    "multa": [
        "¿Puedo pagar la multa en cuotas?",
        "¿Qué pasa si no pago la multa a tiempo?",
        "¿Cómo impugnar un comparendo injusto?",
    ],
    "requisitos": [
        "¿Dónde se hace el trámite?",
        "¿Cuánto tiempo tarda el proceso?",
        "¿Qué pasa si me falta algún documento?",
    ],
    "uso_correcto": [
        "¿Cuánto es la multa si no lo uso correctamente?",
        "¿Qué marca o tipo es obligatorio por ley?",
        "¿Hay excepciones a esta obligación?",
    ],
    "comparativo": [
        "¿Cuál es más económico en términos de multas?",
        "¿Cuáles son los requisitos de cada uno?",
        "¿Hay normas específicas para cada tipo en vías urbanas?",
    ],
    "infraccion": [
        "¿Cuánto tiempo tengo para pagar con descuento?",
        "¿Cómo presento un recurso de reposición?",
        "¿El comparendo afecta mi licencia de conducción?",
    ],
    "procedimiento": [
        "¿Cuáles son los documentos necesarios?",
        "¿Cuánto cuesta el trámite?",
        "¿Se puede hacer el trámite en línea?",
    ],
    "general": [
        "¿Cuáles son las infracciones más comunes?",
        "¿Cómo consulto mis comparendos pendientes?",
        "¿Qué es el RUNT y para qué sirve?",
    ],
}


def _get_sugerencias(question_type: str) -> list:
    return _SUGERENCIAS.get(question_type, _SUGERENCIAS["general"])


# ─── Formateo HTML ────────────────────────────────────────────

def _markdown_to_html(text: str) -> str:
    """
    Convierte las plantillas Markdown de los formatos adaptativos a HTML de Telegram.
    Maneja tablas, negritas, itálicas, listas y emojis.
    """
    # Escapar caracteres HTML obligatorios
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # Convertir tablas Markdown a formato texto estructurado con monospace
    def format_table(match):
        lines = match.group(0).strip().split("\n")
        result_lines = []
        for line in lines:
            # Saltar líneas separadoras (|---|---|)
            if re.match(r'^\|[\s\-|]+\|$', line):
                continue
            # Limpiar celdas
            cells = [c.strip() for c in line.strip("|").split("|")]
            result_lines.append("  ".join(cells))
        return "<pre>" + "\n".join(result_lines) + "</pre>"

    text = re.sub(r'(\|.+\|\n)+', format_table, text)

    # Negritas: **texto** → <b>texto</b>
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)

    # Itálicas con asterisco simple: *texto* → <i>texto</i>
    text = re.sub(r'(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)', r'<i>\1</i>', text)

    # Itálicas con guion bajo: _texto_ → <i>texto</i>
    text = re.sub(r'(?<!\w)_(.*?)(?<!\w)_', r'<i>\1</i>', text)

    # Código inline: `texto` → <code>texto</code>
    text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)

    # Enlaces: [texto](url) → <a href="url">texto</a>
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', text)

    return text


def _split_message(text: str, max_len: int = TG_MAX_CHARS) -> list:
    """Divide un texto largo en partes respetando el límite de Telegram."""
    if len(text) <= max_len:
        return [text]
    parts = []
    while text:
        if len(text) <= max_len:
            parts.append(text)
            break
        cut = text.rfind("\n", 0, max_len)
        if cut == -1:
            cut = max_len
        parts.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return parts


async def _send_response(
    update: Update,
    respuesta: str,
    query: str,
    question_type: str,
    user_id: int,
) -> None:
    """
    Envía la respuesta al usuario con:
    - Formato HTML adaptativo
    - Sugerencias de preguntas relacionadas
    - Botones de feedback 👍/👎
    """
    html_text = _markdown_to_html(respuesta)

    # Añadir sugerencias de preguntas relacionadas
    sugerencias = _get_sugerencias(question_type)
    sugerencias_text = "\n\n💡 <b>También puedes preguntar:</b>\n" + "\n".join(
        f"• <i>{s}</i>" for s in sugerencias[:3]
    )
    html_text += sugerencias_text

    partes = _split_message(html_text)

    # Construir teclado de feedback
    feedback_data = json.dumps({
        "q": query[:200],   # truncar para no exceder límite de callback_data (64 bytes)
        "uid": user_id,
    }, ensure_ascii=False)[:60]  # Telegram limita callback_data a 64 bytes

    teclado_feedback = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("👍 Útil", callback_data=f"fb:1:{user_id}"),
            InlineKeyboardButton("👎 Mejorar", callback_data=f"fb:0:{user_id}"),
        ]
    ])

    # Enviar todas las partes; el feedback solo va en la última
    for i, parte in enumerate(partes):
        is_last = (i == len(partes) - 1)
        try:
            await update.message.reply_text(
                parte,
                parse_mode=ParseMode.HTML,
                reply_markup=teclado_feedback if is_last else None,
            )
        except BadRequest as e:
            logger.warning(f"[Bot] Error HTML en parte {i+1}: {e}. Enviando en texto plano.")
            plain = re.sub(r'<[^>]*>', '', parte)
            await update.message.reply_text(
                plain,
                reply_markup=teclado_feedback if is_last else None,
            )


# ─── Handlers ────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Mensaje de bienvenida."""
    nombre = update.effective_user.first_name or "ciudadano"
    texto = (
        f"👋 Hola, <b>{nombre}</b>. Soy tu asistente de normas de tránsito de Colombia.\n\n"
        "Puedo responder preguntas como:\n"
        "• <b>¿Cuánto es la multa por no usar casco?</b>\n"
        "• <b>¿Qué documentos necesito para renovar la licencia?</b>\n"
        "• <b>Me pusieron un comparendo, ¿qué hago?</b>\n"
        "• <b>¿Qué diferencia hay entre moto y bicicleta en la ley?</b>\n\n"
        "Simplemente escríbeme tu pregunta. 🚦\n\n"
        "<b>Comandos disponibles:</b>\n"
        "/ayuda — Ver esta ayuda\n"
        "/limpiar — Reiniciar el historial de conversación\n"
        "/historial — Ver tus últimas 3 preguntas"
    )
    await update.message.reply_text(texto, parse_mode=ParseMode.HTML)


async def cmd_ayuda(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await cmd_start(update, context)


async def cmd_limpiar(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Limpia el historial de conversación del usuario."""
    user_id = update.effective_user.id
    memoria = PersistentMemory(user_id)
    memoria.clear()
    await update.message.reply_text(
        "🧹 Historial limpiado. Empezamos de cero.",
        parse_mode=ParseMode.HTML,
    )


async def cmd_historial(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Muestra las últimas 3 preguntas del usuario."""
    user_id = update.effective_user.id
    memoria = PersistentMemory(user_id)
    preguntas = memoria.get_recent_questions(n=3)

    if not preguntas:
        await update.message.reply_text(
            "📭 Aún no tienes preguntas en tu historial.",
            parse_mode=ParseMode.HTML,
        )
        return

    texto = "<b>📜 Tus últimas preguntas:</b>\n\n"
    for i, q in enumerate(preguntas, 1):
        texto += f"{i}. {q}\n"
    await update.message.reply_text(texto, parse_mode=ParseMode.HTML)


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Muestra estadísticas de feedback (solo para administradores)."""
    stats = get_feedback_stats()
    texto = (
        f"📊 <b>Estadísticas de feedback:</b>\n\n"
        f"• Total de valoraciones: {stats['total']}\n"
        f"• 👍 Útiles: {stats['utiles']}\n"
        f"• 👎 A mejorar: {stats['no_utiles']}\n"
        f"• Tasa de utilidad: {stats['tasa_utilidad_pct']}%"
    )
    await update.message.reply_text(texto, parse_mode=ParseMode.HTML)


async def handle_feedback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Procesa el feedback 👍/👎 de los botones inline."""
    query = update.callback_query
    await query.answer()  # Quitar el spinner del botón

    data = query.data  # formato: "fb:1:user_id" o "fb:0:user_id"
    parts = data.split(":")
    if len(parts) != 3 or parts[0] != "fb":
        return

    util = parts[1] == "1"
    user_id = int(parts[2])

    # Guardar en BD
    # Recuperar la última pregunta del usuario desde la BD
    memoria = PersistentMemory(user_id)
    preguntas = memoria.get_recent_questions(n=1)
    last_query = preguntas[0] if preguntas else "?"

    save_feedback(
        user_id=user_id,
        query=last_query,
        respuesta=query.message.text or "",
        util=util,
    )

    if util:
        await query.edit_message_reply_markup(reply_markup=None)
        await query.message.reply_text("¡Gracias! Me alegra haber sido útil. 😊")
    else:
        await query.edit_message_reply_markup(reply_markup=None)
        await query.message.reply_text(
            "Gracias por el comentario. Lo tendré en cuenta para mejorar. 🙏\n"
            "Puedes reformular tu pregunta con más detalle si quieres."
        )


async def _keep_typing(bot, chat_id, stop_event) -> None:
    """Mantiene el estado 'escribiendo...' activo hasta que stop_event se active."""
    while not stop_event.is_set():
        try:
            await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        except Exception:
            pass
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=4.0)
        except asyncio.TimeoutError:
            pass


async def handle_pregunta(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handler principal: procesa la pregunta con el pipeline RAG y responde
    con formato adaptativo, sugerencias y botones de feedback.
    """
    chat_id  = update.effective_chat.id
    user_id  = update.effective_user.id
    pregunta = update.message.text.strip()

    if not pregunta:
        return

    # Iniciar indicador de escritura
    stop_event  = asyncio.Event()
    typing_task = asyncio.create_task(
        _keep_typing(context.bot, chat_id, stop_event)
    )

    memoria   = PersistentMemory(user_id)
    pipeline  = _get_pipeline()
    generator = _get_generator()

    try:
        loop = asyncio.get_event_loop()

        historial = memoria.get_history() if not memoria.is_empty() else None

        rag_output = await loop.run_in_executor(
            None,
            lambda: pipeline.run(pregunta, k=10, max_documents=8, history=historial)
        )

        resultado = await loop.run_in_executor(
            None,
            lambda: generator.generate(rag_output, history=historial)
        )

        respuesta     = resultado.get("respuesta", "").strip()
        status        = resultado.get("status", "error")
        question_type = resultado.get("question_type", "general")

        if status == "éxito" and respuesta:
            # Guardar en memoria persistente
            memoria.add_turn(pregunta, respuesta)

            stop_event.set()
            await typing_task

            await _send_response(
                update=update,
                respuesta=respuesta,
                query=pregunta,
                question_type=question_type,
                user_id=user_id,
            )

        else:
            stop_event.set()
            await typing_task
            await update.message.reply_text(
                "No encontré información suficiente sobre eso en las normas consultadas. "
                "Te recomiendo contactar al organismo de tránsito de tu municipio o "
                "consultar directamente el Código Nacional de Tránsito (Ley 769 de 2002).",
                parse_mode=ParseMode.HTML,
            )

    except Exception as e:
        stop_event.set()
        await typing_task
        logger.error(f"[Bot] Error procesando pregunta de {chat_id}: {e}", exc_info=True)
        await update.message.reply_text(
            "⚠️ Ocurrió un error interno. Por favor intenta de nuevo en unos segundos.",
            parse_mode=ParseMode.HTML,
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

    # Inicializar la base de datos de memoria persistente
    init_db()

    app = ApplicationBuilder().token(TOKEN).build()

    # Registrar comandos
    app.add_handler(CommandHandler("start",     cmd_start))
    app.add_handler(CommandHandler("ayuda",     cmd_ayuda))
    app.add_handler(CommandHandler("limpiar",   cmd_limpiar))
    app.add_handler(CommandHandler("historial", cmd_historial))
    app.add_handler(CommandHandler("stats",     cmd_stats))

    # Handler de feedback (botones inline)
    app.add_handler(CallbackQueryHandler(handle_feedback, pattern=r"^fb:"))

    # Handler de mensajes de texto
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_pregunta)
    )

    if WEBHOOK_URL:
        logger.info(f"[Bot] Iniciando en modo WEBHOOK → {WEBHOOK_URL}")
        app.run_webhook(
            listen="0.0.0.0",
            port=WEBHOOK_PORT,
            webhook_url=f"{WEBHOOK_URL}/webhook/{TOKEN}",
            url_path=f"/webhook/{TOKEN}",
        )
    else:
        logger.info("[Bot] Iniciando en modo POLLING (local)...")
        app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()