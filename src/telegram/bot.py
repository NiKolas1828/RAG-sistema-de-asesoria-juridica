import os
import requests

from dotenv import load_dotenv

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

API_URL = "http://localhost:8000/consulta"


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):

    mensaje = (
        "Bienvenido al asistente de normas de tránsito.\n\n"
        "Puedes hacer cualquier pregunta relacionada a normas de tránsito en Colombia"
    )

    await update.message.reply_text(mensaje)


async def consulta(update: Update, context: ContextTypes.DEFAULT_TYPE):

    pregunta = update.message.text

    try:

        response = requests.post(
            API_URL,
            json={
                "question": pregunta,
                "verbose": False
            },
            timeout=60
        )

        response.raise_for_status()

        data = response.json()

        respuesta = data.get(
            "answer",
            "No se obtuvo respuesta."
        )

        await update.message.reply_text(respuesta)

    except Exception:

        await update.message.reply_text(
            "No fue posible conectar con el sistema."
        )


def main():

    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(
        CommandHandler("start", start)
    )

    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            consulta
        )
    )

    print("Bot iniciado...")

    app.run_polling()


if __name__ == "__main__":
    main()