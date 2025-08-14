import logging
from dotenv import load_dotenv
import os
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from rag_backend.generator import generate_answer  # импортируем твой генератор

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Команда /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Задай мне вопрос про классификацию с дрона.")

# Обработка текста от пользователя
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_question = update.message.text
    await update.message.reply_text("Ищу ответ, подожди...")

    # Генерируем ответ (можно вынести в отдельный поток/задачу для асинхронности)
    answer = generate_answer(user_question)

    await update.message.reply_text(answer)

if __name__ == '__main__':
    load_dotenv('D:/uav_rag_project/uav-rag/.env')

    TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Бот запущен...")
    app.run_polling()
