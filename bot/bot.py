import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from rag_engine import ANSWER
import pandas as pd
import sys

sys.path.append("/Users/tadeuskostusko/Documents/Deberta-cls")
sys.path.append("/Users/tadeuskostusko/Documents/Deberta-cls/pl_scripts")


load_dotenv()

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN not set in environment.")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hello! I'm Science helping checked bot. Ask me anything about science! For example: Who was the author of the The Special and General Theory book?"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    from pl_scripts.pl_infer import main

    user_text = update.message.text
    await update.message.chat.send_action(action="typing")
    try:
        answers = [0, user_text]
        columns = ["id", "prompt", "A", "B", "C", "D", "E"]
        count_ans = 0
        while count_ans < 5:
            try:
                resp = ANSWER(user_text)
                answers.append(resp)
                count_ans += 1
            except Exception:
                pass
        df = pd.DataFrame([answers], columns=columns)
        df.to_csv("/Users/tadeuskostusko/Documents/Deberta-cls/data_store/test.csv")
        main()
        best_ans = pd.read_csv(
            "/Users/tadeuskostusko/Documents/Deberta-cls/data_store/outputs.csv"
        )
        response_final = best_ans.iloc[0, 0][0]
        await update.message.reply_text(str(df[response_final][0]))
    except Exception:
        await update.message.reply_text(
            "Ops! Something went wrong. Please try again later."
        )


if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()
