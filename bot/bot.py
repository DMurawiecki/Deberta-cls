import os

import numpy as np
from dotenv import load_dotenv
from rag_engine import ANSWER
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from transformers import AutoTokenizer
from tritonclient.http import InferenceServerClient, InferInput

print("bot started")

load_dotenv()
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TRITON_URL = os.environ.get("TRITON_URL", "localhost:8000")
MODEL_NAME = os.environ.get("TRITON_MODEL_NAME", "my_bert_mc")

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN not set in environment.")

TOKENIZER = AutoTokenizer.from_pretrained(
    "./models/model/hf_pretrained", local_files_only=True
)
TRITON_CLIENT = InferenceServerClient(url=TRITON_URL, verbose=False)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hello! I'm Science helping checked bot. Ask me anything about science!"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    if not isinstance(user_text, str):
        await update.message.reply_text("Internal error: user_text is not a string.")
        return

    await update.message.chat.send_action(action="typing")

    try:
        options = []
        while len(options) < 5:
            try:
                resp = ANSWER(user_text)
                resp = str(resp)
                if isinstance(resp, str) and resp not in options:
                    options.append(resp)
            except Exception:
                continue

        pairs = [(user_text, opt) for opt in options]
        first_texts = [pair[0] for pair in pairs]
        second_texts = [pair[1] for pair in pairs]

        pairs_for_tokenizer = list(zip(first_texts, second_texts))
        encoded = TOKENIZER.batch_encode_plus(
            pairs_for_tokenizer,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=64,
        )

        input_ids = np.array(encoded["input_ids"], dtype=np.int64)
        attention_mask = np.array(encoded["attention_mask"], dtype=np.int64)
        input_ids = np.expand_dims(input_ids, axis=0)  # (1, num_choices, seq_len)
        attention_mask = np.expand_dims(attention_mask, axis=0)

        triton_inputs = []
        inp_ids = InferInput("input_ids", input_ids.shape, "INT64")
        inp_ids.set_data_from_numpy(input_ids)
        triton_inputs.append(inp_ids)

        inp_mask = InferInput("attention_mask", attention_mask.shape, "INT64")
        inp_mask.set_data_from_numpy(attention_mask)
        triton_inputs.append(inp_mask)

        if "token_type_ids" in encoded:
            token_type_ids = np.array(encoded["token_type_ids"], dtype=np.int64)
            token_type_ids = np.expand_dims(token_type_ids, axis=0)
            inp_tti = InferInput("token_type_ids", token_type_ids.shape, "INT64")
            inp_tti.set_data_from_numpy(token_type_ids)
            triton_inputs.append(inp_tti)
        print("start to triton")

        results = TRITON_CLIENT.infer(model_name=MODEL_NAME, inputs=triton_inputs)
        logits = results.as_numpy("logits")
        best_index = int(np.argmax(logits, axis=1)[0])
        best_answer = options[best_index]

        await update.message.reply_text(best_answer)

    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        await update.message.reply_text(
            f"Oops! Something went wrong.\n\n{e}\n{tb[:1000]}"
        )


if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()
