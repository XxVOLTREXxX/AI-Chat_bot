import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, ContextTypes, filters
import pickle
import nltk
import spacy
import json
import random
from nltk.stem import WordNetLemmatizer

# Load models
nltk.download('punkt')
lemmatizer = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm")

with open("chat_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("responses.json", "r") as f:
    responses = json.load(f)

# Core response function
def chat_response(user_input):
    tokens = nltk.word_tokenize(user_input)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens]
    processed_input = " ".join(tokens)
    input_vector = vectorizer.transform([processed_input])
    predicted_tag = model.predict(input_vector)[0]
    return random.choice(responses[predicted_tag])

# Entity extraction
def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# Telegram message handler
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text
    response = chat_response(user_input)
    entities = extract_entities(user_input)
    
    if entities:
        ent_str = "\n🔍 I noticed: " + ", ".join([f"{e[0]} ({e[1]})" for e in entities])
    else:
        ent_str = "\n(No extra details detected.)"
    
    await update.message.reply_text(f"{response}{ent_str}")

# /start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! I’m your meeting assistant bot. Ask me to book meetings!")

# Run bot
def main():
    app = ApplicationBuilder().token("7786060258:AAGjIrO5TcfjzYGp3mBEFj336V9Kc7fJHHA").build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
