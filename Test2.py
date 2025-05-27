import json
import random
import nltk
import pickle
import spacy
from nltk.stem import WordNetLemmatizer

# Load models
nlp = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()
nltk.download('punkt')

with open("chat_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("responses.json", "r") as f:
    responses = json.load(f)

# Chatbot response
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

# Chat loop
print("🤖 Chatbot ready! Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        print("Bot: Goodbye!")
        break

    response = chat_response(user_input)
    print("Bot:", response)

    entities = extract_entities(user_input)
    if entities:
        print("🔍 Extracted Entities:", entities)
    else:
        print("🔍 No entities found.")
