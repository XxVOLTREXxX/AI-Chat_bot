import nltk
import pickle
import json
import random
import spacy
from nltk.stem import WordNetLemmatizer
nlp = spacy.load("en_core_web_sm")


# Load trained model and data
with open("chat_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("responses.json", "r") as f:
    responses = json.load(f)

lemmatizer = WordNetLemmatizer()
nltk.download('punkt')


def chat_response(user_input):
    # Tokenize and lemmatize the input
    tokens = nltk.word_tokenize(user_input)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens]
    processed_input = " ".join(tokens)

    # Vectorize user input
    input_vector = vectorizer.transform([processed_input])

    # Predict the intent
    predicted_tag = model.predict(input_vector)[0]

    # Select a random response from the predicted intent
    return random.choice(responses[predicted_tag])

print("🤖 Chatbot is ready! Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit", "bye"]:
        print("Bot: Goodbye!")
        break
    response = chat_response(user_input)
    print("Bot:", response)



def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities


while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit", "bye"]:
        print("Bot: Goodbye!")
        break
    
    # Generate response
    response = chat_response(user_input)
    print("Bot:", response)

    # Show extracted entities
    entities = extract_entities(user_input)
    if entities:
        print("🔍 Extracted Entities:", entities)
    else:
        print("🔍 No entities found.")
