# Telegram Meeting Assistant Bot

A smart Telegram bot that helps users schedule and manage meetings using Natural Language Processing (NLP) capabilities. The bot can understand natural language inputs, extract relevant entities, and provide contextual responses.

## Features

- 🤖 Natural Language Understanding
- 🔍 Entity Recognition and Extraction
- 💬 Context-Aware Responses
- 📅 Meeting Management Assistance
- 🔄 Intelligent Response Generation

## Technologies Used

- Python 3.x
- Telegram Bot API
- NLTK (Natural Language Toolkit)
- spaCy for Named Entity Recognition
- Scikit-learn (Machine Learning Model)
- pickle (Model Serialization)

## Prerequisites

Before running the bot, make sure you have Python installed and the following dependencies:

```bash
python-telegram-bot
nltk
spacy
scikit-learn
```

You'll also need to download the spaCy English language model:

```bash
python -m spacy download en_core_web_sm
```

## Setup

1. Clone this repository:
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Set up your Telegram Bot:
   - Create a new bot with [@BotFather](https://t.me/botfather) on Telegram
   - Get your bot token
   - Replace the token in `Main.py` with your bot token

4. Ensure all model files are present:
   - `chat_model.pkl`
   - `vectorizer.pkl`
   - `responses.json`

## Running the Bot

To start the bot, simply run:

```bash
python Main.py
```

## Usage

1. Start a chat with your bot on Telegram
2. Send `/start` to begin the conversation
3. Ask about scheduling meetings or any related queries
4. The bot will respond with appropriate information and extract relevant entities from your messages

## Features in Detail

### Natural Language Processing
- Uses NLTK for tokenization and lemmatization
- Implements spaCy for named entity recognition
- Extracts relevant information from user messages

### Response Generation
- Uses a pre-trained machine learning model
- Generates context-aware responses
- Identifies entities in user messages

## Contributing

Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.
