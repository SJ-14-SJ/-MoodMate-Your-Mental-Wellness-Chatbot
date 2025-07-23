🌿 MoodMate – Your Mental Wellness Chatbot
MoodMate is an AI-powered mental wellness companion designed to provide emotional support through real-time chat. Built using PyTorch, Flask, and Natural Language Processing, it helps users express their feelings, get supportive responses, and feel heard — anytime, anywhere.

🧠 Features
🗣️ Conversational chatbot trained on custom intents

💬 Real-time web interface with Flask + JavaScript

🤖 NLP-powered text understanding (tokenization, bag-of-words)

📁 Easily customizable with your own intents

💡 Future-ready for sentiment analysis, mood tracking, and MongoDB integration

📁 Project Structure
moodmate/
├── static/
│   └── style.css               # Styling for the frontend (optional)
├── templates/
│   └── index.html              # Web UI for chatbot
├── app.py                      # Flask app to serve the chatbot
├── chat.py                     # Chat logic (used by both CLI and web)
├── train.py                    # Model training script
├── model.py                    # PyTorch NeuralNet definition
├── nltk_utils.py               # Tokenizer and bag-of-words logic
├── intents.json                # Training data with intents and responses
├── data.pth                    # Trained model (saved state)
├── README.md                   # Project description
└── requirements.txt            # Required Python packages

📈 Future Enhancements
✅ Sentiment analysis using transformers

✅ Mood logging over time

✅ MongoDB integration for persistent chat history

✅ Dark/light UI theme switcher

✅ Mobile-friendly responsive design
