# Import required libraries
from flask import Flask, request, jsonify  # For API creation and response handling
from flask_cors import CORS  # To handle cross-origin requests (from Flutter frontend)
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification  # For Hugging Face model
import openai  # For OpenAI GPT-3.5 integration (v1.x+)
import os
from dotenv import load_dotenv  # To load environment variables from a .env file

# Load variables from .env file (like OPENAI_API_KEY)
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Define model path relative to the current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "finalmodel", "distilbert_sentiment_best")

# Load sentiment analysis model and tokenizer from local directory
try:
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True
    )
    # Create sentiment analysis pipeline
    sentiment_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)
except Exception as e:
    print(f"Error loading Hugging Face model: {e}")
    sentiment_pipeline = None  # Fallback in case model fails to load

# Get OpenAI API key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("❌ OPENAI_API_KEY not found. Make sure it's in your .env file.")

# Configure OpenAI SDK
openai.api_key = openai_api_key

# ------------------ Define Routes ------------------ #

@app.route('/chat', methods=['POST'])
def chat():
    """
    POST endpoint to process user input:
    1. Analyze sentiment using Hugging Face
    2. Generate an empathetic response using OpenAI's GPT
    """
    data = request.get_json()
    user_input = data.get("text", "")

    if not user_input:
        return jsonify({"error": "No input text provided."}), 400

    # Step 1: Sentiment Analysis
    try:
        sentiment_result = sentiment_pipeline(user_input)
        raw_label = sentiment_result[0]["label"]
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return jsonify({"error": "Sentiment analysis failed."}), 500

    # Convert model label to readable sentiment
    sentiment_label = {
        "LABEL_0": "positive",
        "LABEL_1": "neutral",
        "LABEL_2": "negative"
    }.get(raw_label, "neutral")

    # Step 2: Build prompt for chatbot
    prompt = f"""
You are SereniBot, a compassionate mental health chatbot.
The user seems to be feeling {sentiment_label}.
Respond kindly and empathetically.

User: "{user_input}"
SereniBot:"""

    # Step 3: Generate response from GPT
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a supportive mental health chatbot."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=200
        )
        reply = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in OpenAI API call: {e}")
        reply = "Sorry, I'm having trouble responding right now."

    # Return sentiment + AI-generated reply
    return jsonify({
        "sentiment": sentiment_label,
        "response": reply
    })

# Run the Flask app on port 8000
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
