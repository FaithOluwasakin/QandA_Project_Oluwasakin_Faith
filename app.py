import os
import nltk
from flask import Flask, render_template, request
from google import genai

# NOTE: Download NLTK data directly.
# If 'punkt' is already downloaded, nltk.download() will skip it,
# making this method safe and idempotent (runnable multiple times safely).
nltk.download('punkt')

app = Flask(__name__)

# --- Reusable Preprocessing Function (from CLI) ---
def preprocess_question(question: str) -> str:
    """Applies basic NLP preprocessing to the question."""
    question = question.lower()
    tokens = nltk.word_tokenize(question)
    processed_tokens = [
        word for word in tokens
        if word.isalnum()
    ]
    return ' '.join(processed_tokens)

# --- Reusable LLM Interaction Function (from CLI) ---
def get_llm_answer(processed_question: str) -> str:
    """Sends the processed question to the Gemini API and returns the answer."""
    # Ensure the API Key is set
    if not os.getenv("GEMINI_API_KEY"):
        return "ERROR: LLM API Key is not configured on the server."
        
    try:
        client = genai.Client()
        
        prompt = (
            "You are a helpful and concise Question-and-Answer system. "
            "Please provide a direct answer to the following question:\n\n"
            f"Question: {processed_question}"
        )
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        
        return response.text.strip()

    except Exception as e:
        # Log the error for debugging
        print(f"LLM API Error: {e}")
        return "An error occurred while communicating with the LLM API."

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    user_question = ""
    processed_q = ""
    llm_response = ""

    if request.method == 'POST':
        user_question = request.form.get('question')
        
        if user_question:
            # 1. Process Question
            processed_q = preprocess_question(user_question)
            
            # 2. Get LLM Answer
            llm_response = get_llm_answer(processed_q)

    # Render the template with the current state of variables
    return render_template(
        'index.html',
        user_question=user_question,
        processed_q=processed_q,
        llm_response=llm_response
    )

if __name__ == '__main__':
    # When running locally, set API Key (e.g., using python-dotenv or export command)
    # The deployment service (like Render) will use its configuration for the key.
    # IMPORTANT: Never commit your API key directly to your code or GitHub!
    # Example: os.environ["GEMINI_API_KEY"] = "YOUR_API_KEY_HERE"
    
    app.run(debug=True)