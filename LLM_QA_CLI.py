import os
import re
import nltk
from nltk.corpus import stopwords
from google import genai

# Download necessary NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# --- Preprocessing Function ---
def preprocess_question(question: str) -> str:
    """Applies basic NLP preprocessing to the question."""
    # 1. Lowercasing
    question = question.lower()

    # 2. Tokenization
    tokens = nltk.word_tokenize(question)

    # 3. Punctuation Removal (and basic stopword removal - optional but good practice)
    # Using a list comprehension to keep only alphanumeric tokens
    processed_tokens = [
        word for word in tokens
        if word.isalnum()
    ]

    # Reconstruct the processed question string
    processed_question = ' '.join(processed_tokens)
    return processed_question

# --- LLM Interaction Function ---
def get_llm_answer(processed_question: str) -> str:
    """Sends the processed question to the Gemini API and returns the answer."""
    try:
        # Initialize the client (it automatically looks for the GEMINI_API_KEY environment variable)
        client = genai.Client()

        # Construct the final prompt (optional system/user roles for better instruction)
        prompt = (
            "You are a helpful and concise Question-and-Answer system. "
            "Please provide a direct answer to the following question:\n\n"
            f"Question: {processed_question}"
        )

        # Send the request to the model
        response = client.models.generate_content(
            model='gemini-2.5-flash',  # A fast and capable model
            contents=prompt
        )

        return response.text.strip()

    except Exception as e:
        return f"An error occurred while calling the LLM API: {e}"

# --- Main CLI Loop ---
def main_cli():
    """Main function for the CLI application."""
    print("--- LLM Question-and-Answering CLI System ---")
    print("Type 'quit' or 'exit' to close the application.")
    
    # Check for API Key
    if not os.getenv("GEMINI_API_KEY"):
        print("\n⚠️ WARNING: GEMINI_API_KEY environment variable is not set.")
        print("Please set the key to run the application.")
        return

    while True:
        try:
            user_input = input("\nEnter your question: ")

            if user_input.lower() in ['quit', 'exit']:
                print("Exiting application. Goodbye!")
                break

            if not user_input.strip():
                continue

            # 1. Preprocessing
            processed_q = preprocess_question(user_input)
            print(f"\n[Processed Question]: {processed_q}")

            # 2. Get LLM Answer
            print("\n[Querying LLM... please wait]")
            final_answer = get_llm_answer(processed_q)

            # 3. Display Final Answer
            print("\n--- LLM Final Answer ---")
            print(final_answer)
            print("-------------------------")

        except KeyboardInterrupt:
            print("\nExiting application. Goodbye!")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main_cli()
    