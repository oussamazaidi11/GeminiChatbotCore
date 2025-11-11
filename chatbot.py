import google.generativeai as genai
import os
from dotenv import load_dotenv  # <-- 1. IMPORT THIS

# --- Configuration ---

# Load variables from the .env file into the environment
load_dotenv()

# --- Configuration ---
# Read the API key from the environment variable
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    print("Please set the environment variable before running the script.")
    exit()

# --- Model and Chat Initialization ---
# Set up the model
generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

# Apply safety settings
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# Initialize the model
model = genai.GenerativeModel(
    model_name="gemini-1.0-pro",  # A great, reliable model to start with
    generation_config=generation_config,
    safety_settings=safety_settings,
)

# Start a new chat session. This object will store the history!
chat = model.start_chat(history=[])

# --- The Main Chat Loop ---
print("🤖 Hello! I am your Gemini chatbot. Type 'quit' to exit.")

while True:
    # Get input from the user
    user_input = input("You: ")

    # Check if the user wants to quit
    if user_input.lower() == "quit":
        print("🤖 Goodbye! Have a great day.")
        break

    # Send the user's message to the model and get the response
    try:
        response = chat.send_message(user_input, stream=True)
        
        # Print the response in chunks as it comes in (streaming)
        print("Gemini: ", end="")
        for chunk in response:
            print(chunk.text, end="")
        print()  # Add a newline after the full response

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please try again.")