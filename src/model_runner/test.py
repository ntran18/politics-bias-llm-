import os
from dotenv import load_dotenv
from google import genai

# This line is the missing piece!
load_dotenv()

# Initialize the client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("Available Gemini Models:")
print("-" * 30)

# List all models
for model in client.models.list():
    # You can filter for models that support text generation
    if "generateContent" in model.supported_actions:
        print(f"Model Name: {model.name}")
        print(f"Display Name: {model.display_name}")
        print(f"Description: {model.description}")
        print("-" * 30)