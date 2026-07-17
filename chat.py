import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the client
client = genai.Client()

#This list is the memory. We append evry turn here
messages = []

print("Chat Session started. Type 'exit' to quit.\n" + "="*60 + "\n")

while True:
    user_text = input("You: ")
    if user_text.lower() in ["exit", "quit", "q", "bye"]:
        print("\nChat ended. Goodbye!")
        break
    
    # Append the user message to the list
    messages.append(
        types.Content(role="user", parts=[types.Part(text=user_text)])
    )

    # Send the ENTIRE history to the model
    response = client.models.generate_content(
        model='gemini-3.5-flash',
        contents=messages
    )

    print(f"Gemini: {response.text}\n")

    # Append the model's response to the history so it remembers this next turn
    messages.append(
        types.Content(role="model", parts=[types.Part.from_text(text=response.text)])
    )

