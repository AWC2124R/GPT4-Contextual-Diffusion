import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GPT4_APIKEY")

# Calls GPT-4 with the appropriate prompts, combined with the original user prompt and detected classes within the image.
# Returns the subprompts GPT-4 generated for each class.
def call_gpt4(detectedClasses, userPrompt):
    subPrompts = []
    
    return subPrompts