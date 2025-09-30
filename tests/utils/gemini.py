import os
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai  # 1. Import the library

# --- Robustly find and load the .env file ---
try:
    # Navigate from the current script's directory up to the project root
    script_dir = Path(__file__).resolve().parent
    # Adjust this if your script is nested differently
    project_root = script_dir.parent.parent 
    dotenv_path = project_root / "configs" / ".env"
    
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
        print(f"✅ Environment variables loaded from: {dotenv_path}")
    else:
        print(f"⚠️ Warning: .env file not found at {dotenv_path}.")

except Exception as e:
    print(f"Could not load .env file: {e}")

# --- Get API key and list models using the client ---
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    try:
        # 2. Create a client instance. It automatically uses the API key
        #    from the environment variables.
        client = genai.Client()

        print("\n✅ Available models for 'generateContent':")
        # 3. Use the client object to list the models
        for m in client.models.list():
            if 'generateContent' in m.supported_generation_methods:
                print(f"- {m.name}")
    
    except Exception as e:
        print(f"🔴 An error occurred while listing models: {e}")
        print("💡 Tip: Make sure your API key is correct and has the right permissions.")
else:
    print("🔴 Error: GEMINI_API_KEY not found. Please check your .env file.")