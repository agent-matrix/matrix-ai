import os
from pathlib import Path
from groq import Groq
from google import genai # Using the specific import you requested
from dotenv import load_dotenv

def test_groq_connection():
    """
    Loads the Groq API key from a .env file and tests the endpoint
    with a simple streaming query.
    """
    # 1. Build a reliable path to the .env file
    # This finds the script's directory, goes up to the project root,
    # and then into the 'configs' folder.
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    dotenv_path = project_root / "configs" / ".env"
    load_dotenv(dotenv_path=dotenv_path)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print(f"🔴 Error: GROQ_API_KEY not found.")
        print(f"Please ensure it is set in your {dotenv_path} file.")
        return

    print("✅ Groq API key loaded successfully.")

    try:
        # 2. Initialize the Groq client
        client = Groq()
        print("🤖 Initialized Groq client. Sending a test query...")

        # 3. Create a test chat completion request
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": "Explain why low-latency is important for LLMs in one short sentence."
                }
            ],
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )

        # 4. Print the streamed response from the model
        print("\n📝 Groq API Response:")
        print("-" * 20)
        for chunk in completion:
            print(chunk.choices[0].delta.content or "", end="")
        print("\n" + "-" * 20)
        print("\n✅ Test successful! The Groq endpoint is working.")

    except Exception as e:
        print(f"🔴 An error occurred during the Groq API call: {e}")

def test_gemini_connection():
    """
    Loads the Google Gemini API key from a .env file and tests the endpoint
    using the genai.Client pattern.
    """
    # 1. Build a reliable path to the .env file (assuming same location)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    dotenv_path = project_root / "configs" / ".env"
    load_dotenv(dotenv_path=dotenv_path)

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print(f"🔴 Error: GOOGLE_API_KEY not found.")
        print(f"Please ensure it is set in your {dotenv_path} file.")
        return

    print("✅ Google API key loaded successfully.")

    try:
        # 2. Initialize the Gemini client using the specified pattern
        client = genai.Client(api_key=api_key)
        print("🤖 Initialized Gemini client. Sending a test query...")

        # 3. Send a test prompt using the client.models.generate_content method
        response = client.models.generate_content(
            model="gemini-2.5-flash", # Using the qualified model name
            contents="Explain the importance of APIs in one short sentence."
        )

        # 4. Print the response
        print("\n📝 Gemini API Response:")
        print("-" * 20)
        print(response.text)
        print("-" * 20)
        print("\n✅ Test successful! The Gemini endpoint is working.")

    except Exception as e:
        print(f"🔴 An error occurred during the Gemini API call: {e}")


# Run the test functions when the script is executed
if __name__ == "__main__":
    print("--- Running Groq API Connection Test ---")
    test_groq_connection()
    print("\n" + "="*40 + "\n")
    print("--- Running Gemini API Connection Test ---")
    test_gemini_connection()