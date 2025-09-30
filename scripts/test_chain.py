"""
Quick end-to-end smoke test for the provider cascade.
Run after setting configs/.env with your keys.
"""
from app.core.inference import chat

msgs = [
    {"role": "system", "content": "You are concise."},
    {"role": "user", "content": "Say hello in one sentence and mention which provider you are (if you can)."},
]

print(chat(msgs, stream=False))
