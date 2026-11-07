"""Read the OpenAI API key from the OPENAI_API_KEY environment variable."""
import os


def get_key():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return key
