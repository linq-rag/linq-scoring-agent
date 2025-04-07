from openai.types.chat import (
    ParsedChatCompletion,
    ParsedChatCompletionMessage,
    ParsedChoice,
)

DEFAULT_OPENAI_KWARGS = {
    "model": "gpt-4o-mini-2024-07-18",
    "temperature": 0.0,
    "timeout": 20.0, 
}

DEFAULT_GROQ_KWARGS = {
    "model": "llama-3.1-8b-instant",
    "temperature": 0.0,
    "timeout": 15.0,
    # "response_format": {"type": "json_object"}
}

DEFAULT_FURIOSA_KWARGS = {
    "model": "EMPTY",
}

DEFAULT_EMPTY_PARSED_COMPLETION = ParsedChatCompletion(
    id="",
    choices=[ParsedChoice(finish_reason="stop", index=0, message=ParsedChatCompletionMessage(role="assistant"))], 
    created=0,
    model="", 
    object="chat.completion"
)