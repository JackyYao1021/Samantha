# chat module, handles chat interactions with OpenAI and Qwen models

import os
from types import SimpleNamespace
import httpx
from openai import AzureOpenAI, OpenAI
from model_output import parse_json_object


endpoint = "https://sencemaking.openai.azure.com/"
model_name = "gpt-4o-mini"
deployment = "gpt-4o-mini"

api_version = "2024-12-01-preview"


openai_api_key_qwen = "ollama"
openai_api_base_qwen = "http://127.0.0.1:11434/v1"
model_qwen = "qwen3:4b"


def response_text(response):
    """Read the final answer, including Qwen templates that leak thinking text."""
    choice = response.choices[0]
    if choice.finish_reason == "length":
        raise ValueError("The model response reached its token limit before completion.")
    text = choice.message.content
    if not isinstance(text, str) or not text.strip():
        raise ValueError("The model returned no final answer.")
    text = text.strip()
    if text.startswith("<think>") or (
        "</think>" in text and not text.startswith(("{", "[", "```"))
    ):
        _, separator, text = text.partition("</think>")
        if not separator or not text.strip():
            raise ValueError("The model returned thinking without a final answer.")
    return text.strip()


class Chat:
    def __init__(self, begin_messages=None, max_tokens=4096, temperature=0.2, top_p=1.0,
                 response_format=None):
        subscription_key = os.environ.get("AZURE_OPENAI_API_KEY")
        azure_fallback = os.environ.get("SAMANTHA_AZURE_FALLBACK", "false").lower() in {"1", "true", "yes"}
        self.client = AzureOpenAI(
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", api_version),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", endpoint),
            api_key=subscription_key,
        ) if azure_fallback and subscription_key else None
        self.client_qwen = OpenAI(
            api_key=os.environ.get("QWEN_API_KEY", openai_api_key_qwen),
            base_url=os.environ.get("QWEN_BASE_URL", openai_api_base_qwen),
            timeout=float(os.environ.get("QWEN_TIMEOUT", "120")),
            max_retries=0,
        )
        self.api_mode = os.environ.get("QWEN_API_MODE", "openai").lower()
        if self.api_mode not in {"openai", "ollama"}:
            raise ValueError("QWEN_API_MODE must be openai or ollama.")
        self.messages = [
            {
                "role": "system",
                "content": begin_messages if begin_messages else "You are a helpful assistant.",
            }
        ]
        self.model = os.environ.get("AZURE_OPENAI_DEPLOYMENT", deployment)
        self.model_qwen = os.environ.get("QWEN_MODEL", model_qwen)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.response_format = response_format
        self.reasoning_effort = os.environ.get("QWEN_REASONING_EFFORT", "none").strip()

    """
    Chat with context
    """
    def chat_context(self, prompt):
        self.messages = self.messages + [{"role": "user", "content": prompt}]
        response = self.send_message(self.messages)
        answer = response_text(response)
        self.messages.append({"role": "assistant", "content": answer})
        return answer
    
    """
    Temporary chat without context
    Not recommended for use, as it does not retain conversation history
    """
    def temp_chat(self, prompt):
        messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        response = self.send_message(messages)
        return response_text(response)
    
    def send_message(self, message):
        options = {"response_format": self.response_format} if self.response_format else {}
        qwen_options = dict(options)
        qwen_message = message
        if self.reasoning_effort:
            qwen_options["extra_body"] = {"reasoning_effort": self.reasoning_effort}
        # Some Ollama Qwen3 templates still open <think> when effort is none.
        # Use Qwen3's soft switch too, without changing history or Azure input.
        if self.reasoning_effort.lower() == "none" and self.model_qwen.lower().split(":")[0] == "qwen3":
            for index in range(len(message) - 1, -1, -1):
                item = message[index]
                if item.get("role") == "user" and isinstance(item.get("content"), str):
                    if not item["content"].rstrip().endswith("/no_think"):
                        qwen_message = list(message)
                        qwen_message[index] = {**item, "content": item["content"] + "\n/no_think"}
                    break
        try:
            if self.api_mode == "ollama":
                response = self._send_ollama(message)
            else:
                response = self.client_qwen.chat.completions.create(
                    messages=qwen_message,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    model=self.model_qwen,
                    **qwen_options,
                )
        except Exception as e:
            if self.client is None:
                raise RuntimeError(
                    f"Qwen request failed at {self.client_qwen.base_url} "
                    f"for model {self.model_qwen}: {e}. "
                    "Azure fallback is not configured or enabled. "
                    "Check Ollama, QWEN_BASE_URL and QWEN_MODEL."
                ) from e
            response = self.client.chat.completions.create(
                messages=message,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                model=self.model,
                **options,
            )
        return response

    def _send_ollama(self, messages):
        root = str(self.client_qwen.base_url).rstrip("/").removesuffix("/v1")
        payload = {
            "model": self.model_qwen, "messages": messages, "stream": False,
            "options": {"num_predict": self.max_tokens,
                        "temperature": self.temperature, "top_p": self.top_p},
        }
        if self.reasoning_effort:
            payload["think"] = (False if self.reasoning_effort.lower() == "none" else
                                True if self.model_qwen.lower().split(":")[0] == "qwen3" else
                                self.reasoning_effort)
        if self.response_format:
            if self.response_format["type"] == "json_schema":
                payload["format"] = self.response_format["json_schema"]["schema"]
            elif self.response_format["type"] == "json_object":
                payload["format"] = "json"
            else:
                raise ValueError("Ollama requires json_object or json_schema output.")
        with httpx.Client(timeout=float(os.environ.get("QWEN_TIMEOUT", "120")),
                          trust_env=False) as client:
            response = client.post(root + "/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
        if data.get("done") is not True:
            raise ValueError("Ollama returned an incomplete response.")
        # Keep the existing interface; protocol reasoning is excluded from history.
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="length" if data.get("done_reason") == "length" else "stop",
            message=SimpleNamespace(content=data.get("message", {}).get("content")),
        )])


    def get_messages(self):
        return self.messages


def request_text(begin_messages, prompt, *, max_tokens=4096):
    """Use a JSON envelope for text agents, then preserve their text interface."""
    instructions = (
        '\nReturn one JSON object with exactly one field "text". '
        'Put the requested final response, including its Markdown formatting, '
        'in the "text" string. Do not include analysis or thinking.'
    )
    chat = Chat(begin_messages=begin_messages + instructions, max_tokens=max_tokens,
                response_format={"type": "json_object"})
    data = parse_json_object(chat.chat_context(prompt))
    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("The model must return a non-empty text field.")
    return text


