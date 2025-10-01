# chat类函数，用于调用openai的chat接口

import json
from openai import OpenAI
import requests
from openai import AzureOpenAI, OpenAI


endpoint = "https://sencemaking.openai.azure.com/"
model_name = "gpt-4o-mini"
deployment = "gpt-4o-mini"

subscription_key = "REMOVED_CREDENTIAL"
api_version = "2024-12-01-preview"


openai_api_key_qwen = "EMPTY"
openai_api_base_qwen = "http://host.docker.internal:8000/v1"
model_qwen = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"

class Chat:
    def __init__(self, begin_messages=None, max_tokens=4096, temperature=1.0, top_p=1.0):
        self.client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=subscription_key,
        )
        self.client_qwen = OpenAI(
            api_key=openai_api_key_qwen,
            base_url=openai_api_base_qwen,
        )
        self.messages = [
            {
                "role": "system",
                "content": begin_messages if begin_messages else "You are a helpful assistant.",
            }
        ]
        self.model = deployment
        self.max_tokens=4096
        self.temperature=1.0
        self.top_p=1.0

    """
    Chat with context
    """
    def chat_context(self, prompt):
        self.messages = self.messages + [{"role": "user", "content": prompt}]
        response = self.send_message(self.messages)
        self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
        return response.choices[0].message.content
    
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
        return response.choices[0].message.content
    
    def send_message(self, message):
        try:
            response = self.client_qwen.chat.completions.create(
                messages=message,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                model=model_qwen
            )
        except Exception as e:
            # print("Error with Qwen API, switching to Azure OpenAI. Error:", e)
            response = self.client.chat.completions.create(
                messages=message,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                model=self.model
            )
        return response


    def get_messages(self):
        return self.messages
    
    def get_messages(self):
        return self.messages


