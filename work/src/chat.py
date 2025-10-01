# chat类函数，用于调用openai的chat接口

import json
from openai import OpenAI
import requests
from openai import AzureOpenAI

endpoint = "https://sencemaking.openai.azure.com/"
model_name = "gpt-4o-mini"
deployment = "gpt-4o-mini"

subscription_key = "REMOVED_CREDENTIAL"
api_version = "2024-12-01-preview"

class Chat:
    def __init__(self, begin_messages=None, max_tokens=4096, temperature=1.0, top_p=1.0):
        self.client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=subscription_key,
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

    
    def chat_context(self, prompt):
        messages = self.messages + [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            model=self.model
        )
        messages.append({"role": "assistant", "content": response.choices[0].message.content})
        return response.choices[0].message.content
    
    def temp_chat(self, prompt):
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            model=self.model
        )
        return response.choices[0].message.content

    def get_messages(self):
        return self.messages


