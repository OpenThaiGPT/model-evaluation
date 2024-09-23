import requests


class GeminiAPI:

    def __init__(self, model_name: str, api_key: str):
        self._model_name = model_name
        self._api_key = api_key

    def inference(self, prompt: str) -> str:
        headers = {
            "Content-Type": "application/json",
        }

        data = {"contents": [{"parts": [{"text": prompt}]}]}

        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self._model_name}:generateContent?key={self._api_key}",
            headers=headers,
            json=data,
        )
        resp = response.json()

        return resp["candidates"][0]["content"]["parts"][0]["text"]
