import requests

class TyphoonModel:
    def __init__(self, model_name, api_key):
        self._model_name = model_name
        self._api_key = api_key

    @property
    def model_name(self) -> Any:
        return self._model_name

    @property
    def api_key(self) -> Any:
        return self._api_key

    def inference(self, prompt):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        data = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. You must answer only in Thai."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 500,
            "temperature": 0.7,
            "top_p": 1,
            "top_k": 50,
            "repetition_penalty": 1.15,
            "stream": False
        }

        response = requests.post("https://api.opentyphoon.ai/v1/chat/completions", headers=headers, json=data)
        resp = response.json()
        print(resp)

        return resp['choices'][0]['message']['content']

if __name__ == "__main__":
    model = TyphoonModel("your_model_name", "your_api_key")
    response = model.inference("Your prompt here")
    print(response)