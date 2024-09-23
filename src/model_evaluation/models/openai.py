import requests


class OpenAIAPI:
    def __init__(self, model_name, api_key):
        """
        Initialize the OpenAI API client.

        Args:
            model_name (str): The name of the OpenAI model to use.
            api_key (str): The OpenAI API key to authenticate with.
        """
        self._model_name = model_name
        self._api_key = api_key

    def inference(self, prompt):
        """
        Make an inference request to the OpenAI API.

        Args:
            prompt (str): The input text to pass to the OpenAI model.

        Returns:
            str: The generated response from the OpenAI model.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        data = {
            "model": self._model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=data
        )
        resp = response.json()

        return resp["choices"][0]["message"]["content"]


if __name__ == "__main__":
    # Example usage
    open_ai = OpenAIAPI("text-davinci-003", "YOUR_API_KEY")
    print(open_ai.inference("Hello, how are you?"))
