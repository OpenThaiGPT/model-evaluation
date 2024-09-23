import requests


class ClaudeAPI:
    def __init__(self, model_name, api_key):
        """
        Initializes the Claude API client.

        Args:
            model_name (str): The name of the Claude model to use.
            api_key (str): The API key for the Anthropic API.
        """
        self.model_name = model_name
        self.api_key = api_key

    def inference(self, prompt):
        """
        Makes an inference request to the Claude API.

        Args:
            prompt (str): The text prompt to pass to the API.

        Returns:
            str: The generated response from the API.
        """
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        data = {
            "model": self.model_name,
            "max_tokens": 512,
            "messages": [{"role": "user", "content": prompt}],
        }

        response = requests.post(
            "https://api.anthropic.com/v1/messages", headers=headers, json=data
        )
        resp = response.json()

        return resp["content"][0]["text"]


if __name__ == "__main__":
    model_name = ""
    api_key = ""
    # Example usage:
    claude_api = ClaudeAPI(model_name, api_key)
    response = claude_api.inference(prompt="Hello, how are you?")
    print(response)
