from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class SeaLionModel:
    def __init__(self, model_name, model_path=None):
        """
        Initialize the Sea Lion Model.

        Args:
            model_name (str): Name of the pre-trained model.
            model_path (str, optional): Path to a custom model. Defaults to None.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path or model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path or model_name, trust_remote_code=True, torch_dtype=torch.float16
        )

        # Move model to CUDA device
        self.model.to(self.device)

    def inference(self, prompt):
        """
        Perform inference on the input prompt.

        Args:
            prompt (str): The input prompt for which a response is generated.

        Returns:
            str: The generated response.
        """
        tokens = self.tokenizer(
            f"### USER:\n{prompt}\n\n### RESPONSE:\n", return_tensors="pt"
        )

        # Move tokens to CUDA device
        tokens = tokens.to(self.device)
        output = self.model.generate(
            tokens["input_ids"],
            max_new_tokens=20,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response.split("### RESPONSE:")[-1].strip()


if __name__ == "__main__":
    model = SeaLionModel("aisingapore/sea-lion-7b-instruct")
    response = model.inference("Your prompt here")
    print(response)
