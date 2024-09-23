from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class OpenThaiGPTHF7B2023:
    def __init__(self, model_name, model_path=None):
        """
        Initializes the model and tokenizer.

        Args:
            model_name (str): Name of the model to use. Defaults to "openthaigpt/openthaigpt-1.0.0-7b-chat".
            model_path (str, optional): Path to a local copy of the model. Defaults to None.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path or model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path or model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        # Move model to CUDA device
        self.model.to(self.device)

    def inference(self, prompt):
        """
        Performs inference on the given prompt.

        Args:
            prompt (str): The input prompt.

        Returns:
            str: The generated response.
        """
        llama_prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:\n"
        inputs = self.tokenizer.encode(llama_prompt, return_tensors="pt")
        inputs = inputs.to(self.device)
        outputs = self.model.generate(
            inputs, num_return_sequences=1, max_new_tokens=2048
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    # Example usage:
    model = OpenThaiGPTHF7B2023(model_name="openthaigpt/openthaigpt-1.0.0-7b-chat")
    print(model.inference("Hello, how are you?"))
