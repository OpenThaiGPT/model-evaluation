from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class SailorModel:
    def __init__(self, model_name, model_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path or model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path or model_name, trust_remote_code=True, torch_dtype=torch.float16
        )
        self.model.to(self.device)

    def inference(self, prompt):
        # Ref: https://huggingface.co/sail/Sailor-7B-Chat
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "question", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        input_ids = model_inputs.input_ids.to(self.device)

        generated_ids = self.model.generate(
            input_ids,
            max_new_tokens=512,
        )

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]
        return response.strip()
