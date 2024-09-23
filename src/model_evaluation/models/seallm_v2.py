from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class SeaLLM_V2:
    def __init__(self, model_name, model_path=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_path or model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path or model_name, trust_remote_code=True, torch_dtype=torch.float16)
        
        # Move model to CUDA device
        self.model.to(self.device)

    def inference(self, prompt):
        messages = [
            {"role": "system", "content": 'You are a helpful assistant'},
            {"role": "user", "content": prompt}
        ]
        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
        
        model_inputs = encodeds.to(self.device)
        
        generated_ids = self.model.generate(model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)
        decoded = self.tokenizer.batch_decode(generated_ids)
        return decoded[0].strip()

if __name__ == "__main__":
    sea_llm = SeaLLM_V2("SeaLLMs/SeaLLM-7B-v2")
    response = sea_llm.inference("Hello, how are you?")
    print(response)