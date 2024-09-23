from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class WangChangLMModel:
    def __init__(self, model_name, model_path=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Ensure CUDA is available
        self.model_name = model_name
        self.model_path = model_path or model_name
        
        # Ref: https://huggingface.co/pythainlp/wangchanglm-7.5B-sft-enth
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            return_dict=True, 
            torch_dtype=torch.float16, 
            offload_folder="./", 
            low_cpu_mem_usage=True,
        )
        
        # Move model to CUDA device
        self.model.to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def inference(self, prompt):
        batch = self.tokenizer(prompt + "\nตอบว่า:", return_tensors="pt")
        batch.to(self.device)
        
        with torch.cuda.amp.autocast(): 
            output_tokens = self.model.generate(
                input_ids=batch["input_ids"],
                max_new_tokens=512,
                no_repeat_ngram_size=2,
                
                #oasst k50
                top_k=50,
                top_p=0.95, 
                typical_p=1.,
                temperature=0.9, 
                
            )
        
        response = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        return response.strip()