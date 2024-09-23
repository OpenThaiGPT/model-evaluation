from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class OpenThaiGPTHF13B2023:
    def __init__(self, model_name, model_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path or model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path or model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        # Move model to CUDA device
        self.model.to(self.device)

    def inference(self, prompt):
        llama_prompt = f"<s>[INST] <<SYS>>\nYou are a question answering assistant. Answer the question as truthful and helpful as possible คุณคือผู้ช่วยตอบคำถาม จงตอบคำถามอย่างถูกต้องและมีประโยชน์ที่สุด<</SYS>>\n\n{prompt} [/INST]"
        inputs = self.tokenizer.encode(llama_prompt, return_tensors="pt")
        outputs = self.model.generate(inputs, max_length=512, num_return_sequences=1)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    # Example usage:
    model = OpenThaiGPTHF13B2023("openthaigpt/openthaigpt-1.0.0-beta-13b-chat-hf")
    model.move_to_device()
    print(model.inference("What is the meaning of life?"))
