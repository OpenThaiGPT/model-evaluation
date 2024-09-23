from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class OpenThaiGPTHF2024:
    def __init__(self, model_name, model_path=None) -> None:
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

    def inference(self, prompt):  # -> Any:
        llama_prompt = f"<s>[INST] <<SYS>>\nYou are a student sitting in an exam. Answer the question with the correct choice and explain the reasoning คุณคือนักเรียนที่กำลังทำข้อสอบ จงตอบคำถามโดยเลือกช้อยส์ที่ถูกต้องพร้อมทั้งอธิบายเหตุผล<</SYS>>\n\n{prompt} [/INST]"

        inputs = self.tokenizer.encode(llama_prompt, return_tensors="pt")
        inputs = inputs.to(self.device)

        outputs = self.model.generate(inputs, max_length=512, num_return_sequences=1)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    # Example usage:
    model = OpenThaiGPTHF2024("openthaigpt/openthaigpt-1.0.0-7b-chat")
    print(model.inference("Hello!"))
