from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class SeaLLM_V1:

    def __init__(self, model_name: str, model_path=None) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.BOS_TOKEN = "<s>"
        self.EOS_TOKEN = "</s>"
        self.B_INST, self.E_INST = "[INST]", "[/INST]"
        self.B_SYS, self.E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        self.SYSTEM_PROMPT = """You are a multilingual, helpful, respectful and honest assistant. \
            Please always answer as helpfully as possible, while being safe. Your \
            answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure \
            that your responses are socially unbiased and positive in nature.
            
            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not \
            correct. If you don't know the answer to a question, please don't share false information.
            
            As a multilingual assistant, you must respond and follow instructions in the native language of the user by default, unless told otherwise. \
            Your response should adapt to the norms and customs of the respective language and culture.
            """

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path or model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path or model_name, trust_remote_code=True, torch_dtype=torch.float16
        )

        # Move model to CUDA device
        self.model.to(self.device)

    def chat_multiturn_seq_format(
        self,
        message: str,
        history: list[tuple[str, str]] = [],
    ) -> str:
        """
        ```
            <bos>[INST] B_SYS SytemPrompt E_SYS Prompt [/INST] Answer <eos>
            <bos>[INST] Prompt [/INST] Answer <eos>
            <bos>[INST] Prompt [/INST]
        ```

        As the format auto-add <bos>, please turn off add_special_tokens with `tokenizer.add_special_tokens = False`
        Inputs:
          message: the current prompt
          history: list of list indicating previous conversation. [[message1, response1], [message2, response2]]
        Outputs:
          full_prompt: the prompt that should go into the chat model

        e.g:
          full_prompt = chat_multiturn_seq_format("Hello world")
          output = model.generate(tokenizer.encode(full_prompt, add_special_tokens=False), ...)
        """
        text = ""
        for i, (prompt, res) in enumerate(history):
            if i == 0:
                text += f"{self.BOS_TOKEN}{self.B_INST} {self.B_SYS} {self.SYSTEM_PROMPT} {self.E_SYS} {prompt} {self.E_INST}"
            else:
                text += f"{self.BOS_TOKEN}{self.B_INST} {prompt}{self.E_INST}"
            if res is not None:
                text += f" {res} {self.EOS_TOKEN} "
        if len(history) == 0 or text.strip() == "":
            text = f"{self.BOS_TOKEN}{self.B_INST} {self.B_SYS} {self.SYSTEM_PROMPT} {self.E_SYS} {message} {self.E_INST}"
        else:
            text += f"{self.BOS_TOKEN}{self.B_INST} {message} {self.E_INST}"
        return text

    def inference(self, prompt: str):
        inputs = self.tokenizer.encode(
            self.chat_multiturn_seq_format(prompt), return_tensors="pt"
        )
        inputs = inputs.to(self.device)
        outputs = self.model.generate(inputs, max_length=512, num_return_sequences=1)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
