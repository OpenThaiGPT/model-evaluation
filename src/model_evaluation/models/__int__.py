from .claude import ClaudeAPI
from .gemini import GeminiAPI
from .openai import OpenAIAPI
from .openthaigpt_hf_7b_2023 import OpenThaiGPTHF7B2023
from .openthaigpt_hf_13b_2023 import OpenThaiGPTHF13B2023
from .openthaigpt_hf_2024 import OpenThaiGPTHF2024
from .sailor import SailorModel
from .sealion import SeaLionModel
from .seallm_v1 import SeaLLM_V1
from .seallm_v2 import SeaLLM_V2
from .typhoongpt import TyphoonModel
from .wangchanglm import WangChangLMModel


support_models = {
    "openthaigpt/openthaigpt-1.0.0-beta-7b-chat-ckpt-hf": OpenThaiGPTHF7B2023,
    "openthaigpt/openthaigpt-1.0.0-beta-13b-chat-hf": OpenThaiGPTHF13B2023,
    "openthaigpt/openthaigpt-1.0.0-7b-chat": OpenThaiGPTHF2024,
    "openthaigpt/openthaigpt-1.0.0-13b-chat": OpenThaiGPTHF2024,
    "openthaigpt/openthaigpt-1.0.0-70b-chat": OpenThaiGPTHF2024,
    "sail/Sailor-7B-Chat": SailorModel,
    "pythainlp/wangchanglm-7.5B-sft-enth": WangChangLMModel,
    "aisingapore/sea-lion-7b-instruct": SeaLionModel,
    "SeaLLMs/SeaLLM-7B-v1": SeaLLM_V1,
    "SeaLLMs/SeaLLM-7B-v2": SeaLLM_V2,
    "claude-3-opus-20240229": lambda model_name, api_key: ClaudeAPI(model_name, api_key),
    "claude-3-sonnet-20240229": lambda model_name, api_key: ClaudeAPI(model_name, api_key),
    "claude-3-haiku-20240307": lambda model_name, api_key: ClaudeAPI(model_name, api_key),
    "typhoon-instruct": lambda model_name, api_key: TyphoonModel(model_name, api_key),
    "gpt-3.5-turbo": lambda model_name, api_key: OpenAIAPI(model_name, api_key),
    "gpt-4": lambda model_name, api_key: OpenAIAPI(model_name, api_key),
    "gemini-pro-1.5": lambda model_name, api_key: GeminiAPI(model_name, api_key),
}

def get_model(model_name, api_key=None):
    if model_name in support_models:
        return support_models[model_name](model_name, api_key) 
