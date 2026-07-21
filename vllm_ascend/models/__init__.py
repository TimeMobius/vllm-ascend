from transformers import AutoConfig
from vllm import ModelRegistry
from vllm.model_executor.models.config import MODELS_CONFIG_MAP, MambaModelConfig

from vllm_ascend.models.rwkv7_config import RWKV7Config


def register_model():
    AutoConfig.register("rwkv7", RWKV7Config, exist_ok=True)
    MODELS_CONFIG_MAP["RWKV7ForCausalLM"] = MambaModelConfig
    ModelRegistry.register_model(
        "RWKV7ForCausalLM", "vllm_ascend.models.rwkv7:RWKV7ForCausalLM"
    )
    ModelRegistry.register_model("DeepseekV4ForCausalLM", "vllm_ascend.models.deepseek_v4:AscendDeepseekV4ForCausalLM")

    ModelRegistry.register_model("DeepSeekV4MTPModel", "vllm_ascend.models.deepseek_v4_mtp:DeepSeekV4MTP")
    ModelRegistry.register_model(
        "DSparkDraftModel",
        "vllm_ascend.models.deepseek_v4_dspark:DSparkDeepseekV4ForCausalLM",
    )
    ModelRegistry.register_model(
        "LlamaForCausalLMVwnEagle3", "vllm_ascend.models.llama_eagle3_vwn:Eagle3VwnLlamaForCausalLM"
    )
