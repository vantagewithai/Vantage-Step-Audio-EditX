import math
import os
import logging
from typing import Optional, Dict, Any, Union, List, Tuple

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoConfig,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from einops import rearrange

# ---- global config directory (root/configs) ----
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(ROOT_DIR, "configs")
# ------------------------------------------------

# =============================================================================
# Step1Config
# =============================================================================

class Step1Config(PretrainedConfig):
    model_type = "step1"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        hidden_size: int = 5120,
        intermediate_size: int = 13312,
        num_attention_heads: int = 40,
        num_attention_groups: int = 8,
        num_hidden_layers: int = 48,
        max_seq_len: int = 4096,
        vocab_size: int = 65536,
        rms_norm_eps: float = 1e-5,
        initializer_range: float = 0.02,
        bos_token_id: int = 1,
        eos_token_id: int = 3,
        pad_token_id: int = 0,
        **kwargs,
    ) -> None:
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_attention_groups = num_attention_groups
        self.num_hidden_layers = num_hidden_layers
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range

        super().__init__(
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )


def build_alibi_cache(block_size, n_heads, dtype, device):
    # get slopes
    n = 2 ** math.floor(math.log2(n_heads))  # nearest 2**n to n_heads
    m0 = 2.0 ** (-8.0 / n)
    # 2^(-8/n), 2^(-8*2/n), 2^(-8*3/n), ...
    slopes = torch.pow(m0, torch.arange(1, n + 1))
    if n < n_heads:
        m1 = 2.0 ** (-4.0 / n)
        # 2^(-8/(2n)), 2^(-8*3/(2n)), 2^(-8*5/(2n)), ...
        mm = torch.pow(m1, torch.arange(1, 1 + 2 * (n_heads - n), 2))
        slopes = torch.cat([slopes, mm])
    slopes = slopes.to(device)

    tril = torch.tril(torch.ones(1, 1, block_size, block_size, device=device))

    bias_rows = torch.arange(block_size, device=device).view(1, -1)
    bias_cols = torch.arange(block_size, device=device).view(-1, 1)
    bias = -torch.sqrt(bias_cols - bias_rows)
    bias = bias.view(1, block_size, block_size) * slopes.view(-1, 1, 1)
    bias = bias.masked_fill(tril == 0, float("-inf"))

    return bias.type(dtype)


class StepRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        var = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps).to(x.dtype)
        x = x * self.weight
        return x


class StepAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_heads, num_groups, layer_idx: int):
        super().__init__()

        self.num_heads = num_heads
        self.num_groups = num_groups
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        self.q_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = torch.nn.Linear(
            hidden_size, num_groups * self.head_dim, bias=False
        )
        self.v_proj = torch.nn.Linear(
            hidden_size, num_groups * self.head_dim, bias=False
        )
        self.o_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)

        self.layer_idx = layer_idx

    def flash_attn_func(
        self,
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=None,
        causal=True,
        return_attn_probs=False,
        tp_group_rank=0,
        tp_group_size=1,
    ):
        softmax_scale = q.size(-1) ** (-0.5) if softmax_scale is None else softmax_scale
        return torch.ops.Optimus.fwd(
            q,
            k,
            v,
            None,
            dropout_p,
            softmax_scale,
            causal,
            return_attn_probs,
            None,
            tp_group_rank,
            tp_group_size,
        )[0]

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        q: torch.Tensor = self.q_proj(x)
        k: torch.Tensor = self.k_proj(x)
        v: torch.Tensor = self.v_proj(x)
        if past_key_value is not None:
            cache_kwargs = {"cache_position": cache_position}
            k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)

        q = rearrange(q, "b s (h d) -> b s h d", h=self.num_heads)
        k = rearrange(k, "b s (g d) -> b s g d", g=self.num_groups)
        v = rearrange(v, "b s (g d) -> b s g d", g=self.num_groups)

        try:
            if self.head_dim not in (64, 128):
                raise ValueError("head_dim must be 64 or 128")
            attn_output = self.flash_attn_func(q, k, v)
            attn_output = attn_output.flatten(-2, -1)
        except Exception:
            k = k.repeat_interleave(self.num_heads // self.num_groups, dim=-2)
            v = v.repeat_interleave(self.num_heads // self.num_groups, dim=-2)

            attention_mask = build_alibi_cache(
                k.size(1), self.num_heads, dtype=q.dtype, device=q.device
            )[:, :, -q.size(1) :, :].contiguous()

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            attn_output: torch.Tensor = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask
            )

            attn_output = attn_output.transpose(1, 2).flatten(-2, -1)

        out = self.o_proj(attn_output)
        return out, None  # attn weights are not returned


class StepMLP(torch.nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        x = torch.nn.functional.silu(gate) * up
        x = self.down_proj(x)
        return x


class StepLayer(torch.nn.Module):
    def __init__(self, config: Step1Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = StepAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_groups=config.num_attention_groups,
            layer_idx=layer_idx,
        )
        self.mlp = StepMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )
        self.input_layernorm = StepRMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = StepRMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states, past_key_value, attention_mask, cache_position
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs


class StepPreTrainedModel(PreTrainedModel):
    config_class = Step1Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["StepLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_cache_class = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Step1Model(StepPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers.
    """

    def __init__(self, config: Step1Config):
        super().__init__(config)
        self.config = config
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = torch.nn.Sequential(
            *[
                StepLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = StepRMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        causal_mask = attention_mask
        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                past_key_value=past_key_values,
                cache_position=cache_position,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=None,
        )
        return output if return_dict else output.to_tuple()


class Step1ForCausalLM(StepPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Step1Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# Register with HF auto classes
AutoConfig.register("step1", Step1Config)
AutoModelForCausalLM.register(Step1Config, Step1ForCausalLM)


class UnifiedModelLoader:
    """Local-only unified model loader"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def _prepare_quantization_config(
        self,
        quantization_config: Optional[str],
        torch_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[Dict[str, Any], bool]:
        """
        Prepare quantization configuration for model loading

        Args:
            quantization_config: Quantization type ('int4', 'int8', 'awq-4bit', or None)
            torch_dtype: PyTorch data type for compute operations

        Returns:
            Tuple of (quantization parameters dict, should_set_torch_dtype)
        """
        if not quantization_config:
            return {}, True

        quantization_config = quantization_config.lower()

        if quantization_config == "int8":
            compute_dtype = torch_dtype if torch_dtype is not None else torch.bfloat16
            self.logger.info(
                f"🔧 INT8 quantization: using {compute_dtype} for compute operations"
            )

            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=compute_dtype,
            )
            return {"quantization_config": bnb_config}, False

        elif quantization_config == "int4":
            compute_dtype = torch_dtype if torch_dtype is not None else torch.bfloat16
            self.logger.info(
                f"🔧 INT4 quantization: using {compute_dtype} for compute operations"
            )

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            return {"quantization_config": bnb_config}, False

        elif quantization_config == "awq-4bit":
            self.logger.info("🔧 AWQ 4-bit quantization enabled")
            return {}, True

        else:
            raise ValueError(
                f"Unsupported quantization config: {quantization_config}. "
                f"Supported: 'int4', 'int8', 'awq-4bit'"
            )

    def load_transformers_model(
        self,
        model_path: str,
        quantization_config: Optional[str] = None,
        initial_device: str = "cpu",  # NEW: where to keep non-quant model after load
        **kwargs,
    ) -> Tuple:
        """
        Load Transformers model locally (for StepAudioTTS)

        Args:
            model_path: Local directory containing model weights (model.safetensors)
            quantization_config: 'int4', 'int8', 'awq-4bit', or None
            initial_device: for non-quantized models, final device after loading
                            (default 'cpu' -> CPU-first, then you can .to("cuda") later)
            **kwargs: other parameters (torch_dtype, device_map, etc.)

        Returns:
            (model, tokenizer, resolved_model_path)
        """
        self.logger.info(f"Loading Transformers model from local: {model_path}")
        if quantization_config:
            self.logger.info(f"🔧 {quantization_config.upper()} quantization enabled")

        quantization_kwargs, should_set_torch_dtype = self._prepare_quantization_config(
            quantization_config, kwargs.get("torch_dtype")
        )

        try:
            # Respect explicit device_map if user passes one, otherwise None
            #explicit_device_map = kwargs.get("device_map", None)

            load_kwargs = {
                "trust_remote_code": True,
                "local_files_only": True,
            }

            # Add quantization-related kwargs (BitsAndBytesConfig etc.)
            load_kwargs.update(quantization_kwargs)

            # Device placement strategy:
            # - Quantized (int4/int8/AWQ): usually GPU-only; use device_map or "auto".
            # - Non-quantized: default to device_map=None -> load fully on CPU.
            if quantization_config is not None:
                # quantized path
                #load_kwargs["device_map"] = explicit_device_map or "auto"
                load_kwargs["device_map"] = "auto"
            else:
                # full-precision path: CPU-first unless user *explicitly* gives device_map
                #load_kwargs["device_map"] = explicit_device_map  # often None
                load_kwargs["device_map"] = "cpu"

            if should_set_torch_dtype and kwargs.get("torch_dtype") is not None:
                load_kwargs["torch_dtype"] = kwargs.get("torch_dtype")

            config_path = CONFIG_DIR
            self.logger.info(f"Using configs/tokenizer from: {config_path}")

            # 1) Load config.json from CONFIG_DIR
            config = AutoConfig.from_pretrained(
                config_path,
                trust_remote_code=True,
                local_files_only=True,
            )

            # 2) Load model weights from model_path, with that config
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                **load_kwargs,
            )

            # 3) Load tokenizer from CONFIG_DIR
            tokenizer = AutoTokenizer.from_pretrained(
                config_path,
                trust_remote_code=True,
                local_files_only=True,
            )

            # For quantized models (int4/int8/AWQ):
            # - location is controlled by device_map/quantization_config.
            # - You generally should NOT expect reliable .to('cpu').
            
            # --- annotate model with quantization info so other nodes can check ---
            try:
                # Simple flags for downstream nodes
                model.is_quantized = quantization_config is not None
                model.quantization_type = quantization_config  # e.g. "int4", "int8", "awq-4bit", or None
            except Exception:
                # in case some model type doesn't like setattr, just ignore
                pass
                
            self.logger.info("Successfully loaded model from local")
            return model, tokenizer, model_path

        except Exception as e:
            self.logger.error(f"Failed to load model from local: {e}")
            raise

    def resolve_model_path(
        self,
        base_path: str,
        model_name: str,
    ) -> str:
        """
        Resolve model path locally

        Args:
            base_path: Base directory
            model_name: Subdirectory / model name

        Returns:
            Joined local path
        """
        return os.path.join(base_path, model_name)


# Global instance
model_loader = UnifiedModelLoader()

