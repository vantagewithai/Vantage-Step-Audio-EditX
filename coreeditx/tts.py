import hashlib
import io
import os
import re
import logging
import json
from pathlib import Path

import numpy as np
import torch
import librosa
import soundfile as sf
from typing import Tuple, Optional, Dict, Any
from http import HTTPStatus

import torchaudio

from .model_loader import model_loader
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList

# Configure logging
logger = logging.getLogger(__name__)

# Root directory = folder where model_loader.py lives
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(ROOT_DIR, "cache")
# ----------------------------------------------------

AUDIO_EDIT_CLONE_SYSTEM_PROMPT_TPL = """Generate audio with the following timbre, prosody and speaking style

[speaker_start]
speaker name: {speaker}
speaker prompt text: 
{prompt_text}
speaker audio tokens: 
{prompt_wav_tokens}
[speaker_end]
"""

AUDIO_EDIT_SYSTEM_PROMPT = """As a highly skilled audio editing and tuning specialist, you excel in interpreting user instructions and applying precise adjustments to meet their needs. Your expertise spans a wide range of enhancement capabilities, including but not limited to:
# Emotional Enhancement
# Speaking Style Transfer
# Non-linguistic Adjustments
# Audio Tuning & Editing
Note: You will receive instructions in natural language and are expected to accurately interpret and execute the most suitable audio edits and enhancements.
"""


def safe_move_model(model, device: str):
    """
    Move model between CPU and GPU, but avoid sending quantized models to CPU.
    """
    is_quantized = getattr(model, "is_quantized", False)
    q_type = getattr(model, "quantization_type", None)

    # Quantized models are generally GPU-only
    if is_quantized and device == "cpu":
        print(f"Skipping move to CPU for quantized model ({q_type})")
        return model

    return model.to(device)


class HTTPException(Exception):
    """Custom HTTP exception for API errors"""

    def __init__(self, status_code, detail):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


class RepetitionAwareLogitsProcessor(LogitsProcessor):
    """Logits processor to handle repetition in generation"""

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        window_size = 10
        threshold = 0.1

        window = input_ids[:, -window_size:]
        if window.shape[1] < window_size:
            return scores

        last_tokens = window[:, -1].unsqueeze(-1)
        repeat_counts = (window == last_tokens).sum(dim=1)
        repeat_ratios = repeat_counts.float() / window_size

        mask = repeat_ratios > threshold
        scores[mask, last_tokens[mask].squeeze(-1)] = float("-inf")
        return scores


class StepAudioTTS:
    """
    Step Audio TTS wrapper for voice cloning and audio editing tasks.

    Features:
      - Disk-based + in-memory caching of:
          * clone() speaker context for (prompt_wav_path + prompt_text)
          * edit() audio feature context for input_audio_path
      - Lazy CosyVoice loading/unloading
      - LLM movement between CPU and GPU via safe_move_model
    """

    def __init__(
        self,
        model_path,
        audio_tokenizer,
        quantization_config=None,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        cache_dir: Optional[str] = None,
        enable_disk_cache: bool = True,
        enable_edit_disk_cache: bool = False,
    ):
        """
        Initialize StepAudioTTS (local-only)

        Args:
            model_path: Local model path (directory with model.safetensors)
            audio_tokenizer: Audio tokenizer for wav2token processing
            quantization_config: Quantization config ('int4', 'int8', 'awq-4bit', or None)
            torch_dtype: PyTorch dtype for model weights (default: torch.bfloat16)
            device_map: Device mapping for model (default: "cuda" or "auto")
            cache_dir: Directory for disk-based context cache
            enable_disk_cache: Enable/disable disk-based caching
        """
        # Keep model path for cache versioning / info
        self.model_path = os.path.abspath(model_path)

        # Determine model ID or path to load (local)
        tts_model_id = model_path

        logger.info("🔧 StepAudioTTS loading configuration:")
        logger.info(f"   - model_path: {model_path}")
        logger.info(f"   - tts_model_id (local): {tts_model_id}")
        logger.info(f"   - quantization_config: {quantization_config}")
        logger.info(f"   - torch_dtype: {torch_dtype}")
        logger.info(f"   - device_map: {device_map}")

        self.audio_tokenizer = audio_tokenizer

        # Load LLM and tokenizer using local-only model_loader
        try:
            self.llm, self.tokenizer, model_path = model_loader.load_transformers_model(
                tts_model_id,
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
            logger.info(
                f"✅ Successfully loaded LLM and tokenizer from local path: {tts_model_id}"
            )
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise

        # CosyVoice model is loaded lazily
        self.cosy_model = None
        self.cosy_model_dir = os.path.join(model_path, "CosyVoice-300M-25Hz")

        # System prompts
        self.edit_clone_sys_prompt_tpl = AUDIO_EDIT_CLONE_SYSTEM_PROMPT_TPL
        self.edit_sys_prompt = AUDIO_EDIT_SYSTEM_PROMPT

        # Clone / edit context cache configuration
        self.enable_disk_cache = enable_disk_cache
        self.enable_edit_disk_cache = enable_edit_disk_cache
        if cache_dir is None:
            # Default under model path
            cache_dir = CACHE_DIR
        self.cache_dir = Path(cache_dir) if enable_disk_cache or enable_edit_disk_cache else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"🧠 Context disk cache directory: {self.cache_dir}")

        # Simple in-memory caches
        self._clone_context_mem_cache: Dict[str, Dict[str, Any]] = {}
        self._edit_context_mem_cache: Dict[str, Dict[str, Any]] = {}

        # Cache versioning – bump if you change context formats
        self.ctx_cache_version = "stepaudio_tts_ctx_v1"
        if(not self.audio_tokenizer.offload):
            self.audio_tokenizer.load_tokenizer()
            self._load_cosy_model()

    # -------------------------------------------------------------------------
    # CosyVoice load / unload helpers
    # -------------------------------------------------------------------------

    def _load_cosy_model(self):
        """
        Lazy-load CosyVoice model.
        """
        if self.cosy_model is None:
            from stepvocoder.cosyvoice2.cli.cosyvoice import CosyVoice

            self.cosy_model = CosyVoice(self.cosy_model_dir)
            #logger.info("🎤 CosyVoice model loaded successfully")

    def _unload_cosy_model(self):
        """
        Unload CosyVoice and free VRAM.
        """
        if self.cosy_model is not None:
            del self.cosy_model
            self.cosy_model = None
            torch.cuda.empty_cache()
            #logger.info("🧹 CosyVoice model unloaded and VRAM cleared")

    # -------------------------------------------------------------------------
    # Core utilities
    # -------------------------------------------------------------------------

    def process_audio_file(self, audio_path: str) -> Tuple[Any, int]:
        """
        Process audio file and return numpy array and sample rate
        """
        try:
            audio_data, sample_rate = librosa.load(audio_path)
            logger.debug(f"Audio file processed successfully: {audio_path}")
            return audio_data, sample_rate
        except Exception as e:
            logger.error(f"Failed to process audio file: {e}")
            raise

    def preprocess_prompt_wav(self, prompt_wav_path: str):
        """
        Extract speech features, speaker embedding and audio tokens
        for the given prompt wav. (CosyVoice must be loaded)
        """
        prompt_wav, prompt_wav_sr = torchaudio.load(prompt_wav_path)
        if prompt_wav.shape[0] > 1:
            prompt_wav = prompt_wav.mean(dim=0, keepdim=True)

        # volume-normalize avoid clipping
        norm = torch.max(torch.abs(prompt_wav), dim=1, keepdim=True)[0]
        if norm > 0.6:  # max absolute value is 0.6
            prompt_wav = prompt_wav / norm * 0.6

        speech_feat, speech_feat_len = self.cosy_model.frontend.extract_speech_feat(
            prompt_wav, prompt_wav_sr
        )
        speech_embedding = self.cosy_model.frontend.extract_spk_embedding(
            prompt_wav, prompt_wav_sr
        )
        (
            vq0206_codes,
            vq02_codes_ori,
            vq06_codes_ori,
        ) = self.audio_tokenizer.wav2token(prompt_wav, prompt_wav_sr)
        return (
            vq0206_codes,
            vq02_codes_ori,
            vq06_codes_ori,
            speech_feat,
            speech_feat_len,
            speech_embedding,
        )

    def generate_clone_voice_id(self, prompt_text, prompt_wav):
        """
        Deterministic ID derived from prompt_text + a sample of prompt_wav.
        """
        hasher = hashlib.sha256()
        hasher.update(prompt_text.encode("utf-8"))
        wav_data = prompt_wav.cpu().numpy()
        if wav_data.size > 2000:
            audio_sample = np.concatenate(
                [wav_data.flatten()[:1000], wav_data.flatten()[-1000:]]
            )
        else:
            audio_sample = wav_data.flatten()
        hasher.update(audio_sample.tobytes())
        voice_hash = hasher.hexdigest()[:16]
        return f"clone_{voice_hash}"

    # -------------------------------------------------------------------------
    # Audio-edit prompt building and encoding
    # -------------------------------------------------------------------------

    def _build_audio_edit_instruction(
        self,
        audio_text: str,
        edit_type: str,
        edit_info: Optional[str] = None,
        text: Optional[str] = None,
    ) -> str:
        """
        Build audio editing instruction based on request
        """
        audio_text = audio_text.strip() if audio_text else ""
        if edit_type in {"emotion", "speed"}:
            if edit_info == "remove":
                instruct_prefix = (
                    f"Remove any emotion in the following audio and the reference text is: {audio_text}\n"
                )
            else:
                instruct_prefix = (
                    f"Make the following audio more {edit_info}. "
                    f"The text corresponding to the audio is: {audio_text}\n"
                )
        elif edit_type == "style":
            if edit_info == "remove":
                instruct_prefix = (
                    f"Remove any speaking styles in the following audio and the reference text is: {audio_text}\n"
                )
            else:
                instruct_prefix = (
                    f"Make the following audio more {edit_info} style. "
                    f"The text corresponding to the audio is: {audio_text}\n"
                )
        elif edit_type == "denoise":
            instruct_prefix = (
                "Remove any noise from the given audio while preserving the voice content clearly. "
                "Ensure that the speech quality remains intact with minimal distortion, and eliminate all noise from the audio.\n"
            )
        elif edit_type == "vad":
            instruct_prefix = (
                "Remove any silent portions from the given audio while preserving the voice content clearly. "
                "Ensure that the speech quality remains intact with minimal distortion, and eliminate all silence from the audio.\n"
            )
        elif edit_type == "paralinguistic":
            instruct_prefix = (
                f"Add some non-verbal sounds to make the audio more natural, the new text is : {text}\n"
                f"  The text corresponding to the audio is: {audio_text}\n"
            )
        else:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail=f"Unsupported edit_type: {edit_type}",
            )

        return instruct_prefix

    def _encode_audio_edit_prompt(
        self, sys_prompt: str, instruct_prefix: str, audio_token_str: str
    ) -> list[int]:
        """
        Encode audio edit prompt to token sequence
        """
        audio_token_str = audio_token_str.strip()
        history = [1]
        sys_tokens = self.tokenizer.encode(f"system\n{sys_prompt}")
        history.extend([4] + sys_tokens + [3])
        qrole_toks = self.tokenizer.encode("human\n")
        arole_toks = self.tokenizer.encode("assistant\n")
        human_turn_toks = self.tokenizer.encode(
            f"{instruct_prefix}\n{audio_token_str}\n"
        )
        history.extend([4] + qrole_toks + human_turn_toks + [3] + [4] + arole_toks)
        return history

    def _encode_audio_edit_clone_prompt(
        self, text: str, prompt_text: str, prompt_speaker: str, prompt_wav_tokens: str
    ):
        """
        Original (non-cached) encode for clone – kept for compatibility.
        New clone() uses a cached version of the system/human prefix instead.
        """
        prompt = self.edit_clone_sys_prompt_tpl.format(
            speaker=prompt_speaker,
            prompt_text=prompt_text,
            prompt_wav_tokens=prompt_wav_tokens,
        )
        sys_tokens = self.tokenizer.encode(f"system\n{prompt}")

        history = [1]
        history.extend([4] + sys_tokens + [3])

        _prefix_tokens = self.tokenizer.encode("\n")
        target_token_encode = self.tokenizer.encode("\n" + text)
        target_tokens = target_token_encode[len(_prefix_tokens):]

        qrole_toks = self.tokenizer.encode("human\n")
        arole_toks = self.tokenizer.encode("assistant\n")

        history.extend(
            [4] + qrole_toks + target_tokens + [3] + [4] + arole_toks
        )
        return history

    def detect_instruction_name(self, text):
        instruction_name = ""
        match_group = re.match(r"^([（\(][^\(\)()]*[）\)]).*$", text, re.DOTALL)
        if match_group is not None:
            instruction = match_group.group(1)
            instruction_name = instruction.strip("()（）")
        return instruction_name

    # -------------------------------------------------------------------------
    # Shared cache helpers
    # -------------------------------------------------------------------------

    def _ctx_cache_path(self, key: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{key}.pt"

    def _load_context_from_disk(self, key: str) -> Optional[Dict[str, Any]]:
        cache_path = self._ctx_cache_path(key)
        if cache_path is None or not cache_path.exists():
            return None
        try:
            ctx = torch.load(cache_path, map_location="cpu")
            logger.debug(f"Loaded context from disk cache: {cache_path}")
            return ctx
        except Exception as e:
            logger.warning(f"Failed to load context cache ({cache_path}): {e}")
            # If corrupt, best-effort delete
            try:
                cache_path.unlink()
            except Exception:
                pass
            return None

    def _save_context_to_disk(self, key: str, context: Dict[str, Any]) -> None:
        cache_path = self._ctx_cache_path(key)
        if cache_path is None:
            return

        # Move tensors to CPU before saving
        safe_ctx: Dict[str, Any] = {}
        for k, v in context.items():
            if torch.is_tensor(v):
                safe_ctx[k] = v.cpu()
            else:
                safe_ctx[k] = v

        try:
            torch.save(safe_ctx, cache_path)
            logger.debug(f"Saved context to disk cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save context cache ({cache_path}): {e}")

    # -------------------------------------------------------------------------
    # Clone context caching
    # -------------------------------------------------------------------------

    def _clone_cache_key(self, prompt_wav_path: str, prompt_text: str) -> str:
        """
        Build a deterministic cache key for clone() from:
          - audio file bytes
          - prompt_text
          - model path + internal cache version
        """
        p = Path(prompt_wav_path)
        audio_bytes = p.read_bytes()
        audio_hash = hashlib.sha256(audio_bytes).hexdigest()

        payload = {
            "kind": "clone",
            "audio_hash": audio_hash,
            "prompt_text": prompt_text,
            "model_path": self.model_path,
            "cache_version": self.ctx_cache_version,
        }
        key_json = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(key_json).hexdigest()

    def _get_or_build_clone_context(
        self,
        prompt_wav_path: str,
        prompt_text: str,
    ) -> Dict[str, Any]:
        """
        Load clone context from memory/disk, or build it if missing.
        Context includes:
          - vq0206_codes_vocoder (for vocoder)
          - speech_feat
          - speech_embedding
          - base_history_prefix (tokens up to 'human\\n' before target text)
          - arole_toks ('assistant\\n' tokens)
          - prompt_speaker
          - prompt_wav_tokens
        """
        key = self._clone_cache_key(prompt_wav_path, prompt_text)

        # 1) In-memory cache
        if key in self._clone_context_mem_cache:
            logger.debug(f"Loading from in memory cache")
            return self._clone_context_mem_cache[key]

        # 2) Disk cache
        ctx = self._load_context_from_disk(key)
        if ctx is not None:
            logger.debug(f"Loading from disk cache")
            self._clone_context_mem_cache[key] = ctx
            return ctx

        # 3) Build context
        logger.debug(f"Clone cache does not exists, building context for key={key}")
        prompt_wav, _ = torchaudio.load(prompt_wav_path)

        # Extract features + tokens using CosyVoice + audio_tokenizer
        if(self.audio_tokenizer.offload):
            self._load_cosy_model()
            self.audio_tokenizer.load_tokenizer()
        (
            vq0206_codes,
            vq02_codes_ori,
            vq06_codes_ori,
            speech_feat,
            _speech_feat_len,
            speech_embedding,
        ) = self.preprocess_prompt_wav(prompt_wav_path)
        # We don't need CosyVoice until vocoder stage again
        if(self.audio_tokenizer.offload):
            self._unload_cosy_model()

        prompt_speaker = self.generate_clone_voice_id(prompt_text, prompt_wav)
        prompt_wav_tokens = self.audio_tokenizer.merge_vq0206_to_token_str(
            vq02_codes_ori, vq06_codes_ori
        )
        
        if(self.audio_tokenizer.offload):
            self.audio_tokenizer.unload_tokenizer()

        # Build system side of the prompt once
        prompt = self.edit_clone_sys_prompt_tpl.format(
            speaker=prompt_speaker,
            prompt_text=prompt_text,
            prompt_wav_tokens=prompt_wav_tokens,
        )
        sys_tokens = self.tokenizer.encode(f"system\n{prompt}")

        history = [1]  # BOS
        history.extend([4] + sys_tokens + [3])  # [4]=role sep, [3]=eos/sep

        qrole_toks = self.tokenizer.encode("human\n")
        arole_toks = self.tokenizer.encode("assistant\n")

        # This is everything before the target text
        base_history_prefix = history + [4] + qrole_toks

        # Prepare vocoder codes (on CPU)
        vq0206_codes_vocoder = torch.tensor([vq0206_codes], dtype=torch.long) - 65536

        ctx = {
            "vq0206_codes_vocoder": vq0206_codes_vocoder,
            "speech_feat": speech_feat,
            "speech_embedding": speech_embedding,
            "base_history_prefix": base_history_prefix,
            "arole_toks": arole_toks,
            "prompt_speaker": prompt_speaker,
            "prompt_wav_tokens": prompt_wav_tokens,
        }

        # Store in caches
        self._clone_context_mem_cache[key] = ctx
        if self.enable_disk_cache:
            self._save_context_to_disk(key, ctx)

        return ctx

    # -------------------------------------------------------------------------
    # Edit context caching
    # -------------------------------------------------------------------------

    def _edit_cache_key(self, input_audio_path: str) -> str:
        """
        Build a deterministic cache key for edit() from:
          - audio file bytes
          - model path + internal cache version

        Note: does NOT depend on audio_text or edit_type; those affect
        instructions, not audio features.
        """
        p = Path(input_audio_path)
        audio_bytes = p.read_bytes()
        audio_hash = hashlib.sha256(audio_bytes).hexdigest()

        payload = {
            "kind": "edit",
            "audio_hash": audio_hash,
            "model_path": self.model_path,
            "cache_version": self.ctx_cache_version,
        }
        key_json = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(key_json).hexdigest()

    def _get_or_build_edit_context(
        self,
        input_audio_path: str,
    ) -> Dict[str, Any]:
        """
        Load edit context from memory/disk, or build it if missing.
        Context includes:
          - vq0206_codes_vocoder (for vocoder)
          - speech_feat
          - speech_embedding
          - audio_tokens_str (string from vq02/vq06)
        """
        key = self._edit_cache_key(input_audio_path)

        # 1) In-memory cache
        if key in self._edit_context_mem_cache:
            logger.debug(f"Loading from in memory cache")
            return self._edit_context_mem_cache[key]

        # 2) Disk cache
        ctx = self._load_context_from_disk(key)
        if ctx is not None:
            logger.debug(f"Loading from disk cache")
            self._edit_context_mem_cache[key] = ctx
            return ctx

        # 3) Build context
        logger.debug(f"Edit cache does not exists, building context for key={key}")

        # Extract features + tokens using CosyVoice + audio_tokenizer
        
        if(self.audio_tokenizer.offload):
            self._load_cosy_model()
            self.audio_tokenizer.load_tokenizer()
        (
            vq0206_codes,
            vq02_codes_ori,
            vq06_codes_ori,
            speech_feat,
            _speech_feat_len,
            speech_embedding,
        ) = self.preprocess_prompt_wav(input_audio_path)
        # After preprocessing, CosyVoice can be unloaded until vocoder stage
        if(self.audio_tokenizer.offload):
            self._unload_cosy_model()

        audio_tokens_str = self.audio_tokenizer.merge_vq0206_to_token_str(
            vq02_codes_ori, vq06_codes_ori
        )
        if(self.audio_tokenizer.offload):
            self.audio_tokenizer.unload_tokenizer()

        vq0206_codes_vocoder = torch.tensor([vq0206_codes], dtype=torch.long) - 65536

        ctx = {
            "vq0206_codes_vocoder": vq0206_codes_vocoder,
            "speech_feat": speech_feat,
            "speech_embedding": speech_embedding,
            "audio_tokens_str": audio_tokens_str,
        }

        self._edit_context_mem_cache[key] = ctx
        if self.enable_edit_disk_cache:
            self._save_context_to_disk(key, ctx)

        return ctx

        # -------------------------------------------------------------------------
    # Edit context caching for tensor input
    # -------------------------------------------------------------------------

    def _edit_cache_key_from_tensor(self, waveform: torch.Tensor, sr: int) -> str:
        """
        Build a deterministic cache key for edit() when given a waveform tensor
        instead of a file path.

        Uses:
          - audio waveform bytes
          - sample rate
          - model path + internal cache version
        """
        # Ensure on CPU for hashing
        wav_np = waveform.detach().cpu().numpy()
        audio_bytes = wav_np.tobytes()
        audio_hash = hashlib.sha256(audio_bytes).hexdigest()

        payload = {
            "kind": "edit_tensor",
            "audio_hash": audio_hash,
            "sr": int(sr),
            "model_path": self.model_path,
            "cache_version": self.ctx_cache_version,
        }
        key_json = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(key_json).hexdigest()

    def _get_or_build_edit_context_from_tensor(
        self,
        waveform: torch.Tensor,
        sr: int,
    ) -> Dict[str, Any]:
        """
        Load edit context from memory/disk, or build it if missing, using
        an in-memory waveform instead of an audio file path.

        Context includes:
          - vq0206_codes_vocoder (for vocoder)
          - speech_feat
          - speech_embedding
          - audio_tokens_str (string from vq02/vq06)
        """
        # Normalize shape to [1, T] mono
        wav = waveform
        if wav.dim() == 3:
            # [B, C, T] -> assume B=1
            wav = wav.squeeze(0)
        if wav.dim() == 2:
            # [C, T] -> average channels
            wav = wav.mean(dim=0, keepdim=True)  # [1, T]
        elif wav.dim() == 1:
            wav = wav.unsqueeze(0)  # [1, T]

        key = self._edit_cache_key_from_tensor(wav, sr)

        # 1) In-memory cache
        if key in self._edit_context_mem_cache:
            logger.debug("Loading edit tensor context from in-memory cache")
            return self._edit_context_mem_cache[key]

        # 2) Disk cache
        ctx = self._load_context_from_disk(key)
        if ctx is not None:
            logger.debug("Loading edit tensor context from disk cache")
            self._edit_context_mem_cache[key] = ctx
            return ctx

        # 3) Build context
        logger.debug(f"Edit tensor cache does not exist, building context for key={key}")

        # Extract features + tokens using CosyVoice + audio_tokenizer
        if(self.audio_tokenizer.offload):
            self._load_cosy_model()
            self.audio_tokenizer.load_tokenizer()

        # volume-normalize to match preprocess_prompt_wav()
        norm = torch.max(torch.abs(wav), dim=1, keepdim=True)[0]
        if (norm > 0.6).any():
            wav = wav / norm * 0.6

        speech_feat, _speech_feat_len = self.cosy_model.frontend.extract_speech_feat(
            wav, sr
        )
        speech_embedding = self.cosy_model.frontend.extract_spk_embedding(
            wav, sr
        )
        (
            vq0206_codes,
            vq02_codes_ori,
            vq06_codes_ori,
        ) = self.audio_tokenizer.wav2token(wav, sr)

        audio_tokens_str = self.audio_tokenizer.merge_vq0206_to_token_str(
            vq02_codes_ori, vq06_codes_ori
        )

        if(self.audio_tokenizer.offload):
            self.audio_tokenizer.unload_tokenizer()
            self._unload_cosy_model()

        vq0206_codes_vocoder = torch.tensor([vq0206_codes], dtype=torch.long) - 65536

        ctx = {
            "vq0206_codes_vocoder": vq0206_codes_vocoder,
            "speech_feat": speech_feat,
            "speech_embedding": speech_embedding,
            "audio_tokens_str": audio_tokens_str,
        }

        self._edit_context_mem_cache[key] = ctx
        if self.enable_edit_disk_cache:
            self._save_context_to_disk(key, ctx)

        return ctx

        # -------------------------------------------------------------------------
    # Clone context caching for tensor input
    # -------------------------------------------------------------------------

    def _clone_cache_key_from_tensor(
        self,
        waveform: torch.Tensor,
        sr: int,
        prompt_text: str,
    ) -> str:
        """
        Build a deterministic cache key for clone() when given a waveform tensor
        instead of a file path.

        Uses:
          - audio waveform bytes
          - sample rate
          - prompt_text
          - model path + internal cache version
        """
        # Ensure on CPU for hashing
        wav_np = waveform.detach().cpu().numpy()
        audio_bytes = wav_np.tobytes()
        audio_hash = hashlib.sha256(audio_bytes).hexdigest()

        payload = {
            "kind": "clone_tensor",
            "audio_hash": audio_hash,
            "sr": int(sr),
            "prompt_text": prompt_text,
            "model_path": self.model_path,
            "cache_version": self.ctx_cache_version,
        }
        key_json = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(key_json).hexdigest()

    def _get_or_build_clone_context_from_tensor(
        self,
        waveform: torch.Tensor,
        sr: int,
        prompt_text: str,
    ) -> Dict[str, Any]:
        """
        Load clone context from memory/disk, or build it if missing,
        using an in-memory waveform instead of an audio file path.

        Context includes:
          - vq0206_codes_vocoder (for vocoder)
          - speech_feat
          - speech_embedding
          - base_history_prefix (tokens up to 'human\\n' before target text)
          - arole_toks ('assistant\\n' tokens)
          - prompt_speaker
          - prompt_wav_tokens
        """
        # Normalize shape to mono [1, T]
        wav = waveform
        if wav.dim() == 3:
            # [B, C, T] -> assume B=1
            wav = wav.squeeze(0)
        if wav.dim() == 2:
            # [C, T] -> average channels
            wav = wav.mean(dim=0, keepdim=True)  # [1, T]
        elif wav.dim() == 1:
            wav = wav.unsqueeze(0)  # [1, T]

        key = self._clone_cache_key_from_tensor(wav, sr, prompt_text)

        # 1) In-memory cache
        if key in self._clone_context_mem_cache:
            logger.debug("Loading clone tensor context from in-memory cache")
            return self._clone_context_mem_cache[key]

        # 2) Disk cache
        ctx = self._load_context_from_disk(key)
        if ctx is not None:
            logger.debug("Loading clone tensor context from disk cache")
            self._clone_context_mem_cache[key] = ctx
            return ctx

        # 3) Build context
        logger.debug(f"Clone tensor cache does not exist, building context for key={key}")

        # Extract features + tokens using CosyVoice + audio_tokenizer
        if(self.audio_tokenizer.offload):
            self._load_cosy_model()
            self.audio_tokenizer.load_tokenizer()

        # volume-normalize to match preprocess_prompt_wav()
        norm = torch.max(torch.abs(wav), dim=1, keepdim=True)[0]
        if (norm > 0.6).any():
            wav = wav / norm * 0.6

        speech_feat, _speech_feat_len = self.cosy_model.frontend.extract_speech_feat(
            wav, sr
        )
        speech_embedding = self.cosy_model.frontend.extract_spk_embedding(
            wav, sr
        )
        (
            vq0206_codes,
            vq02_codes_ori,
            vq06_codes_ori,
        ) = self.audio_tokenizer.wav2token(wav, sr)

        prompt_speaker = self.generate_clone_voice_id(prompt_text, wav)
        prompt_wav_tokens = self.audio_tokenizer.merge_vq0206_to_token_str(
            vq02_codes_ori, vq06_codes_ori
        )

        if(self.audio_tokenizer.offload):
            self.audio_tokenizer.unload_tokenizer()
            self._unload_cosy_model()

        # Build system side of the prompt once (same as file-path version)
        prompt = self.edit_clone_sys_prompt_tpl.format(
            speaker=prompt_speaker,
            prompt_text=prompt_text,
            prompt_wav_tokens=prompt_wav_tokens,
        )
        sys_tokens = self.tokenizer.encode(f"system\n{prompt}")

        history = [1]  # BOS
        history.extend([4] + sys_tokens + [3])  # [4]=role sep, [3]=eos/sep

        qrole_toks = self.tokenizer.encode("human\n")
        arole_toks = self.tokenizer.encode("assistant\n")

        # This is everything before the target text
        base_history_prefix = history + [4] + qrole_toks

        vq0206_codes_vocoder = torch.tensor([vq0206_codes], dtype=torch.long) - 65536

        ctx = {
            "vq0206_codes_vocoder": vq0206_codes_vocoder,
            "speech_feat": speech_feat,
            "speech_embedding": speech_embedding,
            "base_history_prefix": base_history_prefix,
            "arole_toks": arole_toks,
            "prompt_speaker": prompt_speaker,
            "prompt_wav_tokens": prompt_wav_tokens,
        }

        self._clone_context_mem_cache[key] = ctx
        if self.enable_disk_cache:
            self._save_context_to_disk(key, ctx)

        return ctx


    # -------------------------------------------------------------------------
    # Voice cloning (with caching and lazy loading)
    # -------------------------------------------------------------------------

    def clone(
        self,
        prompt_wav_path: str,
        prompt_text: str,
        target_text: str,
    ) -> Tuple[torch.Tensor, int]:
        """
        Clone voice from reference audio.

        Uses a cache keyed by (audio bytes + prompt_text + model) so that
        repeated calls with the same reference do not re-run:
          - CosyVoice frontend (speech_feat, speech_embedding)
          - audio_tokenizer.wav2token
          - system+speaker prompt tokenization

        Args:
            prompt_wav_path: Path to reference audio file
            prompt_text: Text content of reference audio
            target_text: Text to synthesize with cloned voice

        Returns:
            Tuple[torch.Tensor, int]: Generated audio tensor and sample rate
        """
        try:
            logger.debug(f"Starting voice cloning: {prompt_wav_path}")

            # Get clone context (from memory/disk or compute)
            ctx = self._get_or_build_clone_context(prompt_wav_path, prompt_text)

            base_history_prefix: list[int] = ctx["base_history_prefix"]
            arole_toks: list[int] = ctx["arole_toks"]
            vq0206_codes_vocoder: torch.Tensor = ctx["vq0206_codes_vocoder"]
            speech_feat: torch.Tensor = ctx["speech_feat"]
            speech_embedding: torch.Tensor = ctx["speech_embedding"]

            # Encode only the target text tokens
            _prefix_tokens = self.tokenizer.encode("\n")
            target_token_encode = self.tokenizer.encode("\n" + target_text)
            target_tokens = target_token_encode[len(_prefix_tokens):]

            token_ids = (
                base_history_prefix
                + target_tokens
                + [3]  # end of human turn
                + [4]
                + arole_toks  # assistant\n
            )

            # Run LLM on GPU (lazy move + return to CPU)
            self.llm = safe_move_model(self.llm, "cuda:0")
            input_ids = torch.tensor([token_ids]).to(torch.long).to("cuda")

            output_ids = self.llm.generate(
                input_ids,
                max_length=8192,
                temperature=0.7,
                do_sample=True,
                logits_processor=LogitsProcessorList(
                    [RepetitionAwareLogitsProcessor()]
                ),
            )
            self.llm = safe_move_model(self.llm, "cpu")

            # Remove prompt part and final eos
            output_ids = output_ids[:, len(token_ids): -1]
            logger.debug("Voice cloning generation completed")

            # Vocoder: move cached features to GPU
            vq0206_codes_vocoder_gpu = vq0206_codes_vocoder.to("cuda")
            speech_feat_gpu = speech_feat.to(torch.bfloat16).to("cuda")
            speech_embedding_gpu = speech_embedding.to(torch.bfloat16).to("cuda")

            torch.cuda.empty_cache()
            self._load_cosy_model()

            try:
                audio = self.cosy_model.token2wav_nonstream(
                    output_ids - 65536,
                    vq0206_codes_vocoder_gpu,
                    speech_feat_gpu,
                    speech_embedding_gpu,
                )
            finally:
                # Always unload CosyVoice (even if token2wav throws)
                self._unload_cosy_model()

            return audio, 24000
        except Exception as e:
            logger.error(f"Clone failed: {e}")
            raise

    # -------------------------------------------------------------------------
    # Audio editing (with caching and lazy loading)
    # -------------------------------------------------------------------------

    def edit(
        self,
        input_audio_path: str,
        audio_text: str,
        edit_type: str,
        edit_info: Optional[str] = None,
        text: Optional[str] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Edit audio based on specified edit type.

        Uses a cache keyed by (audio bytes + model) so that repeated edits
        on the same base audio do not re-run:
          - CosyVoice frontend (speech_feat, speech_embedding)
          - audio_tokenizer.wav2token

        Args:
            input_audio_path: Path to input audio file
            audio_text: Text content of input audio
            edit_type: Type of edit (emotion, style, speed, denoise, vad, paralinguistic)
            edit_info: Specific edit information (happy, sad, remove, etc.)
            text: Target text for para-linguistic editing

        Returns:
            Tuple[torch.Tensor, int]: Edited audio tensor and sample rate
        """
        try:
            logger.debug(
                f"Starting audio editing: path={input_audio_path}, type={edit_type}, info={edit_info}"
            )

            # Get edit context (from memory/disk or compute)
            ctx = self._get_or_build_edit_context(input_audio_path)

            vq0206_codes_vocoder: torch.Tensor = ctx["vq0206_codes_vocoder"]
            speech_feat: torch.Tensor = ctx["speech_feat"]
            speech_embedding: torch.Tensor = ctx["speech_embedding"]
            audio_tokens_str: str = ctx["audio_tokens_str"]

            # Build instruction prefix based on edit type
            instruct_prefix = self._build_audio_edit_instruction(
                audio_text, edit_type, edit_info, text
            )

            # Encode the complete prompt to token sequence
            prompt_tokens = self._encode_audio_edit_prompt(
                self.edit_sys_prompt, instruct_prefix, audio_tokens_str
            )

            logger.debug(f"Edit instruction: {instruct_prefix}")
            logger.debug(f"Encoded prompt length: {len(prompt_tokens)}")

            # LLM on GPU with lazy movement
            self.llm = safe_move_model(self.llm, "cuda:0")
            input_ids = torch.tensor([prompt_tokens]).to(torch.long).to("cuda")

            output_ids = self.llm.generate(
                input_ids,
                max_length=8192,
                temperature=0.7,
                do_sample=True,
                logits_processor=LogitsProcessorList(
                    [RepetitionAwareLogitsProcessor()]
                ),
            )
            self.llm = safe_move_model(self.llm, "cpu")

            # Strip prompt tokens and final eos
            output_ids = output_ids[:, len(prompt_tokens): -1]  # skip eos token
            logger.debug("Audio editing generation completed")

            # Vocoder using cached features (lazy Cosy load/unload)
            vq0206_codes_vocoder_gpu = vq0206_codes_vocoder.to("cuda")
            speech_feat_gpu = speech_feat.to(torch.bfloat16).to("cuda")
            speech_embedding_gpu = speech_embedding.to(torch.bfloat16).to("cuda")

            torch.cuda.empty_cache()
            self._load_cosy_model()

            try:
                audio = self.cosy_model.token2wav_nonstream(
                    output_ids - 65536,
                    vq0206_codes_vocoder_gpu,
                    speech_feat_gpu,
                    speech_embedding_gpu,
                )
            finally:
                self._unload_cosy_model()

            return audio, 24000
        except Exception as e:
            logger.error(f"Edit failed: {e}")
            raise

        # -------------------------------------------------------------------------
    # Audio editing directly from tensor (ComfyUI Load Audio output)
    # -------------------------------------------------------------------------

    def edit_from_tensor(
        self,
        audio,
        sr: Optional[int],
        audio_text: str,
        edit_type: str,
        edit_info: Optional[str] = None,
        text: Optional[str] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, int]:
        """
        Edit audio when the input comes from a Load Audio node / tensor,
        instead of a file path.

        Expected usage from ComfyUI:
            result_audio, result_sr = tts.edit_from_tensor(
                audio_in["waveform"],      # or audio_in dict
                audio_in["sample_rate"],
                audio_text,
                edit_type,
                edit_info,
                text,
            )

        Args:
            audio:
                - Either a dict: {"waveform": tensor[B,C,T], "sample_rate": sr}
                - Or a torch.Tensor waveform
            sr: sample rate (ignored if 'audio' is dict with 'sample_rate')
            audio_text: Text content of input audio
            edit_type: Type of edit (emotion, style, speed, denoise, vad, paralinguistic)
            edit_info: Specific edit information (happy, sad, remove, etc.)
            text: Target text for para-linguistic editing

        Returns:
            Tuple[torch.Tensor, int]: Edited audio tensor and sample rate (24000)
        """
        self.enable_edit_disk_cache = use_cache
        try:
            # Accept dict from Load Audio node or bare tensor
            if isinstance(audio, dict):
                waveform = audio["waveform"]
                sr_local = int(audio["sample_rate"])
            else:
                waveform = audio
                if sr is None:
                    raise ValueError(
                        "edit_from_tensor: 'sr' must be provided when 'audio' is a tensor."
                    )
                sr_local = int(sr)

            logger.debug(
                f"Starting tensor audio editing: sr={sr_local}, "
                f"type={edit_type}, info={edit_info}"
            )

            # Get edit context (from memory/disk or compute) based on waveform
            ctx = self._get_or_build_edit_context_from_tensor(waveform, sr_local)

            vq0206_codes_vocoder: torch.Tensor = ctx["vq0206_codes_vocoder"]
            speech_feat: torch.Tensor = ctx["speech_feat"]
            speech_embedding: torch.Tensor = ctx["speech_embedding"]
            audio_tokens_str: str = ctx["audio_tokens_str"]

            # Build instruction prefix based on edit type
            instruct_prefix = self._build_audio_edit_instruction(
                audio_text, edit_type, edit_info, text
            )

            # Encode the complete prompt to token sequence
            prompt_tokens = self._encode_audio_edit_prompt(
                self.edit_sys_prompt, instruct_prefix, audio_tokens_str
            )

            logger.debug(f"Edit instruction: {instruct_prefix}")
            logger.debug(f"Encoded prompt length: {len(prompt_tokens)}")

            # LLM on GPU with lazy movement
            self.llm = safe_move_model(self.llm, "cuda:0")
            input_ids = torch.tensor([prompt_tokens]).to(torch.long).to("cuda")

            output_ids = self.llm.generate(
                input_ids,
                max_length=8192,
                temperature=0.7,
                do_sample=True,
                logits_processor=LogitsProcessorList(
                    [RepetitionAwareLogitsProcessor()]
                ),
            )
            self.llm = safe_move_model(self.llm, "cpu")

            # Strip prompt tokens and final eos
            output_ids = output_ids[:, len(prompt_tokens): -1]
            logger.debug("Tensor audio editing generation completed")

            # Vocoder using cached features (lazy Cosy load/unload)
            vq0206_codes_vocoder_gpu = vq0206_codes_vocoder.to("cuda")
            speech_feat_gpu = speech_feat.to(torch.bfloat16).to("cuda")
            speech_embedding_gpu = speech_embedding.to(torch.bfloat16).to("cuda")

            torch.cuda.empty_cache()
            self._load_cosy_model()

            try:
                audio_out = self.cosy_model.token2wav_nonstream(
                    output_ids - 65536,
                    vq0206_codes_vocoder_gpu,
                    speech_feat_gpu,
                    speech_embedding_gpu,
                )
            finally:
                self._unload_cosy_model()

            # Return exactly like the old edit(): (audio_tensor, sr)
            # so your Save Audio node code still works
            return audio_out, 24000
        except Exception as e:
            logger.error(f"edit_from_tensor failed: {e}")
            raise

        # -------------------------------------------------------------------------
    # Voice cloning directly from tensor (ComfyUI Load Audio output)
    # -------------------------------------------------------------------------

    def clone_from_tensor(
        self,
        audio,
        sr: Optional[int],
        prompt_text: str,
        target_text: str,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, int]:
        """
        Clone voice when the input comes from a Load Audio node / tensor,
        instead of a file path.

        Expected usage from ComfyUI:
            audio_out, sr_out = tts.clone_from_tensor(
                audio_in,               # dict or tensor
                None,                   # sr ignored if dict
                prompt_text,
                target_text,
            )

        Args:
            audio:
                - Either a dict: {"waveform": tensor[B,C,T], "sample_rate": sr}
                - Or a torch.Tensor waveform
            sr: sample rate (ignored if 'audio' is dict with 'sample_rate')
            prompt_text: Text content that was spoken in the prompt audio
            target_text: Text to synthesize with cloned voice

        Returns:
            Tuple[torch.Tensor, int]: Generated audio tensor and sample rate (24000)
        """
        self.enable_disk_cache = use_cache
        try:
            # Accept dict from Load Audio node or bare tensor
            if isinstance(audio, dict):
                waveform = audio["waveform"]
                sr_local = int(audio["sample_rate"])
            else:
                waveform = audio
                if sr is None:
                    raise ValueError(
                        "clone_from_tensor: 'sr' must be provided when 'audio' is a tensor."
                    )
                sr_local = int(sr)

            logger.debug(
                f"Starting tensor voice cloning: sr={sr_local}, prompt_text_len={len(prompt_text)}, "
                f"target_text_len={len(target_text)}"
            )

            # Get clone context (from memory/disk or compute) based on waveform + prompt_text
            ctx = self._get_or_build_clone_context_from_tensor(
                waveform, sr_local, prompt_text
            )

            base_history_prefix: list[int] = ctx["base_history_prefix"]
            arole_toks: list[int] = ctx["arole_toks"]
            vq0206_codes_vocoder: torch.Tensor = ctx["vq0206_codes_vocoder"]
            speech_feat: torch.Tensor = ctx["speech_feat"]
            speech_embedding: torch.Tensor = ctx["speech_embedding"]

            # Encode only the target text tokens
            _prefix_tokens = self.tokenizer.encode("\n")
            target_token_encode = self.tokenizer.encode("\n" + target_text)
            target_tokens = target_token_encode[len(_prefix_tokens):]

            token_ids = (
                base_history_prefix
                + target_tokens
                + [3]  # end of human turn
                + [4]
                + arole_toks  # assistant\n
            )

            # Run LLM on GPU (lazy move + return to CPU)
            self.llm = safe_move_model(self.llm, "cuda:0")
            input_ids = torch.tensor([token_ids]).to(torch.long).to("cuda")

            output_ids = self.llm.generate(
                input_ids,
                max_length=8192,
                temperature=0.7,
                do_sample=True,
                logits_processor=LogitsProcessorList(
                    [RepetitionAwareLogitsProcessor()]
                ),
            )
            self.llm = safe_move_model(self.llm, "cpu")

            # Remove prompt part and final eos
            output_ids = output_ids[:, len(token_ids): -1]
            logger.debug("Tensor voice cloning generation completed")

            # Vocoder: move cached features to GPU
            vq0206_codes_vocoder_gpu = vq0206_codes_vocoder.to("cuda")
            speech_feat_gpu = speech_feat.to(torch.bfloat16).to("cuda")
            speech_embedding_gpu = speech_embedding.to(torch.bfloat16).to("cuda")

            torch.cuda.empty_cache()
            self._load_cosy_model()

            try:
                audio_out = self.cosy_model.token2wav_nonstream(
                    output_ids - 65536,
                    vq0206_codes_vocoder_gpu,
                    speech_feat_gpu,
                    speech_embedding_gpu,
                )
            finally:
                # Always unload CosyVoice
                self._unload_cosy_model()

            # Return like existing clone(): (audio_tensor, sr)
            return audio_out, 24000
        except Exception as e:
            logger.error(f"clone_from_tensor failed: {e}")
            raise

