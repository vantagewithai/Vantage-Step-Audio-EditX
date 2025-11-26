import os
import torch
import logging
import folder_paths
import re
from comfy.utils import ProgressBar  # <-- ComfyUI progress bar

try:
    from tqdm import tqdm            # <-- Optional CLI progress bar
except ImportError:
    tqdm = None
    
# Reuse your existing modules
from .coreeditx.tokenizer import StepAudioTokenizer          # 
from .coreeditx.tts import StepAudioTTS                      # 

logger = logging.getLogger(__name__)

# simple in-process cache so the loader node doesn't reload every time
_EDITX_MODEL_CACHE = {}

import re

def clean_tagged_text(text: str):
    """
    Removes [ ... ] tag blocks but ensures that removing them
    does not merge adjacent words.
    
    Returns:
        cleaned_text (str)
        changed (bool)
    """
    original = text

    # Replace tags with a space placeholder
    cleaned = re.sub(r"\[[^\]]*\]", " ", text)

    # Collapse multiple spaces to single space
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned, (cleaned != original)
    
def _map_torch_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(dtype_str, torch.bfloat16)


class EditXModelLoader:
    """
    ComfyUI node: loads all EditX models once and returns them as a dictionary.

    Output type: "EDITX_MODELS" – a plain Python dict containing:
        {
            "encoder": StepAudioTokenizer,
            "tts_engine": StepAudioTTS,
            "model_path": str,
        }
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "quantization": (["none", "int4", "int8", "awq-4bit"], {
                    "default": "none"
                }),

                # Torch dtype (only used when quantization != int4/int8 or when allowed)
                "torch_dtype": (["bfloat16", "float16", "float32"], {
                    "default": "bfloat16"
                }),
                "offload": ("BOOLEAN", {
                    "default": True
                }),
                "offload_funasr": ("BOOLEAN", {
                    "default": False
                }),
            }
        }

    RETURN_TYPES = ("EDITX_MODELS",)
    RETURN_NAMES = ("models",)
    FUNCTION = "load"
    CATEGORY = "Vantage/Step-Audio-EditX"

    def load(
        self,
        quantization: str,
        torch_dtype: str,
        offload: bool,
        offload_funasr: bool
    ):
        logger.info("[EditXModelLoader] Loading models...")
        editx_path = os.path.join(folder_paths.models_dir, "Step-Audio-EditX")

        # 1) Load the audio tokenizer (FunASR + ONNX speech tokenizer) locally
        encoder = StepAudioTokenizer(
            encoder_path=editx_path,
            offload=offload_funasr,
        )

        # 2) Load the TTS engine (LLM + CosyVoice vocoder) locally using your UnifiedModelLoader internally
        quant_config = None if quantization == "none" else quantization
        torch_dtype_obj = _map_torch_dtype(torch_dtype)

        tts_engine = StepAudioTTS(
            model_path=editx_path,
            audio_tokenizer=encoder,
            quantization_config=quant_config,
            torch_dtype=torch_dtype_obj,
        )

        # Package everything into a dict that downstream nodes will consume
        models_dict = {
            "encoder": encoder,
            "tts_engine": tts_engine,
            "model_path": editx_path,
        }

        return (models_dict,)


class EditXSingleVoiceCloner:
    """
    ComfyUI node: Voice cloning using loaded EditX models.

    Uses StepAudioTTS.clone(prompt_wav_path, prompt_text, target_text) 

    Input:
        - models: dict from EditXModelLoader
        - prompt_audio_path: path to reference wav file
        - prompt_text: text spoken in the reference audio
        - target_text: new text to synthesize with cloned voice

    Output:
        - AUDIO: (waveform_tensor, sample_rate)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "models": ("EDITX_MODELS",),
                "audio_path": ("AUDIO",),
                "prompt_text": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "target_text": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "enable_disk_cache": ("BOOLEAN", {
                    "default": True
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "clone"
    CATEGORY = "Vantage/Step-Audio-EditX"

    def clone(
        self,
        models,
        audio_path: str,
        prompt_text: str,
        target_text: str,
        enable_disk_cache: bool = True,
    ):
        if not models or "tts_engine" not in models:
            raise ValueError(
                "[EditXVoiceCloner] 'models' input is missing or invalid. "
                "Make sure to connect the output of EditXModelLoader."
            )

        tts_engine: StepAudioTTS = models["tts_engine"]

        if not audio_path:
            raise ValueError("[EditXVoiceCloner] prompt_audio_path is empty.")
        if not prompt_text.strip():
            raise ValueError("[EditXVoiceCloner] prompt_text cannot be empty.")
        if not target_text.strip():
            raise ValueError("[EditXVoiceCloner] target_text cannot be empty.")

        logger.info("[EditXVoiceCloner] Starting voice cloning...")
        audio_tensor, sr = tts_engine.clone_from_tensor(
            audio_path,
            None,
            prompt_text=prompt_text,
            target_text=target_text,
            use_cache=enable_disk_cache,
        )
        logger.info("[EditXVoiceCloner] Voice cloning finished.")

        # In ComfyUI, the underlying object type is arbitrary; we use a tuple (tensor, sr)
        #return ((audio_tensor, sr),)
        
        # Ensure tensor
        if not torch.is_tensor(audio_tensor):
            audio_tensor = torch.tensor(audio_tensor)

        # Use float32
        audio_tensor = audio_tensor.float()

        # Flatten whatever shape we got and reinterpret as [B=1, C=1, T]
        audio_tensor = audio_tensor.flatten().view(1, 1, -1)

        audio = {
            "waveform": audio_tensor,
            "sample_rate": sr,
        }

        return (audio,)
        
class EditXSingleVoiceClonerFromPath:
    """
    ComfyUI node: Voice cloning using loaded EditX models.

    Uses StepAudioTTS.clone(prompt_wav_path, prompt_text, target_text) 

    Input:
        - models: dict from EditXModelLoader
        - prompt_audio_path: path to reference wav file
        - prompt_text: text spoken in the reference audio
        - target_text: new text to synthesize with cloned voice

    Output:
        - AUDIO: (waveform_tensor, sample_rate)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "models": ("EDITX_MODELS",),
                "prompt_audio_path": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
                "prompt_text": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "target_text": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "enable_disk_cache": ("BOOLEAN", {
                    "default": True
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "clone"
    CATEGORY = "Vantage/Step-Audio-EditX"

    def clone(
        self,
        models,
        prompt_audio_path: str,
        prompt_text: str,
        target_text: str,
        enable_disk_cache: bool = True
    ):
        if not models or "tts_engine" not in models:
            raise ValueError(
                "[EditXVoiceCloner] 'models' input is missing or invalid. "
                "Make sure to connect the output of EditXModelLoader."
            )

        tts_engine: StepAudioTTS = models["tts_engine"]

        if not prompt_audio_path:
            raise ValueError("[EditXVoiceCloner] prompt_audio_path is empty.")
        if not prompt_text.strip():
            raise ValueError("[EditXVoiceCloner] prompt_text cannot be empty.")
        if not target_text.strip():
            raise ValueError("[EditXVoiceCloner] target_text cannot be empty.")

        logger.info("[EditXVoiceCloner] Starting voice cloning...")
        audio_tensor, sr = tts_engine.clone(
            prompt_wav_path=prompt_audio_path,
            prompt_text=prompt_text,
            target_text=target_text,
            use_cache=enable_disk_cache,
        )
        logger.info("[EditXVoiceCloner] Voice cloning finished.")

        # In ComfyUI, the underlying object type is arbitrary; we use a tuple (tensor, sr)
        #return ((audio_tensor, sr),)
        
        # Ensure tensor
        if not torch.is_tensor(audio_tensor):
            audio_tensor = torch.tensor(audio_tensor)

        # Use float32
        audio_tensor = audio_tensor.float()

        # Flatten whatever shape we got and reinterpret as [B=1, C=1, T]
        audio_tensor = audio_tensor.flatten().view(1, 1, -1)

        audio = {
            "waveform": audio_tensor,
            "sample_rate": sr,
        }

        return (audio,)


class EditXSingleVoiceEditorFromPath:
    """
    ComfyUI node: Audio editing using loaded EditX models.

    Uses StepAudioTTS.edit(input_audio_path, audio_text, edit_type, edit_info, text) 

    Input:
        - models: dict from EditXModelLoader
        - input_audio_path: path to input wav file
        - audio_text: text corresponding to the input audio (can be empty for denoise/vad)
        - edit_type: one of the supported types ("emotion", "style", "vad", "denoise", "paralinguistic", "speed")
        - edit_info: subtype, e.g. "happy" / "sad" / "remove" etc.
        - target_text: used only for "paralinguistic" edits (new text)

    Output:
        - AUDIO: (waveform_tensor, sample_rate)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "models": ("EDITX_MODELS",),
                "input_audio_path": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
                "audio_text": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "edit_iterations": ("INT", {"default": 2, "min": 1, "max": 5, "step": 1}),
                "edit_type": ([
                    "emotion",
                    "style",
                    "vad",
                    "denoise",
                    "paralinguistic",
                    "speed",
                ], {
                    "default": "emotion"
                }),
                "edit_info": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "happy / sad / remove / faster / slower / ..."
                }),
                "target_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "edit"
    CATEGORY = "Vantage/Step-Audio-EditX"

    def edit(
        self,
        models,
        input_audio_path: str,
        edit_iterations: int,
        audio_text: str,
        edit_type: str,
        edit_info: str,
        target_text: str,
    ):
        if not models or "tts_engine" not in models:
            raise ValueError(
                "[EditXVoiceEditor] 'models' input is missing or invalid. "
                "Make sure to connect the output of EditXModelLoader."
            )

        tts_engine: StepAudioTTS = models["tts_engine"]

        if not input_audio_path:
            raise ValueError("[EditXVoiceEditor] input_audio_path is empty.")

        # Mirror the guard in StepAudioEditX.generate_edit where text is required
        # except for 'denoise' and 'vad'. 
        if edit_type not in ["denoise", "vad"] and not audio_text.strip():
            raise ValueError(
                "[EditXVoiceEditor] audio_text cannot be empty "
                "for this edit_type (only denoise/vad can omit text)."
            )

        # For non-paralinguistic edits, we follow your generate_edit logic:
        # generated_text = text_to_use (i.e., the same as audio_text/history).
        # Here we don't manage multi-iteration history inside the node,
        # so we simply use audio_text as the effective text_to_use.
        if edit_type == "paralinguistic":
            effective_target_text = target_text
        else:
            effective_target_text = audio_text
        
        logger.info(
            f"[EditXVoiceEditor] Starting edit: type={edit_type}, info='{edit_info}'"
        )
        
        audio_tensor, sr = torchaudio.load(input_audio_path)
        # Ensure tensor
        if not torch.is_tensor(audio_tensor):
            audio_tensor = torch.tensor(audio_tensor)

        # Use float32
        audio_tensor = audio_tensor.float()

        # Flatten whatever shape we got and reinterpret as [B=1, C=1, T]
        audio_tensor = audio_tensor.flatten().view(1, 1, -1)

        audio = {
            "waveform": audio_tensor,
            "sample_rate": sr,
        }
            
        for i in range(edit_iterations):
            audio_tensor, sr = tts_engine.edit_from_tensor(
                audio,
                None,
                audio_text=audio_text,
                edit_type=edit_type,
                edit_info=edit_info if edit_info else None,
                text=effective_target_text,
                use_cache=False,
            )
            # Ensure tensor
            if not torch.is_tensor(audio_tensor):
                audio_tensor = torch.tensor(audio_tensor)

            # Use float32
            audio_tensor = audio_tensor.float()

            # Flatten whatever shape we got and reinterpret as [B=1, C=1, T]
            audio_tensor = audio_tensor.flatten().view(1, 1, -1)

            audio = {
                "waveform": audio_tensor,
                "sample_rate": sr,
            }
        
        logger.info("[EditXVoiceEditor] Audio editing finished.")
        

        return (audio,)

class EditXSingleVoiceEditor:
    """
    ComfyUI node: Audio editing using loaded EditX models.

    Uses StepAudioTTS.edit(input_audio_path, audio_text, edit_type, edit_info, text) 

    Input:
        - models: dict from EditXModelLoader
        - input_audio_path: path to input wav file
        - audio_text: text corresponding to the input audio (can be empty for denoise/vad)
        - edit_type: one of the supported types ("emotion", "style", "vad", "denoise", "paralinguistic", "speed")
        - edit_info: subtype, e.g. "happy" / "sad" / "remove" etc.
        - target_text: used only for "paralinguistic" edits (new text)

    Output:
        - AUDIO: (waveform_tensor, sample_rate)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "models": ("EDITX_MODELS",),
                "input_audio": ("AUDIO",),
                "audio_text": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "edit_iterations": ("INT", {"default": 2, "min": 1, "max": 5, "step": 1}),
                "edit_type": ([
                    "emotion",
                    "style",
                    "vad",
                    "denoise",
                    "paralinguistic",
                    "speed",
                ], {
                    "default": "emotion"
                }),
                "edit_info": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "happy / sad / remove / faster / slower / ..."
                }),
                "target_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "edit"
    CATEGORY = "Vantage/Step-Audio-EditX"

    def edit(
        self,
        models,
        input_audio,
        edit_iterations: int,
        audio_text: str,
        edit_type: str,
        edit_info: str,
        target_text: str,
    ):
        if not models or "tts_engine" not in models:
            raise ValueError(
                "[EditXVoiceEditor] 'models' input is missing or invalid. "
                "Make sure to connect the output of EditXModelLoader."
            )

        tts_engine: StepAudioTTS = models["tts_engine"]

        if not input_audio:
            raise ValueError("[EditXVoiceEditor] input_audio_path is empty.")

        # Mirror the guard in StepAudioEditX.generate_edit where text is required
        # except for 'denoise' and 'vad'. 
        if edit_type not in ["denoise", "vad"] and not audio_text.strip():
            raise ValueError(
                "[EditXVoiceEditor] audio_text cannot be empty "
                "for this edit_type (only denoise/vad can omit text)."
            )

        # For non-paralinguistic edits, we follow your generate_edit logic:
        # generated_text = text_to_use (i.e., the same as audio_text/history).
        # Here we don't manage multi-iteration history inside the node,
        # so we simply use audio_text as the effective text_to_use.
        if edit_type == "paralinguistic":
            effective_target_text = target_text
        else:
            effective_target_text = audio_text
        
        logger.info(
            f"[EditXVoiceEditor] Starting edit: type={edit_type}, info='{edit_info}'"
        )
        audio = input_audio
        for i in range(edit_iterations):
            audio_tensor, sr = tts_engine.edit_from_tensor(
                audio,
                None,
                audio_text=audio_text,
                edit_type=edit_type,
                edit_info=edit_info if edit_info else None,
                text=effective_target_text,
                use_cache=False,
            )
            # Ensure tensor
            if not torch.is_tensor(audio_tensor):
                audio_tensor = torch.tensor(audio_tensor)

            # Use float32
            audio_tensor = audio_tensor.float()

            # Flatten whatever shape we got and reinterpret as [B=1, C=1, T]
            audio_tensor = audio_tensor.flatten().view(1, 1, -1)

            audio = {
                "waveform": audio_tensor,
                "sample_rate": sr,
            }
            
        logger.info("[EditXVoiceEditor] Audio editing finished.")
        

        return (audio,)

class LoadSpeakers:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_1": ("AUDIO", ),
                "prompt_1": ("STRING", {
                    "multiline": True,
                    "default": ""
                } ),
            },
            "optional": {
                "audio_2": ("AUDIO", ),
                "prompt_2": ("STRING", {
                    "multiline": True,
                    "default": ""
                } ),
                "audio_3": ("AUDIO", ),
                "prompt_3": ("STRING", {
                    "multiline": True,
                    "default": ""
                } ),
                "audio_4": ("AUDIO", ),
                "prompt_4": ("STRING", {
                    "multiline": True,
                    "default": ""
                } ),
                "audio_5": ("AUDIO", ),
                "prompt_5": ("STRING", {
                    "multiline": True,
                    "default": ""
                } ),
                "audio_6": ("AUDIO", ),
                "prompt_6": ("STRING", {
                    "multiline": True,
                    "default": ""
                } ),
            },
        }

    RETURN_TYPES = ("EDITX_SPEAKERS",)
    RETURN_NAMES = ("speakers",)
    FUNCTION = "load"

    CATEGORY = "Vantage/Step-Audio-EditX"

    def load(self, **kwargs):
        # Aggregate all available audio/prompt pairs into a single payload
        speakers = {
            "audios": [],
            "prompts": [],
        }
        for idx in range(1, 6):
            audio_key = f"audio_{idx}"
            prompt_key = f"prompt_{idx}"
        
            audio = kwargs.get(audio_key, None)
            prompt = kwargs.get(prompt_key, None)

            # Skip slots that are not wired / empty
            if audio is None or prompt is None:
                continue

            speakers["audios"].append(audio)
            speakers["prompts"].append(prompt)

        return (speakers,)
        
class EditXMultiVoiceCloner:
    """
    ComfyUI node: Voice cloning using loaded EditX models.

    Uses StepAudioTTS.clone_from_tensor(prompt_audio_tensor, ..., prompt_text, target_text)

    Supports:
        [speaker1]Hello there...
        [speaker2][happy][serious][faster]Reply...
        [pause]300      -> insert 300 ms silence
    """

    SPEAKER_TAG_RE = re.compile(r'^\s*\[speaker(\d+)\]\s*(.*)$', re.IGNORECASE)
    PAUSE_TAG_RE   = re.compile(r'^\s*\[pause\]\s*(\d+)\s*$', re.IGNORECASE)
    GENERIC_TAG_RE = re.compile(r'^\s*\[([^\]]+)\]\s*(.*)$')

    # Allowed tag values
    EMOTION_VALUES = {
        'happy', 'angry', 'sad', 'humour', 'confusion', 'disgusted',
        'empathy', 'embarrass', 'fear', 'surprised', 'excited',
        'depressed', 'coldness', 'admiration', 'remove'
    }

    STYLE_VALUES = {
        'serious', 'arrogant', 'child', 'older', 'girl', 'pure',
        'sister', 'sweet', 'ethereal', 'whisper', 'gentle', 'recite',
        'generous', 'act_coy', 'warm', 'shy', 'comfort', 'authority',
        'chat', 'radio', 'soulful', 'story', 'vivid', 'program',
        'news', 'advertising', 'roar', 'murmur', 'shout', 'deeply',
        'loudly', 'remove', 'exaggerated'
    }

    SPEED_VALUES = {
        'faster', 'slower', 'more faster', 'more slower'
    }
    
    PARALINGUISTIC_TAGS = {
        'breathing',
        'laughter',
        'suprise-oh',
        'confirmation-en',
        'uhm',
        'suprise-ah',
        'suprise-wa',
        'sigh',
        'question-ei',
        'dissatisfaction-hnn',
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "models": ("EDITX_MODELS",),
                "speakers": ("EDITX_SPEAKERS",),
                "prompt_text": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "edit_iterations": ("INT", {"default": 2, "min": 1, "max": 5, "step": 1}),
                "enable_disk_cache": ("BOOLEAN", {
                    "default": True
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "clone"
    CATEGORY = "Vantage/Step-Audio-EditX"
    
    def combine_audio_results(self, processed_results):
        if not processed_results:
            return None

        sample_rate = processed_results[0]["sample_rate"]
        waveforms = [item["waveform"] for item in processed_results]
        combined_waveform = torch.cat(waveforms, dim=2)

        return {
            "waveform": combined_waveform,
            "sample_rate": sample_rate,
        }

    def _parse_tags_after_speaker(self, text: str):
        """
        Given the string after [speakerX] (or the whole line if no speaker),
        consume leading [tag] blocks and extract:
            emotion, style, speed, remaining_text

        Rules:
        - Any leading [xxx] before the first non-blank non-[ char is treated as a tag.
        - Known tags:
            * emotion: from EMOTION_VALUES
            * style:   from STYLE_VALUES
            * speed:   from SPEED_VALUES
        - Only the first tag of each category is kept; later tags of same category are ignored.
        - Paralinguistic tags (Breathing/Laughter/etc) are KEPT as part of text
          at their position.
        - Other unknown tags at the start are consumed and ignored.
        - Parsing stops once the remaining string's first non-space char is not '['.
        - The remaining text is returned as-is and may contain [xxxx] sequences inside.
        """
        emotion = ""
        style = ""
        speed = ""

        rest = text
        prefix_text = ""  # where we keep paralinguistic tags in order

        while True:
            # If first non-space char is not '[', we are at real text → stop.
            stripped = rest.lstrip()
            if not stripped.startswith('['):
                break

            m = self.GENERIC_TAG_RE.match(rest)
            if not m:
                # malformed tag (no closing ']') → treat everything as text
                break

            raw_tag = m.group(1)
            rest_after = m.group(2)
            tag = raw_tag.strip()
            tag_lower = tag.lower()

            # Recognized emotion tag
            if tag_lower in self.EMOTION_VALUES:
                if not emotion:
                    emotion = tag_lower
                # even if duplicate, we still consume it and continue
                rest = rest_after
                continue

            # Recognized style tag
            if tag_lower in self.STYLE_VALUES:
                if not style:
                    style = tag_lower
                rest = rest_after
                continue

            # Recognized speed tag
            if tag_lower in self.SPEED_VALUES:
                if not speed:
                    speed = tag_lower
                rest = rest_after
                continue

            # Paralinguistic tag (keep it as part of the visible text)
            if tag_lower in self.PARALINGUISTIC_TAGS:
                if prefix_text and not prefix_text.endswith(" "):
                    prefix_text += " "
                # Keep original casing in output
                prefix_text += f"[{raw_tag}]"
                rest = rest_after
                continue

            # Unknown tag at the start: consume it but don't store anywhere.
            rest = rest_after
            continue

        # Combine kept paralinguistic prefix + remaining text
        rest_stripped = rest.strip()
        if prefix_text:
            if rest_stripped:
                final_text = f"{prefix_text} {rest_stripped}"
            else:
                final_text = prefix_text
        else:
            final_text = rest_stripped

        return emotion, style, speed, final_text

    def clone(
        self,
        models,
        speakers,
        prompt_text: str,
        edit_iterations: int,
        enable_disk_cache: bool,
    ):
        if not models or "tts_engine" not in models:
            raise ValueError(
                "[EditXVoiceCloner] 'models' input is missing or invalid. "
                "Make sure to connect the output of EditXModelLoader."
            )

        tts_engine: StepAudioTTS = models["tts_engine"]

        if not speakers:
            raise ValueError("[EditXVoiceCloner] speakers are empty.")
        if not prompt_text.strip():
            raise ValueError("[EditXVoiceCloner] prompt_text cannot be empty.")

        logger.info("[EditXVoiceCloner] Starting voice cloning...")
        
        audios = speakers.get("audios", [])
        prompts = speakers.get("prompts", [])

        # --- Build list of items: speech or pause ---
        items = []
        for raw_line in prompt_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            # Check for [pause]xxx lines first
            m_pause = self.PAUSE_TAG_RE.match(line)
            if m_pause:
                duration_ms = int(m_pause.group(1))
                items.append({
                    "type": "pause",
                    "duration_ms": duration_ms,
                })
                continue

            # Normal [speakerX] ... or text (default speaker 1)
            m_speaker = self.SPEAKER_TAG_RE.match(line)
            if m_speaker:
                speaker_idx = int(m_speaker.group(1)) - 1  # 1-based to 0-based
                rest = m_speaker.group(2).strip()
            else:
                speaker_idx = 0
                rest = line

            if speaker_idx < 0 or speaker_idx >= len(audios):
                speaker_idx = 0

            # Parse optional [emotion][style][speed] tags at start of `rest`
            emotion, style, speed, final_text = self._parse_tags_after_speaker(rest)

            items.append({
                "type": "speech",
                "audio": audios[speaker_idx],
                "prompt": prompts[speaker_idx],
                "text": final_text,
                "emotion": emotion,
                "style": style,
                "speed": speed,
            })

        total_items = len(items)
        if total_items == 0:
            raise ValueError("[EditXVoiceCloner] No valid lines found in prompt_text.")

        # --- ComfyUI progress bar ---
        pbar = ProgressBar(total_items)

        processed_results = []

        # Prepare iterable for tqdm if available
        iterable = items
        if tqdm is not None:
            iterable = tqdm(
                iterable,
                total=total_items,
                desc="Cloning",
                leave=True
            )

        # --- Main processing loop with progress ---
        for idx, item in enumerate(iterable, start=1):
            if item["type"] == "pause":
                duration_ms = item["duration_ms"]
                status = f"Adding Pause"
                iterable.set_description(status)  
                # Determine sample rate for silence
                if processed_results:
                    sr = processed_results[0]["sample_rate"]
                else:
                    # Fallback: try engine's sample_rate or use 24000
                    sr = getattr(tts_engine, "sample_rate", 24000)

                num_samples = int(sr * duration_ms / 1000.0)
                if num_samples <= 0:
                    num_samples = 1  # avoid empty tensor edge case

                silence = torch.zeros(1, 1, num_samples, dtype=torch.float32)
                audio_out = {
                    "waveform": silence,
                    "sample_rate": sr,
                }
                processed_results.append(audio_out)
            else:  # speech
                audio = item["audio"]
                prompt = item["prompt"]
                text = item["text"]
                emotion = item["emotion"]
                style = item["style"]
                speed = item["speed"]
                
                result, changed = clean_tagged_text(text)
                    
                status = f"Cloning"
                iterable.set_description(status)
                
                logger.debug(
                    f"[EditXVoiceCloner] Cloning line {idx}/{total_items} "
                    f"(emotion='{emotion}', style='{style}', speed='{speed}')."
                    f"(changed='{changed}', result='{result}')."
                    f"(text='{text}')."
                )
                if((emotion == "" or emotion == None) and (style == "" or style == None) and (speed == "" or speed == None)):
                    changed = False
                    audio_tensor, sr = tts_engine.clone_from_tensor(
                        audio,
                        None,
                        prompt_text=prompt,
                        target_text=text,
                        use_cache=enable_disk_cache,
                    )
                else:
                    audio_tensor, sr = tts_engine.clone_from_tensor(
                        audio,
                        None,
                        prompt_text=prompt,
                        target_text=result,
                        use_cache=enable_disk_cache,
                    )
                    
                if not torch.is_tensor(audio_tensor):
                    audio_tensor = torch.tensor(audio_tensor)

                audio_tensor = audio_tensor.float()
                audio_tensor = audio_tensor.flatten().view(1, 1, -1)

                audio_out = {
                    "waveform": audio_tensor,
                    "sample_rate": sr,
                }
                
                if(not emotion == "" and not emotion == None):
                    for i in range(edit_iterations):
                        status = f"Emotion: {emotion} {i+1}/{edit_iterations}"
                        iterable.set_description(status)
                        audio_tensor, sr = tts_engine.edit_from_tensor(
                            audio_out,
                            None,
                            audio_text=result,
                            edit_type="emotion",
                            edit_info=emotion,
                            text=result,
                            use_cache=False,
                        )
                        if not torch.is_tensor(audio_tensor):
                            audio_tensor = torch.tensor(audio_tensor)

                        audio_tensor = audio_tensor.float()
                        audio_tensor = audio_tensor.flatten().view(1, 1, -1)

                        audio_out = {
                            "waveform": audio_tensor,
                            "sample_rate": sr,
                        }
                
                if(not style == "" and not style == None):
                    for i in range(edit_iterations):
                        status = f"Style: {style} {i+1}/{edit_iterations}"
                        iterable.set_description(status)
                    
                        audio_tensor, sr = tts_engine.edit_from_tensor(
                            audio_out,
                            None,
                            audio_text=result,
                            edit_type="style",
                            edit_info=style,
                            text=result,
                            use_cache=False,
                        )
                        if not torch.is_tensor(audio_tensor):
                            audio_tensor = torch.tensor(audio_tensor)

                        audio_tensor = audio_tensor.float()
                        audio_tensor = audio_tensor.flatten().view(1, 1, -1)

                        audio_out = {
                            "waveform": audio_tensor,
                            "sample_rate": sr,
                        }
                
                if(not speed == "" and not speed == None):
                    for i in range(edit_iterations):
                        status = f"Speed: {speed} {i+1}/{edit_iterations}"
                        iterable.set_description(status)
                    
                        audio_tensor, sr = tts_engine.edit_from_tensor(
                            audio_out,
                            None,
                            audio_text=result,
                            edit_type="speed",
                            edit_info=speed,
                            text=result,
                            use_cache=False,
                        )
                        if not torch.is_tensor(audio_tensor):
                            audio_tensor = torch.tensor(audio_tensor)

                        audio_tensor = audio_tensor.float()
                        audio_tensor = audio_tensor.flatten().view(1, 1, -1)

                        audio_out = {
                            "waveform": audio_tensor,
                            "sample_rate": sr,
                        }
                
                if(changed):
                    for i in range(edit_iterations):
                        status = f"Paralinguistic: {speed} {i+1}/{edit_iterations}"
                        iterable.set_description(status)
                    
                        audio_tensor, sr = tts_engine.edit_from_tensor(
                            audio_out,
                            None,
                            audio_text=result,
                            edit_type="paralinguistic",
                            text=text,
                            use_cache=False,
                        )
                        if not torch.is_tensor(audio_tensor):
                            audio_tensor = torch.tensor(audio_tensor)

                        audio_tensor = audio_tensor.float()
                        audio_tensor = audio_tensor.flatten().view(1, 1, -1)

                        audio_out = {
                            "waveform": audio_tensor,
                            "sample_rate": sr,
                        }

                processed_results.append(audio_out)

            # Update ComfyUI progress bar for every item (speech or pause)
            pbar.update(1)

        logger.info("[EditXVoiceCloner] All items processed, combining audio segments.")
        return (self.combine_audio_results(processed_results),)

