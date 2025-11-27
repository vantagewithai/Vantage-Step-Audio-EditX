import io
import threading
import time
import os
import logging

import numpy as np
import torch
import torchaudio
import onnxruntime
import whisper
import tempfile
from funasr_detach import AutoModel
from .utils import resample_audio, energy_norm_fn, trim_silence
import soundfile as sf

# Root directory = folder where model_loader.py lives
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(ROOT_DIR, "configs")
# ----------------------------------------------------

# Configure logging
logger = logging.getLogger(__name__)

class StepAudioTokenizer:
    def __init__(
        self,
        encoder_path,
        device: str = "cpu",   # "cpu" or "cuda:0", "cuda:1", ...        
        offload: bool = False,
    ):
        """
        Initialize StepAudioTokenizer (local only)

        Args:
            encoder_path: Path to encoder directory containing speech_tokenizer_v1.onnx
            funasr_model_id: Local path name under configs/ for FunASR model files
        """
        self.funasr_model = None
        self.kms = None
        self.model_path = encoder_path
        self.offload = offload
        self.ort_session = None
        
        # track where we want to run things
        self.device = device          # logical device for FunASR
        self._ort_providers = ["CPUExecutionProvider"]  # start on CPU
        
        self.chunk_size = [0, 4, 5]
        self.encoder_chunk_look_back = 4
        self.decoder_chunk_look_back = 1

        self.vq02_sessions = {}
        self.vq02_lock = threading.Lock()
        self.vq06_lock = threading.Lock()
    
    def load_tokenizer(self):
        logger.debug("Loading audio tokenizer")
        funasr_model_id="dengcunqin/speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online"
        config_path = CONFIG_DIR
        funasr_model_path = os.path.join(self.model_path, funasr_model_id)
        
        try:
            self.funasr_model = AutoModel(
                repo_path=config_path,
                model=funasr_model_path,
                model_hub="local",
                model_revision="main",
                disable_log=True,
            )
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load FunASR model locally: {e}")
        
        kms_path = os.path.join(self.funasr_model.repo_path, "linguistic_tokenizer.npy")
        self.cosy_tokenizer_path = os.path.join(self.model_path, "speech_tokenizer_v1.onnx")
        
        if not os.path.exists(self.cosy_tokenizer_path):
            raise FileNotFoundError(f"Cosy tokenizer file not found: {self.cosy_tokenizer_path}")
        if not os.path.exists(kms_path):
            raise FileNotFoundError(f"KMS file not found: {kms_path}")
        
        #providers = ["CUDAExecutionProvider"]
        #session_option = onnxruntime.SessionOptions()
        #session_option.graph_optimization_level = (
        #    onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        #)
        #session_option.intra_op_num_threads = 1
        
        #self.ort_session = onnxruntime.InferenceSession(
        #    cosy_tokenizer_path, sess_options=session_option, providers=providers
        #)
        
        self.kms = torch.tensor(np.load(kms_path))
        
        # build initial ORT session on CPU by default
        #self._create_ort_session(cosy_tokenizer_path)
        self.ort_session = None
        self.to_cuda(device_id=0)
        logger.debug("Loaded audio tokenizer")
        
    def _create_ort_session(self):
        session_option = onnxruntime.SessionOptions()
        session_option.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        session_option.intra_op_num_threads = 1

        self.ort_session = onnxruntime.InferenceSession(
            self.cosy_tokenizer_path,
            sess_options=session_option,
            providers=self._ort_providers,
        )
    
    def _clear_ort_session(self):
        if self.ort_session is not None:
            self.ort_session = None
            # optional: force cleanup
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def __call__(self, audio, sr):
        _, vq02, vq06 = self.wav2token(audio, sr, False)
        text = self.merge_vq0206_to_token_str(vq02, vq06)
        return text

    def preprocess_wav(self, audio, sample_rate, enable_trim=True, energy_norm=True):
        audio = resample_audio(audio, sample_rate, 16000)
        if energy_norm:
            audio = energy_norm_fn(audio)

        if enable_trim:
            audio = audio.cpu().numpy().squeeze(0)
            audio = trim_silence(audio, 16000)
            audio = torch.from_numpy(audio).unsqueeze(0)
        return audio

    def wav2token(self, audio, sample_rate, enable_trim=True, energy_norm=True):
        audio = self.preprocess_wav(audio, sample_rate, enable_trim, energy_norm)
        vq02_ori = self.get_vq02_code(audio)
        vq02 = [int(x) + 65536 for x in vq02_ori]
        vq06_ori = self.get_vq06_code(audio)
        vq06 = [int(x) + 65536 + 1024 for x in vq06_ori]

        chunk = 1
        chunk_nums = min(len(vq06) // (3 * chunk), len(vq02) // (2 * chunk))
        speech_tokens = []
        for idx in range(chunk_nums):
            speech_tokens += vq02[idx * chunk * 2 : (idx + 1) * chunk * 2]
            speech_tokens += vq06[idx * chunk * 3 : (idx + 1) * chunk * 3]
        return speech_tokens, vq02_ori, vq06_ori
    
    def _get_funasr_device_arg(self):
        """
        Returns the value to pass as `device` into self.funasr_model.infer_encoder`.
        """
        if self.device == "cpu":
            # FunASR usually accepts 'cpu' or -1 for CPU; adjust if your version needs something else
            return "cpu"
        if self.device.startswith("cuda:"):
            # e.g. 'cuda:0' -> 0
            return int(self.device.split(":")[1])
        # fallback
        return self.device
        
    def get_vq02_code(self, audio, session_id=None, is_final=True):
        generated = True
        try:
            logger.debug("Using BytesIO")
            _tmp_wav = io.BytesIO()
            torchaudio.save(_tmp_wav, audio, 16000, format="wav")
        except Exception as e:
            generated = False
        if(not generated):
            logger.debug("Using Temp File")
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmpfile:
                sf.write(tmpfile.name, audio.cpu().numpy().T, 16000)
                with open(tmpfile.name, "rb") as f:
                    _tmp_wav = io.BytesIO(f.read())
        
        _tmp_wav.seek(0)
    
        with self.vq02_lock:
            cache = {}
            if session_id in self.vq02_sessions:
                cache = self.vq02_sessions[session_id].get("cache", {})

            res, new_cache = self.funasr_model.infer_encoder(
                input=[_tmp_wav],
                chunk_size=self.chunk_size,
                encoder_chunk_look_back=self.encoder_chunk_look_back,
                decoder_chunk_look_back=self.decoder_chunk_look_back,
                device=self._get_funasr_device_arg(),  # <--- changed
                is_final=is_final,
                cache=cache,
            )
            c_list = []
            for res_ in res:
                feat = res_["enc_out"]
                if len(feat) > 0:
                    c_list = self.dump_label([feat], self.kms)[0]

            if is_final:
                if session_id in self.vq02_sessions:
                    self.vq02_sessions.pop(session_id)
            else:
                if isinstance(session_id, str) and len(session_id) > 0:
                    self.vq02_sessions[session_id] = {"cache": new_cache, "update_time": time.time()}

            return c_list

    def get_vq06_code(self, audio):
        def split_audio(audio, chunk_duration=480000):
            chunks = []
            start = 0
            while start < len(audio):
                end = min(start + chunk_duration, len(audio))
                chunk = audio[start:end]
                if len(chunk) >= 480:
                    chunks.append(chunk)
                start = end
            return chunks

        with self.vq06_lock:
            audio = audio.squeeze(0)
            chunk_audios = split_audio(audio, 30 * 16000)
            speech_tokens = []
            for chunk in chunk_audios:
                duration = round(chunk.shape[0] / 16000, 2)
                feat = whisper.log_mel_spectrogram(chunk, n_mels=128).unsqueeze(0)
                feat_len = np.array([feat.shape[2]], dtype=np.int32)
                chunk_token = (
                    self.ort_session.run(
                        None,
                        {
                            self.ort_session.get_inputs()[0].name: feat.detach().cpu().numpy(),
                            self.ort_session.get_inputs()[1].name: feat_len,
                        },
                    )[0]
                    .flatten()
                    .tolist()
                )
                assert abs(len(chunk_token) - duration * 25) <= 2
                speech_tokens += chunk_token

            return speech_tokens

    def kmean_cluster(self, samples, means):
        dists = torch.cdist(samples, means)
        indices = dists.argmin(dim=1).cpu().numpy()
        return indices.tolist()

    def dump_label(self, samples, mean):
        dims = samples[0].shape[-1]
        x_lens = [x.shape[1] for x in samples]
        total_len = sum(x_lens)
        x_sel = torch.FloatTensor(1, total_len, dims)
        start_len = 0
        for sample in samples:
            sample_len = sample.shape[1]
            x_sel[:, start_len:start_len + sample_len] = sample
            start_len += sample_len
        dense_x = x_sel.squeeze(0)
        indices = self.kmean_cluster(dense_x, mean)
        return [indices[:x_lens[0]]]

    def merge_vq0206_to_token_str(self, vq02, vq06):
        _vq06 = [1024 + x for x in vq06]
        result = []
        i = j = 0
        while i < len(vq02) - 1 and j < len(_vq06) - 2:
            sublist = vq02[i:i+2] + _vq06[j:j+3]
            result.extend(sublist)
            i += 2
            j += 3
        return "".join([f"<audio_{x}>" for x in result])
    
    def unload_tokenizer(self):
        # 1. Kill ORT session
        self._clear_ort_session()

        # 2. Drop FunASR and other big objects
        self.funasr_model = None
        self.kms = None

        # 3. Clear caches
        self.vq02_sessions = {}
        # any other caches you use

        # 4. GC + CUDA cache cleanup
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def to_cuda(self, device_id: int = 0):
        """
        Move tokenizer computation to CUDA:
        - FunASR uses GPU (device_id)
        - ONNX Runtime session uses CUDAExecutionProvider
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, cannot move tokenizer to CUDA.")

        self.device = f"cuda:{device_id}"
        self._ort_providers = ["CUDAExecutionProvider"]
        
        self._clear_ort_session()
        self._create_ort_session()
        logger.debug("Audio tokenizer moved to cuda")
