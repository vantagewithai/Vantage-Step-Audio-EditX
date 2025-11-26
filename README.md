# Step‑Audio‑EditX Multi‑Voice Cloner Node 🎙️

This project is a custom node implementation built on top of Step-Audio-EditX. It adapts and extends EditX capabilities to support **multi‑speaker**, **long‑format**, **voice cloning**, and **emotion/style/speed editing**, enabling you to feed in a script with multiple speakers, inline pauses, paralinguistic cues (like laughter, breathing), and get a concatenated audio output — all in one pass.

---

## Description

The original Step‑Audio‑EditX model enables single‑speaker voice cloning and emotion/style editing given a reference prompt audio + text.  

This node extends that capability, allowing you to:

- Provide multiple “speaker” reference voices at once.  
- Write a simple script with speaker tags, inline pauses, and optionally emotion/style/speed tags.  
- Generate a **single contiguous audio file** with all voices, pauses, and editing applied.  
- Handle paralinguistic markers (like `[Laughter]`, `[Breathing]`, etc.) — these are preserved and synthesis attempts to reflect them as natural speech or silence, depending on your engine’s capabilities.  

In short: you can build multi‑voice dialogues, audio stories, podcasts, or voice‑over sequences in one go.

---

## Features

- Multi‑speaker support (map each speaker to a reference audio + prompt).  
- Inline speaker switching via `[speakerX]` tags.  
- Inline pauses via `[pause]N]` syntax (pause of N milliseconds).  
- Emotion / style / speed tags (e.g. `[happy]`, `[serious]`, `[faster]`) for each line.  
- Paralinguistic tag support — e.g. `[Laughter]`, `[Breathing]`, `[Sigh]`, `[Dissatisfaction-hnn]`, etc. Those tags remain in the output text.  
- Automatic concatenation of generated audio segments into one final waveform.  
- Progress reporting (with progress bar).  
- Graceful handling of missing speaker‑tags (defaults to first speaker).  

---

## How It Works

1. Parse the input script line by line.  
2. Detect tags:
   - `[speakerX]` — which reference voice/prompt to use.  
   - Optional leading tags like emotion, style, speed (e.g. `[happy]`, `[whisper]`, `[slower]`).  
   - Paralinguistic tags (preserved).  
   - `[pause]` tags — interpreted as “generate silence for N ms.”  
3. For each speech line, call `clone_from_tensor(...)` (and optionally repeated editing for emotion / style / speed).  
4. For pause lines, generate a tensor of zeros of the requested duration.  
5. Collect all segments (speech or silence), concatenate them, and return a single audio output.  

---

## Usage

Example usage in a ComfyUI flow:

```text
[speaker1][happy]Hello there!  
[pause]500  
[speaker2][sad][whisper]I’m not sure about this…  
[speaker1][Laughter]That’s hilarious!  
```

- Provide reference audios & prompts for each speaker.  
- Feed this script to the node.  
- Get a single AUDIO output: concatenated waveform with cloned voices, pauses, and editing.  

---

## Installation / Integration
```bash
   cd custom_nodes
   git clone https://github.com/vantagewithai/Vantage-Step-Audio-EditX.git
   cd Vantage-Step-Audio-EditX
   pip install -r requirements.txt
   ```  
4. Launch ComfyUI — the node should appear under category **`Vantage/Step-Audio-EditX`**.  

## Download Models
[Download](https://huggingface.co/vantagewithai/Step-Fun-EditX-ComfyUI)

After downloading the models, copy them into ComfyUI/models, you should have the following structure:
```
ComfyUI/
├── models/
│   ├── Step-Audio-EditX/
│   ├──── CosyVoice-300M-25Hz/
│   │     ├─── campplus.onnx
│   │     ├─── cosyvoice.yaml
│   │     ├─── flow.pt
│   │     └─── hift.pt
│   ├──── dengcunqin/
│   ├──── └─── speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online/
│   │          ├─── am.mvn
│   │          ├─── config.yaml
│   │          ├─── configuration.json
│   │          ├─── model.pt
│   │          ├─── seg_dict
│   │          ├─── tokens.json
│   │          ├─── tokens.txt
│   │          └─── write_tokens_from_txt.py
│   ├── model.safetensors
│   └── speech_tokenizer_v1.onnx
```
---

## Script Syntax

| Tag Type | Syntax | Meaning |
|---------|--------|---------|
| Speaker switch | `[speakerX]` | Use speaker number X (1-based) |
| Pause / silence | `[pause]300` | Insert 300 ms of silence |
| Emotion | `[happy]`, `[sad]`, … | First valid emotion tag per line is applied |
| Style | `[whisper]`, `[serious]`, … | First valid style tag per line is applied |
| Speed modifier | `[faster]`, `[slower]`, … | First valid speed tag per line is applied |
| Paralinguistic cue | `[Laughter]`, `[Breathing]`, `[Sigh]`, … | Preserved in the text — not stripped. May be used for downstream effects. |

Tags **must** come before the actual text of the line (after `[speakerX]`).  

Example:

```
[speaker2][happy][whisper][slower]I am fine!  
[speaker1][Laughter]That was funny!  
[pause]500  
```

---

## Limitations & Notes

- The quality of voice cloning / emotion/style editing depends on the underlying `Step‑Audio‑EditX` engine and your reference audio & prompt.  
- Paralinguistic tags are preserved in the text passed to the engine — if the engine doesn’t support them, they may just render as silence or be ignored.  
- If sample rates of speakers vary, the node currently assumes uniform sample rate in the concatenation step.  
- Long scripts may consume significant VRAM / memory — monitor usage accordingly.  
- The node does **not** perform grammar or punctuation correction — script should be well‑formatted.  

---

## License & Credits

**License: MIT**

This project builds upon the original **Step‑Audio‑EditX** repository (see [https://github.com/stepfun-ai/Step-Audio-EditX](https://github.com/stepfun-ai/Step-Audio-EditX)).  

Please refer to the original repository for the base license.

