# __init__.py
import os
import sys

PLUGIN_DIR = os.path.dirname(__file__)
CORE_DIR = os.path.join(PLUGIN_DIR, "coreeditx")
print(f"{CORE_DIR}")
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)
    
from .nodes import EditXModelLoader,EditXSingleVoiceCloner,EditXSingleVoiceEditor,EditXSingleVoiceEditorFromPath,EditXSingleVoiceClonerFromPath,LoadSpeakers,EditXMultiVoiceCloner

NODE_CLASS_MAPPINGS = {
    "EditXModelLoader": EditXModelLoader,
    "EditXSingleVoiceCloner": EditXSingleVoiceCloner,
    "EditXSingleVoiceEditor": EditXSingleVoiceEditor,
    "EditXSingleVoiceEditorFromPath": EditXSingleVoiceEditorFromPath,
    "EditXSingleVoiceClonerFromPath": EditXSingleVoiceClonerFromPath,
    "EditXMultiVoiceCloner": EditXMultiVoiceCloner,
    "LoadSpeakers": LoadSpeakers,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EditXModelLoader": "Step-Audio-EditX Model Loader",
    "EditXSingleVoiceCloner": "Step-Audio-EditX Single Voice Cloner",
    "EditXSingleVoiceEditor": "Step-Audio-EditX Single Voice Editor",
    "EditXSingleVoiceEditorFromPath": "Step-Audio-EditX Single Voice Editor From Path",
    "EditXSingleVoiceClonerFromPath": "Step-Audio-EditX Single Voice Cloner From Path",
    "EditXMultiVoiceCloner": "Step-Audio-EditX Multi Voice Cloner",
    "LoadSpeakers": "Step-Audio-EditX Load Speakers",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

