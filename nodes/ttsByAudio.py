import os
import random

import torch
import torchaudio
import folder_paths

from ..server.infer_v2 import IndexTTS2


class TTsByAudioNode:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))

    CATEGORY = "ComfyUI-Simple-IndexTTS"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("IndexTTsModel", ),
                "audio": ("AUDIO",),
                "text": ("STRING", {"multiline": True}),
                "format": (["wav", "mp3", "flac"],),
            },
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ("AUDIO",)
    # RETURN_NAMES = ()
    FUNCTION = "ttsByAudio"

    def ttsByAudio(self, model,audio, text, format):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            self.prefix_append, self.output_dir)
        file = f"{filename}_{counter:05}_.{format}"
        output_path = os.path.join(full_output_folder, file)
        print(f">> {output_path}")
        waveform = audio["waveform"].squeeze(0)
        sample_rate = audio["sample_rate"]
        waveform = torchaudio.functional.resample(waveform, sample_rate, 22050).mean(dim=0, keepdim=True)

        model.infer(spk_audio_prompt=waveform, text=text,
                  output_path=output_path,
                  verbose=True)

        waveform, sample_rate = torchaudio.load(output_path)
        audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        return (audio,)
