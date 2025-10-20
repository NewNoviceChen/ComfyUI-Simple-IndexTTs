import math
import os
import random
import re

import folder_paths
import torch
import torchaudio
from comfy_api.latest import ComfyExtension, io, ui


class TTsNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TTsNode",
            display_name="音频生成",
            category="ComfyUI-Simple-IndexTTS",
            description="音频生成",
            is_output_node=True,
            inputs=[
                io.Custom("IndexTTsModel").Input("IndexTTsModel"),
                io.String.Input("text", multiline=True),
                io.Custom("emotion").Input("emotion", optional=True),
            ],
            outputs=[
                io.Audio.Output()
            ]
        )

    @classmethod
    def execute(cls, IndexTTsModel, spk_audio=None, text=None, emotion=None) -> io.NodeOutput:
        output_dir = folder_paths.get_temp_directory()
        prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            prefix_append, output_dir)
        file = f"{filename}_{counter:05}_.flac"
        output_path = os.path.join(full_output_folder, file)
        if emotion is not None:
            spk_waveform = emotion["spk_waveform"]
            if emotion["type"] == "audio":
                emo_waveform = emotion["emo_waveform"]
                emo_waveform_weight = emotion["emo_waveform_weight"]
                IndexTTsModel.infer(spk_audio_prompt=spk_waveform,
                                    text=text,
                                    output_path=output_path,
                                    emo_audio_prompt=emo_waveform,
                                    emo_alpha=emo_waveform_weight,
                                    verbose=True)
            elif emotion["type"] == "tensor":
                use_random = emotion["use_random"]
                emo_tensor = emotion["emo_tensor"]
                IndexTTsModel.infer(spk_audio_prompt=spk_waveform,
                                    text=text,
                                    output_path=output_path,
                                    emo_vector=emo_tensor,
                                    use_random=use_random,
                                    verbose=True)
            elif emotion["type"] == "text":
                emo_text = emotion["emo_text"]
                emo_text_weight = emotion["emo_text_weight"]
                IndexTTsModel.infer(spk_audio_prompt=spk_waveform,
                                    text=text,
                                    output_path=output_path,
                                    use_emo_text=True,
                                    emo_alpha=emo_text_weight,
                                    emo_text=emo_text,
                                    verbose=True)
        waveform, sample_rate = torchaudio.load(output_path)
        audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        return io.NodeOutput(audio, ui=ui.PreviewAudio(audio, cls=cls))


class BatchTTsNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BatchTTsNode",
            display_name="批量音频生成",
            category="ComfyUI-Simple-IndexTTS",
            description="批量音频生成",
            is_output_node=True,
            inputs=[
                io.Custom("IndexTTsModel").Input("IndexTTsModel"),
                io.Custom("emotion_list").Input("emotion_list", optional=True),
                io.String.Input("text", multiline=True),
            ],
            outputs=[
                io.Audio.Output()
            ]
        )

    @classmethod
    def execute(cls, IndexTTsModel, emotion_list, text) -> io.NodeOutput:
        pattern = r'(.+?):\s*(.+)'
        matches = re.findall(pattern, text)
        waveforms = []
        sample_rate = 22050
        for name, content in matches:
            if name == "pause":
                duration = float(content)
                silence = torch.zeros(1, int(sample_rate * duration))
                waveforms.append(silence)
            for emotion in emotion_list:
                if emotion["timbre_name"] == name:
                    output_dir = folder_paths.get_temp_directory()
                    prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
                    full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
                        prefix_append, output_dir)
                    file = f"{filename}_{counter:05}_.flac"
                    output_path = os.path.join(full_output_folder, file)
                    spk_waveform = emotion["spk_waveform"]
                    if emotion["type"] == "audio":
                        emo_waveform = emotion["emo_waveform"]
                        emo_waveform_weight = emotion["emo_waveform_weight"]
                        IndexTTsModel.infer(spk_audio_prompt=spk_waveform,
                                            text=content,
                                            output_path=output_path,
                                            emo_audio_prompt=emo_waveform,
                                            emo_alpha=emo_waveform_weight,
                                            verbose=True)
                    elif emotion["type"] == "tensor":
                        use_random = emotion["use_random"]
                        emo_tensor = emotion["emo_tensor"]
                        IndexTTsModel.infer(spk_audio_prompt=spk_waveform,
                                            text=content,
                                            output_path=output_path,
                                            emo_vector=emo_tensor,
                                            use_random=use_random,
                                            verbose=True)
                    elif emotion["type"] == "text":
                        emo_text = emotion["emo_text"]
                        emo_text_weight = emotion["emo_text_weight"]
                        IndexTTsModel.infer(spk_audio_prompt=spk_waveform,
                                            text=content,
                                            output_path=output_path,
                                            use_emo_text=True,
                                            emo_alpha=emo_text_weight,
                                            emo_text=emo_text,
                                            verbose=True)
                    waveform, sample_rate = torchaudio.load(output_path)
                    waveforms.append(waveform)
        result = torch.cat(waveforms, dim=1)
        audio = {"waveform": result.unsqueeze(0), "sample_rate": sample_rate}
        return io.NodeOutput(audio, ui=ui.PreviewAudio(audio, cls=cls))
