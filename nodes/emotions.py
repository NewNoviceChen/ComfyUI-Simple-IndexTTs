import torchaudio

from comfy_api.latest import io


class EmotionFromAudioNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EmotionFromAudioNode",
            display_name="情感-音频",
            category="ComfyUI-Simple-IndexTTS",
            description="音频生成音色",
            inputs=[
                io.String.Input("timbre_name"),
                io.Audio.Input("spk_audio"),
                io.Audio.Input("emo_audio",optional=True),
                io.Float.Input("emo_waveform_weight", min=0.0, max=1.0, default=0.6, step=0.01),
            ],
            outputs=[
                io.Custom("emotion").Output("emotion"),
            ]
        )

    @classmethod
    def execute(cls, timbre_name, spk_audio, emo_audio=None, emo_waveform_weight=None):
        spk_waveform = spk_audio["waveform"].squeeze(0)
        spk_sample_rate = spk_audio["sample_rate"]
        spk_waveform = torchaudio.functional.resample(spk_waveform, spk_sample_rate, 22050).mean(dim=0, keepdim=True)
        if emo_audio is None:
            emo_audio = spk_audio
        emo_waveform = emo_audio["waveform"].squeeze(0)
        emo_sample_rate = emo_audio["sample_rate"]
        emo_waveform = torchaudio.functional.resample(emo_waveform, emo_sample_rate, 22050).mean(dim=0, keepdim=True)
        emotion = {"type": "audio",
                   "timbre_name": timbre_name,
                   "spk_waveform": spk_waveform,
                   "emo_waveform": emo_waveform,
                   "emo_waveform_weight": emo_waveform_weight}
        return io.NodeOutput(emotion)


class EmotionFromTensorNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EmotionFromTensorNode",
            display_name="情感-张量",
            category="ComfyUI-Simple-IndexTTS",
            description="音频生成音色",
            inputs=[
                io.String.Input("timbre_name"),
                io.Audio.Input("spk_audio"),
                io.Boolean.Input("use_random"),
                io.Float.Input("happy", min=0.0, max=1.0, default=0, step=0.01),
                io.Float.Input("angry", min=0.0, max=1.0, default=0, step=0.01),
                io.Float.Input("sad", min=0.0, max=1.0, default=0, step=0.01),
                io.Float.Input("afraid", min=0.0, max=1.0, default=0, step=0.01),
                io.Float.Input("disgusted", min=0.0, max=1.0, default=0, step=0.01),
                io.Float.Input("melancholic", min=0.0, max=1.0, default=0, step=0.01),
                io.Float.Input("surprised", min=0.0, max=1.0, default=0, step=0.01),
                io.Float.Input("calm", min=0.0, max=1.0, default=0, step=0.01),
            ],
            outputs=[
                io.Custom("emotion").Output("emotion"),
            ]
        )

    @classmethod
    def execute(cls, timbre_name, spk_audio, use_random, happy, angry, sad, afraid, disgusted, melancholic, surprised,
                calm):
        spk_waveform = spk_audio["waveform"].squeeze(0)
        spk_sample_rate = spk_audio["sample_rate"]
        spk_waveform = torchaudio.functional.resample(spk_waveform, spk_sample_rate, 22050).mean(dim=0, keepdim=True)

        emotion = {"type": "tensor",
                   "timbre_name": timbre_name,
                   "spk_waveform": spk_waveform,
                   "use_random": use_random,
                   "emo_tensor": [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]}
        return io.NodeOutput(emotion)


class EmotionFromTextNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EmotionFromTextNode",
            display_name="情感-文本",
            category="ComfyUI-Simple-IndexTTS",
            description="音频生成音色",
            inputs=[
                io.String.Input("timbre_name"),
                io.Audio.Input("spk_audio"),
                io.String.Input("emo_text"),
                io.Float.Input("emo_text_weight", min=0.0, max=1.0, default=0.6, step=0.01),
            ],
            outputs=[
                io.Custom("emotion").Output("emotion"),
            ]
        )

    @classmethod
    def execute(cls, timbre_name, spk_audio, emo_text, emo_text_weight):
        spk_waveform = spk_audio["waveform"].squeeze(0)
        spk_sample_rate = spk_audio["sample_rate"]
        spk_waveform = torchaudio.functional.resample(spk_waveform, spk_sample_rate, 22050).mean(dim=0, keepdim=True)

        emotion = {"type": "text",
                   "timbre_name": timbre_name,
                   "spk_waveform": spk_waveform,
                   "emo_text": emo_text,
                   "emo_text_weight": emo_text_weight}
        return io.NodeOutput(emotion)


class MergeEmotionNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MergeEmotionNode",
            display_name="合并情感音色",
            category="ComfyUI-Simple-IndexTTS",
            description="合并情感音色",
            inputs=[
                io.Custom("emotion").Input("emotion1"),
                io.Custom("emotion").Input("emotion2", optional=True),
                io.Custom("emotion").Input("emotion3", optional=True),
                io.Custom("emotion").Input("emotion4", optional=True),
                io.Custom("emotion").Input("emotion5", optional=True),
                io.Custom("emotion").Input("emotion6", optional=True),
            ],
            outputs=[
                io.Custom("emotion_list").Output("emotion_list"),
            ]
        )

    @classmethod
    def execute(cls, emotion1, emotion2=None, emotion3=None, emotion4=None, emotion5=None, emotion6=None):
        emotion_list = []
        if emotion1 is not None:
            if emotion1["timbre_name"] is None:
                emotion1["timbre_name"] = "1"
            emotion_list.append(emotion1)
        if emotion2 is not None:
            if emotion2["timbre_name"] is None:
                emotion2["timbre_name"] = "2"
            emotion_list.append(emotion2)
        if emotion3 is not None:
            if emotion3["timbre_name"] is None:
                emotion3["timbre_name"] = "3"
            emotion_list.append(emotion3)
        if emotion4 is not None:
            if emotion4["timbre_name"] is None:
                emotion4["timbre_name"] = "4"
            emotion_list.append(emotion4)
        if emotion5 is not None:
            if emotion5["timbre_name"] is None:
                emotion5["timbre_name"] = "5"
            emotion_list.append(emotion5)
        if emotion6 is not None:
            if emotion6["timbre_name"] is None:
                emotion6["timbre_name"] = "6"
            emotion_list.append(emotion6)
        return io.NodeOutput(emotion_list)
