import torchaudio

from server.infer_v2 import IndexTTS2

if __name__ == "__main__":
    prompt_wav = "../examples/test.wav"
    text = '欢迎大家来体验indextts2，并给予我们意见与反馈，谢谢大家。'
    waveform, sample_rate = torchaudio.load(prompt_wav)

    waveform = torchaudio.functional.resample(waveform, sample_rate, 22050).mean(dim=0, keepdim=True)
    tts = IndexTTS2(cfg_path="D:\\gitProject\\ComfyUI_windows_portable\\ComfyUI\\models\\indextts\\config.yaml",
                    model_dir="D:\\gitProject\\ComfyUI_windows_portable\\ComfyUI\\models\\indextts",
                    use_cuda_kernel=False)
    tts.infer(spk_audio_prompt=waveform, text=text,
                                      output_path="gen.flac", verbose=True)
