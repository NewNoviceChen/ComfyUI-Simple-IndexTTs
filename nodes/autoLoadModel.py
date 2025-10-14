import os
import folder_paths
from ..server.infer_v2 import IndexTTS2


class AutoLoadModelNode:
    def __init__(self):
        pass

    CATEGORY = "ComfyUI-Simple-IndexTTS"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {

            },
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ("IndexTTsModel",)
    RETURN_NAMES = ("IndexTTsModel",)
    FUNCTION = "autoLoadModel"

    def autoLoadModel(self):
        model_dir = os.path.join(folder_paths.models_dir, "indextts")
        cfg_path = os.path.join(model_dir, "config.yaml")
        tts = IndexTTS2(cfg_path=cfg_path,
                        model_dir=model_dir,
                        use_cuda_kernel=False)
        return (tts,)
