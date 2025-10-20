import os
import folder_paths
from ..server.infer_v2 import IndexTTS2
from comfy_api.latest import io


class AutoLoadModelNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AutoLoadModelNode",
            display_name="AutoLoadModel",
            category="ComfyUI-Simple-IndexTTS",
            description="AutoLoadModel",
            inputs=[

            ],
            outputs=[
                io.Custom("IndexTTsModel").Output("IndexTTsModel"),
            ]
        )

    @classmethod
    def execute(cls) -> io.NodeOutput:
        model_dir = os.path.join(folder_paths.models_dir, "indextts")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        cfg_path = os.path.join(model_dir, "config.yaml")
        tts = IndexTTS2(cfg_path=cfg_path,
                        model_dir=model_dir,
                        use_cuda_kernel=False)
        return io.NodeOutput(tts)
