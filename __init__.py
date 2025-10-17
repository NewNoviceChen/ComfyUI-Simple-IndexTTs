from comfy_api.latest import ComfyExtension, io
from .nodes.autoLoadModel import AutoLoadModelNode
from .nodes.ttsByAudio import TTsNode, BatchTTsNode
from .nodes.emotions import EmotionFromAudioNode, EmotionFromTensorNode, EmotionFromTextNode, MergeEmotionNode


class MyExtension(ComfyExtension):
    # 必须声明为异步
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            AutoLoadModelNode, TTsNode, BatchTTsNode, EmotionFromAudioNode, EmotionFromTensorNode,
            EmotionFromTextNode, MergeEmotionNode
            # 在这里添加更多节点
        ]


# 可以声明为异步或不是，两者都可以工作
async def comfy_entrypoint() -> MyExtension:
    return MyExtension()
