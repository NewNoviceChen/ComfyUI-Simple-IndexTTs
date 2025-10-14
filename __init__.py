from .nodes.ttsByAudio import TTsByAudioNode
from .nodes.autoLoadModel import AutoLoadModelNode

NODE_CLASS_MAPPINGS = {
    "ttsByAudio": TTsByAudioNode,
    "autoLoadModelNode": AutoLoadModelNode,
}
