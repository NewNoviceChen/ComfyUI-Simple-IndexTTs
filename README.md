# ComfyUI-Simple-IndexTTs
Bilibili的IndexTTs2的ComfyUI版本

## 使用
自动加载模型节点会自动在comfyui下的 models文件夹下生成一个indextts文件夹，自动从huggingface下载模型

音频组的生成可以指定说话人，格式为 名称:文案 ， 句间可以指定间隔，格式为 pause:时间

### 单个音频生成

![单个音频.png](examples/%E5%8D%95%E4%B8%AA%E9%9F%B3%E9%A2%91.png)

### 音频组生成

![音频组.png](examples/%E9%9F%B3%E9%A2%91%E7%BB%84.png)

## 引用
参考[index-tts](https://github.com/index-tts/index-tts)