# 2025-11-19-20:58-edit 提交解析

## 1. 提交目的
- 在 YOLOv12 主干和检测头中引入 Difference Mask Mixed Attention（DMMA）与 ECA 通道注意力，以提升对小目标和复杂背景的建模能力。
- 新增 `yolov12-dmma.yaml` 模型配置，同时将 DMMA 模块注册到 `ultralytics.nn` 中，使其可被解析与训练。

## 2. 文件级变更
### ultralytics/cfg/models/v12/yolov12-dmma.yaml
- 新建配置文件，定义五档 scale（n/s/m/l/x）及 Backbone/Head 拓扑。
- Backbone 在 P4、P5 位置堆叠两段 `C2fDMMA`，Head 在上采样融合后继续使用 `C2fDMMA`，最终仍输出 P3/P4/P5 三尺度检测层。

### ultralytics/nn/modules/__init__.py
- 将 `C2fDMMA` 导入并加入 `__all__`，保证外部引用及解析流程能够找到该模块。

### ultralytics/nn/modules/block.py
- 在 `__all__` 中注册 `C2fDMMA`，并实现对应类：
  - 构造阶段把输入通道拆分为两路，引入 `DMMALayer` 组成的序列，支持窗口大小、head 数、是否交替平移等参数；
  - 在初始化时校验隐藏通道数能被 head 数整除，避免运行期 shape 错误；
  - 提供 `forward` 与 `forward_split` 两种路径，沿袭 C2f 系列的接口约定。

### ultralytics/nn/modules/transformer.py
- 扩展 Transformer 模块集合，加入 DMMA 全栈实现：
  - 工具函数 `dmma_window_partition/reverse` 与 `_to_2tuple` 负责窗口化及形状处理；
  - `DifferenceMaskAttention` 构建包含差分 mask 的自注意力，并复用相对位置偏置；
  - `DMMAChannelAttention`、`MinusSigmoid`、`DMMAMlp`、`DropPath` 等组件提供通道增强与正则化；
  - `DMMALayer` 将上述模块打包成带 Shift-Window 策略的注意力块，为 `C2fDMMA` 提供主体逻辑。

### ultralytics/nn/tasks.py
- 在模块注册、解析白名单以及自动插入 repeat 的集合中新增 `C2fDMMA`，确保 YAML 配置能够被 `parse_model` 正确构建。

## 3. 影响与建议
- `C2fDMMA` 对 hidden channel 与 head 数的可整除性较为敏感，新增配置中通过 `e` 和 head 数手动控制，请在自定义 scale 时保持同样约束。
- DMMA 层包含窗口划分与 shift 操作，对输入尺寸有整除要求；当前实现会自动 `pad`，但可能带来边缘像素的轻微影响。
- 由于引入大量注意力计算，推理显著更加耗时，如需在资源受限设备部署，可考虑提供降级配置（例如替换部分 `C2fDMMA` 为常规 `C2f`）。
