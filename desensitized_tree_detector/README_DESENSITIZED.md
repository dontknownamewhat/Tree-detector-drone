# Tree Detector 脱敏代码说明

本目录为 tree detector 的脱敏重构版本，仅保留算法核心，不包含组织内部业务实现细节。

## 已做的脱敏处理

- 移除私有模块引用：
  - 原工程中的 common/*、alg/*、mission/*、db/*、utils/* 等私有依赖已去除。
- 移除私有业务流程：
  - 设备ID生成、数据库写入、照片路径落盘、内部业务备注字段等已移除。
- 保留核心能力：
  - YOLO TensorRT 推理封装。
  - 树目标检测主流程。
  - 点云3D距离估算。
  - 树冠凸出检测与净空计算。
- 抽象外部交互：
  - 使用通用接口 `IThreadSafeData` / `IPointCloudBase` / `IAlgRemoteControl` / `ILogSink`。

## 文件结构

- include/tree_detector.hpp: 脱敏后的公共接口和核心数据结构。
- include/yolov5_trt.h: YOLO TensorRT 封装声明。
- src/tree_detector.cpp: 脱敏后的核心业务流程实现。
- src/yolov5_trt.cpp: YOLO TensorRT 推理实现。
- CMakeLists.txt: 脱敏后的最小构建脚本。

## 对接建议

如果需要在目标环境运行，请在接入层提供以下实现：

- 线程安全结果缓冲实现（适配 `IThreadSafeData`）。
- 点云数据源与点云计算实现（适配 `IPointCloudBase`）。
- 业务告警上报实现（适配 `IAlgRemoteControl::onTreeAlert`）。
- 日志实现（适配 `ILogSink`）。

本版本用于代码共享、评审与安全交付，不包含任何可识别的内部业务资产路径和数据库资产信息。
