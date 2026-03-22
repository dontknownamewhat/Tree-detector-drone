# Tree-detector-drone
# Tree Detector 脱敏代码说明

本目录为无人机项目 tree detector 的脱敏重构版本，仅保留算法核心，不包含组织内部业务实现细节。

## 文件结构

- include/tree_detector.hpp: 公共接口和核心数据结构。
- include/yolov5_trt.h: YOLO TensorRT 封装声明。
- src/tree_detector.cpp: 核心业务流程实现。
- src/yolov5_trt.cpp: YOLO TensorRT 推理实现。
- CMakeLists.txt: 最小构建脚本。

## 对接建议

如果需要在目标环境运行，请在接入层提供以下实现：

- 线程安全结果缓冲实现（适配 `IThreadSafeData`）。
- 点云数据源与点云计算实现（适配 `IPointCloudBase`）。
- 业务告警上报实现（适配 `IAlgRemoteControl::onTreeAlert`）。
- 日志实现（适配 `ILogSink`）。
