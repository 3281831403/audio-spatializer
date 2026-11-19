# audio-spatializer

## 题面

给 VR 场景加空间音频后，声音方位全乱：距离衰减不做 0 距离钳制，除零产生 inf；多普勒只算声源速度，忽略听者速度；方位角用 atan 而不是 atan2，左右半区算反；声道增益直接按角度插值不归一，5.1 下总能量随角度波动 3 dB；ITD 用固定 0.6 ms，不随方位变化；HRTF 按方位角线性差选，仰角变化时选错；遮挡低通不设下限，远处直接变哑；仰角超过 ±90 度不钳制，声源翻到脑后；批量遇无 HRTF 的声源只跳过，输出缺声道；单位混用（米与厘米），距离衰减差 100 倍。spatial/panning.py 是核心，tests/test_panning.py 15个红。要修的是：① 距离为 0 时钳制到正数并保证增益有限；② 多普勒同时投影声源与听者速度；③ 用 atan2 计算方位角并保留左右符号；④ 各布局的声道增益总和归一；⑤ ITD 随方位角按球头模型变化；⑥ HRTF 按球面角距离选择；⑦ 遮挡低通截止不低于 200 Hz；⑧ 仰角钳制在 ±90 度；⑨ 批量中任一声源缺少 HRTF 就整批抛 `SpatialError`；⑩ 全程统一使用米和秒。

## 复现

在仓库根目录运行：

```console
$ python -m pytest tests -q
15 failed
```

初始状态必须恰好是 `15 failed`：没有 collection error、import error 或 syntax error，15 条测试全部因 `NotImplementedError` 失败。修复后目标为 `15 passed`。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

- 只使用 Python 标准库（`math`、`cmath`）与 pytest；离线运行、不使用随机数、不 `sleep`、不联网。
- 位置与距离统一用米，速度用米/秒，ITD 用秒，角度用度；断言容差为 `1e-6`。
- 坐标采用听者对齐的右手系：`+X` 向右、`+Y` 向上、`+Z` 向后（正前方为 `-Z`）。
- `tests/test_panning.py` 是验收规格，**不可修改**；修复只发生在 `spatial/panning.py`（`spatial/__init__.py` 仅做导出）。
