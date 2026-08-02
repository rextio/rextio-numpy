# rextio-numpy

<p align="center">
  <img src="./assets/readme/rextio-icon.png" width="96" alt="Rextio 图标">
</p>

<p align="center"><strong>为 Rextio 能够证明安全的代码提供有边界的 NumPy→Rust 降级。</strong></p>

<p align="center">
  <a href="README.md">English</a> · <a href="README.ko.md">한국어</a> · 简体中文 · <a href="README.zh-hant.md">繁體中文</a> · <a href="README.ja.md">日本語</a>
</p>

`rextio-numpy` 通过 rust-numpy/`ndarray`，把一组刻意限定、带类型的 NumPy 操作降级为 Rust。符合条件的调用原生执行。分析时未认领的表达式保留在 Rextio 的 Python fallback 上；原生路由一旦选定，若未满足精确运行时边界，则会拒绝调用，而不会静默 deopt。

> **公开 Alpha 0.1.3**（2026-07-27 发布）。需要 Python 3.11+、`rextio>=0.1.6,<0.2` 和插件 API 1.5。本版本不声称通用零拷贝、全面提速或支持 rank-2 矩阵乘法。

## 已验证内容

| 领域 | 原生范围 |
| --- | --- |
| 逐元素 | 同 dtype、rank 1–2 的 float64/float32/int64 数组上的 `+ - * /`、严格双参数 `numpy.add/subtract/multiply/divide` 和部分一元调用 |
| 比较 | 非链式 `== != < <= > >=`；常驻布尔掩码可供 `logical_not/and/or` 和严格三参数 `numpy.where` 使用 |
| 线性代数 | 仅限 1-D float64/int64 的 `numpy.dot(a, b)` 和 `a.dot(b)` |
| 归约 | 有边界的整个数组与字面量轴 `sum`、`mean`、`max`、`min`；精确矩阵见下文 |
| 融合 | 支持 dtype/rank 矩阵内由 2–8 个纯逐元素操作组成的链 |

形状错误、整数回绕、顺序、dtype 和已记录的异常文本由真实 Cargo 原生/fallback 测试覆盖。浮点归约和 dot 在运算顺序可能不同时采用容差等价。原生路径可能省略 NumPy `RuntimeWarning`；warning 一致性不在认证范围内。

## 工作原理

1. 类型注解标识有边界的数组 dtype 和 rank。
2. 插件只认领严格受支持的表达式，并使用 rust-numpy/`ndarray` 生成 Rust。
3. Rextio 用 Cargo 编译已接受的路径。未认领的 NumPy 代码仍走 Python fallback。

对 `F64Arr1`，精确的基础 `ndarray` 输入在原生帧中保持为只读 rust-numpy 视图，rank-1 float64 结果直接填入新的 NumPy 自有输出。其他受支持的 dtype/rank 路径仍把输入复制为 Rust 自有数组，并通过 `ToPyArray` 返回。这是分配结构改进，不是通用零拷贝或公开性能声明。

## 快速开始

```bash
python -m pip install "rextio-numpy==0.1.3" numpy
```

```toml
# rextio.toml
[rust]
build_tool = "cargo"

[plugins]
enabled = ["rextio-numpy"]
```

```python
import numpy as np
from rextio_numpy.types import F64Arr1

def dot(a: F64Arr1, b: F64Arr1) -> float:
    return np.dot(a, b)
```

```bash
rextio capabilities --format json
rextio build .
```

`F64Arr1` 在运行时只是 `numpy.ndarray` 的普通别名；注解用于指导静态分析，本身不是运行时验证。

## 精确支持范围

### 注解词汇

| 注解 | Rank | dtype |
| --- | ---: | --- |
| `F64Arr1`, `F64Arr2` | 1, 2 | float64 |
| `F32Arr1`, `F32Arr2` | 1, 2 | float32 |
| `I64Arr1`, `I64Arr2` | 1, 2 | int64 |

常驻布尔掩码没有公开注解，不能跨越 Python 参数或返回边界。

### 操作与约束

- 逐元素运算符支持 rank-1/rank-2 NumPy 广播、数组↔数组和数组↔匹配 Python 标量，包括零长度轴。整数真除法返回 float64。
- 严格 binary ufunc 形式只接受两个位置操作数。`out`、ufunc `where`、`dtype`、`casting`、额外参数和混合 dtype 均走 fallback。
- `numpy.where(condition, x, y)` 要求常驻比较/逻辑条件以及同 dtype 数值分支，或一个数组加匹配标量。至少一个分支必须是数组；关键字、仅条件、双标量和混合 dtype 形式均走 fallback。
- `dot`：仅 1-D float64/int64。float32 dot、所有 2-D dot、`numpy.matmul` 和 `@` 均走 fallback。
- 整个数组、无关键字归约：float64/int64 `sum`、float64 `mean`、int64 `max/min`；也包括受支持的等价 ndarray 方法。
- 严格一个字面量整数轴：rank 1–2 的 float64/int64 `sum`、float64 `mean`、int64 `max/min`。负轴会归一化。动态/tuple/`None`/越界轴及额外选项均走 fallback。
- 一元 `numpy.negative`、`numpy.absolute`/`numpy.abs`、`numpy.square` 支持 rank 1–2 的 float64/float32/int64，不接受可选参数或方法形式。
- 融合接受含 2–8 个数组名操作的纯二元运算树：float64/float32 使用 `+ - * /`，int64 使用 `+ - *`。超出范围的树保留普通逐操作处理。

## 边界与 fallback

- 运行时输入必须是精确的基础 `numpy.ndarray`。`numpy.matrix`、`numpy.memmap` 和自定义子类会在原生边界确定性抛出 `TypeError`；这不是自动 fallback。
- 支持精确 ndarray 视图、只读输入和正/负 stride。只有 `F64Arr1` 保留借用输入视图；其他路径物化自有副本。
- 数组结果是新的 NumPy 自有数组，具有 `OWNDATA`、`base is None` 和普通 resize 可观察行为。任何路径都不使用 `IntoPyArray`。
- float32 sum/mean/dot、int64 mean 和浮点 `max/min` 因尚未证明稳定的 NumPy 等价累加或 NaN/signed-zero 选择而保留 fallback。
- int64 `+ - *`、`sum`、`dot` 使用与 NumPy release build 一致的回绕。Python 整数标量路径限于 signed i64；越界值会在插件 helper 前抛出 `OverflowError`。
- 链式/identity/membership 比较、会修改的别名视图、rank 大于 2、reshape/view、不支持的方法形式及其他 NumPy API 均走 fallback。
- 带 `@numba.*` 装饰器的函数交给 Numba。

## 证据与非声明

[诚实 benchmark suite](benchmarks/README.md) 测量固定 `F64Arr1` 子集并同时报告收益和损失；仓库刻意不提交结果数字。[边界分配 PoC](benchmarks/boundary_allocation_poc/README.md) 仅供研究，[rank-2 matmul harness](benchmarks/matmul_wave2/README.md) 保持 **NO-GO / fallback** 产品决定。它们都不是通用提速证据。

## 开发

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python "rextio>=0.1.6,<0.2"
uv pip install --python .venv/bin/python --no-deps -e .
uv pip install --python .venv/bin/python pytest ruff mypy numpy hypothesis
.venv/bin/python -m pytest
```

发布历史见 [CHANGELOG.md](CHANGELOG.md)，复现细节见各 benchmark README。

## 许可证

MIT
