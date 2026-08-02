# rextio-numpy

<p align="center">
  <img src="./assets/readme/rextio-icon.png" width="96" alt="Rextio 圖示">
</p>

<p align="center"><strong>為 Rextio 能夠證明安全的程式碼提供有邊界的 NumPy→Rust lowering。</strong></p>

<p align="center">
  <a href="README.md">English</a> · <a href="README.ko.md">한국어</a> · <a href="README.zh-hans.md">简体中文</a> · 繁體中文 · <a href="README.ja.md">日本語</a>
</p>

`rextio-numpy` 透過 rust-numpy/`ndarray`，把一組刻意限定、帶型別的 NumPy 操作 lowering 為 Rust。符合條件的呼叫以原生方式執行。分析時未 claim 的運算式保留在 Rextio 的 Python fallback；原生路由一旦選定，若不符合精確執行時邊界，呼叫會被拒絕，而不會靜默 deopt。

> **公開 Alpha 0.1.3**（2026-07-27 發布）。需要 Python 3.11+、`rextio>=0.1.6,<0.2` 與外掛 API 1.5。本版本不宣稱通用零拷貝、全面加速或支援 rank-2 矩陣乘法。

## 已驗證內容

| 領域 | 原生範圍 |
| --- | --- |
| 逐元素 | 同 dtype、rank 1–2 的 float64/float32/int64 陣列上的 `+ - * /`、嚴格雙參數 `numpy.add/subtract/multiply/divide` 與部分一元呼叫 |
| 比較 | 非鏈式 `== != < <= > >=`；常駐布林遮罩可供 `logical_not/and/or` 與嚴格三參數 `numpy.where` 使用 |
| 線性代數 | 僅限 1-D float64/int64 的 `numpy.dot(a, b)` 與 `a.dot(b)` |
| 歸約 | 有邊界的整體陣列及字面值軸 `sum`、`mean`、`max`、`min`；精確矩陣見下文 |
| 融合 | 支援 dtype/rank 矩陣內由 2–8 個純逐元素操作組成的鏈 |

shape 錯誤、整數 wraparound、順序、dtype 與已記錄的例外文字由真實 Cargo 原生/fallback 測試涵蓋。浮點歸約和 dot 在運算順序可能不同時採用容差等價。原生路徑可能省略 NumPy `RuntimeWarning`；warning 一致性不在認證範圍內。

## 運作方式

1. 型別 annotation 識別有邊界的陣列 dtype 與 rank。
2. 外掛只 claim 嚴格支援的運算式，並使用 rust-numpy/`ndarray` 產生 Rust。
3. Rextio 以 Cargo 編譯已接受路徑。未 claim 的 NumPy 程式碼仍走 Python fallback。

對 `F64Arr1`，精確的基礎 `ndarray` 輸入在原生 frame 中保持為唯讀 rust-numpy view，rank-1 float64 結果直接填入新的 NumPy 自有輸出。其他受支援 dtype/rank 路徑仍把輸入複製為 Rust 自有陣列，並透過 `ToPyArray` 回傳。這是配置結構改善，不是通用零拷貝或公開效能聲明。

## 快速開始

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

`F64Arr1` 在執行時只是 `numpy.ndarray` 的普通 alias；annotation 用於引導靜態分析，本身不是執行時驗證。

## 精確支援範圍

### Annotation vocabulary

| Annotation | Rank | dtype |
| --- | ---: | --- |
| `F64Arr1`, `F64Arr2` | 1, 2 | float64 |
| `F32Arr1`, `F32Arr2` | 1, 2 | float32 |
| `I64Arr1`, `I64Arr2` | 1, 2 | int64 |

常駐布林遮罩沒有公開 annotation，不能跨越 Python parameter 或 return 邊界。

### 操作與限制

- 逐元素運算子支援 rank-1/rank-2 NumPy broadcasting、陣列↔陣列與陣列↔相符 Python scalar，包括零長度軸。整數 true division 回傳 float64。
- 嚴格 binary ufunc 形式只接受兩個 positional operand。`out`、ufunc `where`、`dtype`、`casting`、額外參數與混合 dtype 均走 fallback。
- `numpy.where(condition, x, y)` 要求常駐比較/邏輯條件與同 dtype 數值分支，或一個陣列加相符 scalar。至少一個分支必須是陣列；keyword、condition-only、雙 scalar 與混合 dtype 均走 fallback。
- `dot`：僅 1-D float64/int64。float32 dot、所有 2-D dot、`numpy.matmul` 與 `@` 均走 fallback。
- 整體陣列、無 keyword 歸約：float64/int64 `sum`、float64 `mean`、int64 `max/min`；也包含受支援的等價 ndarray method。
- 嚴格一個字面值整數軸：rank 1–2 的 float64/int64 `sum`、float64 `mean`、int64 `max/min`。負軸會正規化。動態/tuple/`None`/越界軸與額外 option 均走 fallback。
- 一元 `numpy.negative`、`numpy.absolute`/`numpy.abs`、`numpy.square` 支援 rank 1–2 的 float64/float32/int64，不接受選用參數或 method 形式。
- fusion 接受含 2–8 個陣列名稱操作的純 binary-op tree：float64/float32 使用 `+ - * /`，int64 使用 `+ - *`。超出範圍的 tree 保留普通逐操作處理。

## 邊界與 fallback

- 執行時輸入必須是精確的基礎 `numpy.ndarray`。`numpy.matrix`、`numpy.memmap` 與自訂 subclass 會在原生邊界確定性拋出 `TypeError`；這不是自動 fallback。
- 支援精確 ndarray view、唯讀輸入與正/負 stride。只有 `F64Arr1` 保留借用輸入 view；其他路徑實體化自有 copy。
- 陣列結果是新的 NumPy 自有陣列，具有 `OWNDATA`、`base is None` 與一般 resize 可觀察行為。任何路徑都不使用 `IntoPyArray`。
- float32 sum/mean/dot、int64 mean 與浮點 `max/min` 因尚未證明穩定的 NumPy 等價累加或 NaN/signed-zero 選擇而保留 fallback。
- int64 `+ - *`、`sum`、`dot` 使用與 NumPy release build 一致的 wraparound。Python 整數 scalar 路徑限於 signed i64；越界值會在外掛 helper 前拋出 `OverflowError`。
- 鏈式/identity/membership 比較、會修改的 alias view、rank 大於 2、reshape/view、不支援的 method 形式與其他 NumPy API 均走 fallback。
- 帶 `@numba.*` decorator 的函式交給 Numba。

## 證據與非聲明

[誠實 benchmark suite](benchmarks/README.md) 測量固定 `F64Arr1` 子集並同時報告收益與損失；儲存庫刻意不 commit 結果數字。[邊界配置 PoC](benchmarks/boundary_allocation_poc/README.md) 僅供研究，[rank-2 matmul harness](benchmarks/matmul_wave2/README.md) 保持 **NO-GO / fallback** 產品決定。它們都不是通用加速證據。

## 開發

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python "rextio>=0.1.6,<0.2"
uv pip install --python .venv/bin/python --no-deps -e .
uv pip install --python .venv/bin/python pytest ruff mypy numpy hypothesis
.venv/bin/python -m pytest
```

發布歷史見 [CHANGELOG.md](CHANGELOG.md)，重現細節見各 benchmark README。

## 授權條款

MIT
