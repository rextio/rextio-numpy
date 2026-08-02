# rextio-numpy

<p align="center">
  <img src="./assets/readme/rextio-icon.png" width="96" alt="Rextio アイコン">
</p>

<p align="center"><strong>Rextio が安全性を証明できるコードのための、範囲を限定した NumPy→Rust lowering。</strong></p>

<p align="center">
  <a href="README.md">English</a> · <a href="README.ko.md">한국어</a> · <a href="README.zh-hans.md">简体中文</a> · <a href="README.zh-hant.md">繁體中文</a> · 日本語
</p>

`rextio-numpy` は、意図的に限定した型付き NumPy 操作を rust-numpy/`ndarray` 経由で Rust に lowering します。条件を満たす呼び出しはネイティブ実行されます。解析時に claim されない式は Rextio の Python fallback に残りますが、ネイティブ route の選択後に厳密な実行時境界を満たさなければ、静かに deopt せず拒否されます。

> **Public Alpha 0.1.3**（2026-07-27 リリース）。Python 3.11+、`rextio>=0.1.6,<0.2`、プラグイン API 1.5 が必要です。一般的な zero-copy、包括的な高速化、rank-2 行列積のサポートは主張しません。

## 検証済みの範囲

| 分野 | ネイティブ範囲 |
| --- | --- |
| 要素ごと | 同一 dtype、rank 1–2 の float64/float32/int64 配列に対する `+ - * /`、厳密な 2 引数 `numpy.add/subtract/multiply/divide`、一部の単項呼び出し |
| 比較 | 非連鎖 `== != < <= > >=`。常駐 bool マスクは `logical_not/and/or` と厳密な 3 引数 `numpy.where` で利用可能 |
| 線形代数 | 1-D float64/int64 に限定した `numpy.dot(a, b)` と `a.dot(b)` |
| reduction | 範囲を限定した全配列およびリテラル軸の `sum`、`mean`、`max`、`min`。正確な行列は後述 |
| fusion | サポート対象 dtype/rank 行列内の純粋な 2–8 要素演算チェーン |

shape エラー、整数 wraparound、順序、dtype、文書化された例外テキストは実 Cargo のネイティブ/fallback テストで検証されています。演算順が異なり得る浮動小数点 reduction と dot は許容誤差による等価性を使います。ネイティブ経路は NumPy `RuntimeWarning` を省略する場合があり、warning の同等性は認証対象外です。

## 仕組み

1. 型 annotation が限定された配列 dtype と rank を識別します。
2. プラグインは厳密にサポートされる式だけを claim し、rust-numpy/`ndarray` を使う Rust を生成します。
3. Rextio が受理した経路を Cargo でコンパイルします。claim されない NumPy コードは Python fallback のままです。

`F64Arr1` では、厳密な base `ndarray` 入力をネイティブ frame の間読み取り専用 rust-numpy view として保持し、rank-1 float64 結果を新しい NumPy 所有出力へ直接格納します。他のサポート対象 dtype/rank 経路は入力を所有 Rust 配列へコピーし、`ToPyArray` で返します。これは allocation 構造の改善であり、一般的な zero-copy や公開速度主張ではありません。

## クイックスタート

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

`F64Arr1` は実行時には `numpy.ndarray` の通常の alias です。annotation は静的解析を導くもので、それ自体が実行時検証ではありません。

## 正確なサポート範囲

### Annotation vocabulary

| Annotation | Rank | dtype |
| --- | ---: | --- |
| `F64Arr1`, `F64Arr2` | 1, 2 | float64 |
| `F32Arr1`, `F32Arr2` | 1, 2 | float32 |
| `I64Arr1`, `I64Arr2` | 1, 2 | int64 |

常駐 bool マスクには公開 annotation がなく、Python parameter または return 境界を越えられません。

### 操作と制約

- 要素演算子は rank-1/rank-2 NumPy broadcasting、配列↔配列、配列↔対応 Python scalar、zero-size 軸をサポートします。整数 true division は float64 を返します。
- 厳密な binary ufunc 形式は 2 個の positional operand だけを受け付けます。`out`、ufunc `where`、`dtype`、`casting`、追加引数、混合 dtype は fallback です。
- `numpy.where(condition, x, y)` は常駐 comparison/logical condition と同一 dtype の数値 branch、または一方の配列と対応 scalar を必要とします。少なくとも一方の branch は配列でなければならず、keyword、condition-only、scalar 2 個、混合 dtype は fallback です。
- `dot`：1-D float64/int64 のみ。float32 dot、すべての 2-D dot、`numpy.matmul`、`@` は fallback です。
- 全配列・keyword なし reduction：float64/int64 `sum`、float64 `mean`、int64 `max/min`。サポート対象の同等 ndarray method も含みます。
- 厳密に 1 個のリテラル整数軸：rank 1–2 の float64/int64 `sum`、float64 `mean`、int64 `max/min`。負の軸は正規化されます。動的/tuple/`None`/範囲外の軸や追加 option は fallback です。
- 単項 `numpy.negative`、`numpy.absolute`/`numpy.abs`、`numpy.square` は optional 引数や method 形式なしで rank 1–2 float64/float32/int64 をサポートします。
- fusion は 2–8 個の配列名演算からなる純 binary-op tree を受け付けます。float64/float32 は `+ - * /`、int64 は `+ - *`。範囲外の tree は通常の演算ごとの処理に残ります。

## 境界と fallback

- 実行時入力は厳密な base `numpy.ndarray` でなければなりません。`numpy.matrix`、`numpy.memmap`、独自 subclass はネイティブ境界で決定的な `TypeError` となり、自動 fallback ではありません。
- 厳密な ndarray view、読み取り専用入力、正負 stride をサポートします。借用入力 view を保持するのは `F64Arr1` だけで、他の経路は所有 copy を実体化します。
- 配列結果は `OWNDATA`、`base is None`、通常の resize 観測を持つ新しい NumPy 所有配列です。どの経路も `IntoPyArray` を使いません。
- float32 sum/mean/dot、int64 mean、浮動小数点 `max/min` は、安定した NumPy 等価累積または NaN/signed-zero 選択が証明されていないため fallback です。
- int64 `+ - *`、`sum`、`dot` は NumPy release build と同じ wraparound を使います。Python 整数 scalar 経路は signed i64 に限定され、範囲外の値はプラグイン helper の前に `OverflowError` になります。
- chained/identity/membership 比較、変更を伴う alias view、rank 2 超、reshape/view、未対応 method 形式、その他の NumPy API は fallback です。
- `@numba.*` で装飾された関数は Numba に委ねます。

## 根拠と非主張

[honest benchmark suite](benchmarks/README.md) は固定 `F64Arr1` サブセットを測定し、勝ち負けの両方を報告します。リポジトリは結果数値を意図的に commit しません。[boundary-allocation PoC](benchmarks/boundary_allocation_poc/README.md) は研究専用で、[rank-2 matmul harness](benchmarks/matmul_wave2/README.md) は **NO-GO / fallback** の製品判断を維持します。いずれも一般的な高速化の根拠ではありません。

## 開発

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python "rextio>=0.1.6,<0.2"
uv pip install --python .venv/bin/python --no-deps -e .
uv pip install --python .venv/bin/python pytest ruff mypy numpy hypothesis
.venv/bin/python -m pytest
```

リリース履歴は [CHANGELOG.md](CHANGELOG.md)、再現の詳細は各 benchmark README を参照してください。

## ライセンス

MIT
