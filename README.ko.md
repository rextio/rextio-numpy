# rextio-numpy

<p align="center">
  <img src="./assets/readme/rextio-icon.png" width="96" alt="Rextio 아이콘">
</p>

<p align="center"><strong>Rextio가 안전성을 증명할 수 있는 코드를 위한 한정된 NumPy→Rust lowering.</strong></p>

<p align="center">
  <a href="README.md">English</a> · 한국어 · <a href="README.zh-hans.md">简体中文</a> · <a href="README.zh-hant.md">繁體中文</a> · <a href="README.ja.md">日本語</a>
</p>

`rextio-numpy`는 의도적으로 제한된 타입 지정 NumPy 연산을 rust-numpy/`ndarray` 기반 Rust로 lowering합니다. 조건에 맞는 호출은 네이티브로 실행됩니다. 분석 시 claim되지 않은 식은 Rextio의 Python fallback에 남지만, 네이티브 route가 선택된 뒤 정확한 런타임 경계를 충족하지 못하면 조용히 deopt하지 않고 거부됩니다.

> **공개 Alpha 0.1.3** (2026-07-27 릴리스). Python 3.11+, `rextio>=0.1.6,<0.2`, 플러그인 API 1.5가 필요합니다. 일반적인 zero-copy, 포괄적인 속도 향상, rank-2 행렬 곱셈 지원을 주장하지 않습니다.

## 검증된 범위

| 영역 | 네이티브 범위 |
| --- | --- |
| 원소별 연산 | 같은 dtype의 rank 1–2 float64/float32/int64 배열에 대한 `+ - * /`, 정확히 두 인자의 `numpy.add/subtract/multiply/divide`, 일부 단항 호출 |
| 비교 | 체인 아닌 `== != < <= > >=`; 상주 bool 마스크는 `logical_not/and/or`와 정확히 세 인자의 `numpy.where`에 사용 가능 |
| 선형대수 | 1-D float64/int64에 한정된 `numpy.dot(a, b)`와 `a.dot(b)` |
| reduction | 제한된 전체 배열 및 리터럴 축 `sum`, `mean`, `max`, `min`; 정확한 행렬은 아래 참고 |
| fusion | 지원 dtype/rank 행렬 안의 순수한 2–8개 원소별 연산 체인 |

shape 오류, 정수 wraparound, 순서, dtype, 문서화된 예외 문자열은 실제 Cargo 네이티브/fallback 테스트로 검증됩니다. 연산 순서가 달라질 수 있는 부동소수 reduction과 dot은 허용 오차 기반 동등성을 사용합니다. 네이티브 경로는 NumPy `RuntimeWarning`을 생략할 수 있으며 warning 동등성은 인증 대상이 아닙니다.

## 동작 방식

1. 타입 annotation이 제한된 배열 dtype과 rank를 식별합니다.
2. 플러그인은 정확히 지원되는 식만 claim하고 rust-numpy/`ndarray`를 사용하는 Rust를 생성합니다.
3. Rextio가 허용된 경로를 Cargo로 컴파일합니다. claim되지 않은 NumPy 코드는 Python fallback으로 남습니다.

`F64Arr1`의 정확한 base `ndarray` 입력은 네이티브 프레임 동안 읽기 전용 rust-numpy view로 유지되고, rank-1 float64 결과는 새 NumPy 소유 출력에 직접 채워집니다. 다른 지원 dtype/rank 경로는 입력을 소유 Rust 배열로 복사하고 `ToPyArray`로 반환합니다. 이는 allocation 구조 개선일 뿐 일반 zero-copy나 공개 속도 주장이 아닙니다.

## 빠른 시작

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

`F64Arr1`은 런타임에서 `numpy.ndarray`의 단순 alias입니다. annotation은 정적 분석을 안내하며 그 자체가 런타임 검증은 아닙니다.

## 정확한 지원 범위

### Annotation vocabulary

| Annotation | Rank | dtype |
| --- | ---: | --- |
| `F64Arr1`, `F64Arr2` | 1, 2 | float64 |
| `F32Arr1`, `F32Arr2` | 1, 2 | float32 |
| `I64Arr1`, `I64Arr2` | 1, 2 | int64 |

상주 bool 마스크에는 공개 annotation이 없으며 Python parameter나 return 경계를 넘을 수 없습니다.

### 연산과 제약

- 원소별 연산자는 rank-1/rank-2 NumPy broadcasting, 배열↔배열과 배열↔일치하는 Python scalar, zero-size 축을 지원합니다. 정수 true division 결과는 float64입니다.
- 정확한 binary ufunc 형식은 positional operand 두 개만 받습니다. `out`, ufunc `where`, `dtype`, `casting`, 추가 인자와 혼합 dtype은 fallback입니다.
- `numpy.where(condition, x, y)`는 상주 comparison/logical condition과 같은 dtype의 숫자 branch 또는 배열 하나와 일치하는 scalar를 요구합니다. 적어도 한 branch는 배열이어야 하며 keyword, condition-only, 두 scalar, 혼합 dtype은 fallback입니다.
- `dot`: 1-D float64/int64만 지원합니다. float32 dot, 모든 2-D dot, `numpy.matmul`, `@`는 fallback입니다.
- 전체 배열, keyword 없는 reduction: float64/int64 `sum`, float64 `mean`, int64 `max/min`; 지원되는 동등한 ndarray method도 포함됩니다.
- 정확히 하나의 리터럴 정수 축: rank 1–2 float64/int64 `sum`, float64 `mean`, int64 `max/min`. 음수 축은 정규화됩니다. 동적/tuple/`None`/범위 밖 축과 추가 option은 fallback입니다.
- 단항 `numpy.negative`, `numpy.absolute`/`numpy.abs`, `numpy.square`는 선택 인자나 method 형식 없이 rank 1–2 float64/float32/int64를 지원합니다.
- fusion은 2–8개의 배열 이름 binary-op tree를 받습니다. float64/float32는 `+ - * /`, int64는 `+ - *`를 사용합니다. 범위 밖 tree는 일반 원소별 처리로 남습니다.

## 경계와 fallback

- 런타임 입력은 정확한 base `numpy.ndarray`여야 합니다. `numpy.matrix`, `numpy.memmap`, 사용자 subclass는 결정적인 native-boundary `TypeError`를 발생시키며 자동 fallback이 아닙니다.
- 정확한 ndarray view, 읽기 전용 입력, 양/음 stride를 지원합니다. 입력 view를 빌려 유지하는 경로는 `F64Arr1`뿐이며 나머지는 소유 copy를 만듭니다.
- 배열 결과는 `OWNDATA`, `base is None`, 일반 resize 관찰값을 가진 새 NumPy 소유 배열입니다. 어떤 경로도 `IntoPyArray`를 사용하지 않습니다.
- float32 sum/mean/dot, int64 mean, 부동소수 `max/min`은 안정적인 NumPy 동등 누산 또는 NaN/signed-zero 선택이 증명되지 않아 fallback입니다.
- int64 `+ - *`, `sum`, `dot`은 NumPy release build와 같은 wraparound를 사용합니다. Python 정수 scalar 경로는 signed i64로 제한되며 범위 밖 값은 플러그인 helper 전에 `OverflowError`를 발생시킵니다.
- chained/identity/membership 비교, 변경되는 alias view, rank 2 초과, reshape/view, 미지원 method 형식과 기타 NumPy API는 fallback입니다.
- `@numba.*`로 장식된 함수는 Numba에 맡깁니다.

## 근거와 비주장

[정직한 benchmark suite](benchmarks/README.md)는 고정된 `F64Arr1` 부분집합을 측정하고 이득과 손실을 모두 보고하며, 저장소는 결과 수치를 의도적으로 commit하지 않습니다. [Boundary-allocation PoC](benchmarks/boundary_allocation_poc/README.md)는 연구 전용이고 [rank-2 matmul harness](benchmarks/matmul_wave2/README.md)는 **NO-GO / fallback** 제품 결정을 유지합니다. 어느 것도 일반 속도 향상의 근거가 아닙니다.

## 개발

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python "rextio>=0.1.6,<0.2"
uv pip install --python .venv/bin/python --no-deps -e .
uv pip install --python .venv/bin/python pytest ruff mypy numpy hypothesis
.venv/bin/python -m pytest
```

릴리스 이력은 [CHANGELOG.md](CHANGELOG.md), 재현 세부사항은 각 benchmark README를 참고하세요.

## 라이선스

MIT
