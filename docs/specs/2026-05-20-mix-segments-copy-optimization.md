# mix_segments 첫 세그먼트 copy 최적화

## 목표

`mix_segments`에서 첫 번째 세그먼트를 zero 버퍼와 mix하는 대신 `copy_from_slice`로 직접 복사하여 불필요한 mix 연산을 제거

## 상세

### 현재 동작

`mix_segments`는 출력 버퍼를 0으로 초기화한 뒤, 모든 세그먼트를 동일하게 `mix_fn`으로 처리:

```rust
out_buf.fill(0);
for (seg, &pos) in slices.iter().zip(positions.iter()) {
    // mix_fn(0, seg[i]) → 사실상 복사와 동일한 연산
}
```

`mix(0, x)`는 항상 `x`를 반환하므로, 첫 번째 세그먼트는 mix 없이 복사해도 결과가 동일함

### 왜 첫 번째 세그먼트인가

`mix`는 saturating addition(`clamp(a + b)`)이므로 교환법칙은 성립하지만 결합법칙은 성립하지 않음. 가장 긴 세그먼트를 copy 대상으로 선택하면 fold 순서가 바뀌어, 3개 이상 세그먼트가 겹치고 중간 합이 overflow/underflow하는 경우 기존과 결과가 달라짐. 첫 번째 세그먼트(index 0)를 copy하면 fold 순서가 보존되어 bit-for-bit 동일한 결과를 보장

### 변경 내용

#### 내부 함수 추출

`mix_segments`의 바이트 수준 mix 루프를 순수 Rust 함수로 추출:

```rust
fn mix_segments_raw(out_buf: &mut [u8], slices: &[&[u8]], positions: &[i32], sample_width: usize)
```

`mix_segments`는 PyO3 입력 검증과 `PyBytes` 할당만 수행하고, 실제 mix 로직은 `mix_segments_raw`에 위임

#### copy + mix 최적화

`mix_segments_raw` 내부에서 첫 번째 세그먼트를 `copy_from_slice`로 복사하고, 나머지만 mix:

변경 후 흐름:

1. `out_buf.fill(0)` (기존과 동일)
2. 첫 번째 세그먼트(`slices[0]`)를 `out_buf[pos..pos+len].copy_from_slice(seg)`로 복사
3. 나머지 세그먼트(index 1~)만 `mix_at!` 매크로로 mix 처리

변경 후 `mix_at!` 매크로:

```rust
macro_rules! mix_at {
    ($sample_type:ty, $mix_fn:ident) => {
        for (seg, &pos) in slices[1..].iter().zip(positions[1..].iter()) {
            // 기존 mix 로직 동일
        }
    };
}
```

### 변경 파일

- `src/overlay.rs`:
  - `mix_segments_raw` 함수 추가 (바이트 수준 mix 로직)
  - `mix_segments`에서 `mix_segments_raw` 호출로 변경
  - `mix_segments_raw` 내부에 첫 세그먼트 copy 최적화 적용

### 기대 효과

- 2개 세그먼트 기준: mix 호출 2N -> N으로 감소, `overlay_segments`와 동등한 수준
- N개 세그먼트 기준: 첫 번째 세그먼트의 sample 수만큼 mix 연산 절약 (memcpy로 대체)
- 피크 메모리 변경 없음

## 경계

- 항상: `mix(0, x) == x` 동치성이 보장되는 범위에서만 copy 적용 (i8, i16, i32 signed integer - 현재 지원 타입 전부 해당)
- 항상: 기존 Python 테스트(`test/test_mix_segments.py`)가 모두 통과해야 함
- 절대: 함수 시그니처, 외부 동작 변경 없음

## 검증

Rust 단위 테스트를 `src/overlay.rs`의 `mod tests`에 추가 (`mix_segments_raw`를 직접 호출):

- position이 다른 세그먼트들의 mix 결과가 순차 mix와 동일한지 검증: copy된 첫 세그먼트와 다른 세그먼트의 overlap 영역에서 mix가 정확히 수행되는지 확인

기존 Python 테스트(`uv run pytest test/test_mix_segments.py`)로 통합 동작 동등성 검증. `_reference_mix`와 비교하는 기존 테스트가 회귀를 감지함
