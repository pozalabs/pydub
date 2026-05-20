# mix 세그먼트별 position 지정

## 목표

`AudioSegment.mix()`에서 세그먼트별 시작 위치(ms)를 지정하여, overlay 체이닝 없이 N개 세그먼트를 한 번에 합성할 수 있도록 확장

## 상세

### Python API 변경

기존 `mix(*segs: Self)` 시그니처를 확장하여 각 인자가 `AudioSegment` 또는 `(AudioSegment, position_ms)` 튜플을 받을 수 있도록 변경:

```python
# before
@classmethod
def mix(cls, *segs: Self) -> Self:

# after
@classmethod
def mix(cls, *segs: Self | tuple[Self, int]) -> Self:
```

- `AudioSegment`만 전달하면 position=0으로 간주
- `(AudioSegment, int)` 튜플로 전달하면 두 번째 값이 시작 위치(ms)
- position은 밀리초 단위, 0 이상의 정수

호출 예시:

```python
# 기존 호출 그대로 동작 (하위 호환)
AudioSegment.mix(seg1, seg2, seg3)

# 세그먼트별 position 지정
AudioSegment.mix(
    seg1,
    (seg2, 1000),
    (seg3, 2000),
)
```

### Python 구현 흐름

1. `*segs`에서 AudioSegment와 position을 분리
2. 모든 AudioSegment를 `_sync()`로 동기화
3. 각 position(ms)을 `_ms_to_byte_offset()`으로 바이트 오프셋 변환
4. Rust `mix_segments(segments, sample_width, positions)` 호출
5. 결과로 `_spawn()` 호출

### Rust `mix_segments` 시그니처 변경

```rust
// before
pub fn mix_segments<'py>(
    py: Python<'py>,
    segments: Vec<Bound<'py, PyBytes>>,
    sample_width: i32,
) -> PyResult<Bound<'py, PyBytes>>

// after
pub fn mix_segments<'py>(
    py: Python<'py>,
    segments: Vec<Bound<'py, PyBytes>>,
    sample_width: i32,
    positions: Vec<i32>,
) -> PyResult<Bound<'py, PyBytes>>
```

- `positions` 길이는 `segments` 길이와 동일해야 함
- position은 바이트 오프셋 (Python 측에서 ms -> byte 변환)
- 각 position은 0 이상, `sample_width`의 배수

### Rust 구현 로직

1. 출력 버퍼 크기: `max(positions[i] + segments[i].len())` for all i
2. 출력 버퍼를 0으로 초기화
3. 각 세그먼트를 `positions[i]` 오프셋부터 출력 버퍼에 mix (기존 `define_mix!` 매크로 재사용)
4. 기존처럼 "가장 긴 세그먼트를 base로 복사 후 나머지를 mix"하는 최적화는 제거하고, 0-initialized 버퍼에 모든 세그먼트를 순차 mix

### 변경 대상 파일

- `pydub/audio_segment.py`: `mix()` 메서드 시그니처 및 구현 변경
- `src/overlay.rs`: `mix_segments()` 시그니처 및 구현 변경
- `test/test_mix_segments.py`: position 관련 테스트 추가

## 경계

- 항상: 기존 `mix(seg1, seg2)` 호출이 동일하게 동작해야 함 (하위 호환)
- 항상: position은 밀리초 단위 비음수 정수
- 절대: `gain_during_overlay`, `loop`, `times` 파라미터는 추가하지 않음
- 절대: `overlay()` 메서드는 변경하지 않음

## 검증

- position 지정한 mix 결과가 동일한 position으로 overlay를 체이닝한 결과와 바이트 단위로 일치하는지 검증 (overlay 체이닝은 gain_during_overlay=None이므로 동일 조건)
- position으로 인해 출력 길이가 가장 긴 세그먼트보다 길어지는 케이스 검증 (position + segment length > max segment length)
- 모든 세그먼트에 position=0을 지정했을 때 기존 mix와 동일한 결과 검증
