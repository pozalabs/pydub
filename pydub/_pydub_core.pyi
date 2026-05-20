def overlay_segments(
    seg1_data: bytes,
    seg2_data: bytes,
    sample_width: int,
    position: int,
    times: int,
    gain_during_overlay: int = 0,
) -> bytes: ...
def mix_segments(
    segments: list[bytes],
    sample_width: int,
    positions: list[int],
) -> bytes: ...
def fade_segment(
    data: bytes,
    sample_width: int,
    start_byte: int,
    end_byte: int,
    from_power: float,
    to_power: float,
) -> bytes: ...
def extend_24bit_to_32bit(data: bytes) -> bytes: ...
def measure_rms(
    data: bytes,
    sample_width: int,
    channels: int,
    sample_rate: int,
) -> float: ...
def measure_peak(
    data: bytes,
    sample_width: int,
    channels: int,
) -> float: ...

class Loudness:
    @property
    def integrated(self) -> float: ...
    @property
    def momentary(self) -> list[float]: ...

def measure_loudness(
    data: bytes,
    sample_width: int,
    channels: int,
    sample_rate: int,
) -> Loudness: ...
