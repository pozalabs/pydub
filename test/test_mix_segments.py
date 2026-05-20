import audioop

import pytest

from pydub import AudioSegment, _pydub_core
from test.helper import make_8bit, make_16bit, make_32bit


def _reference_mix(segments, sample_width):
    if not segments:
        raise ValueError
    result = segments[0]
    for seg in segments[1:]:
        longer, shorter = (result, seg) if len(result) >= len(seg) else (seg, result)
        mixed = audioop.add(longer[: len(shorter)], shorter, sample_width)
        result = mixed + longer[len(shorter) :]
    return result


def test_8bit_two_segments():
    a = make_8bit(10, 20, 30)
    b = make_8bit(5, 10, 15)
    result = _pydub_core.mix_segments([a, b], 1, [0, 0])
    assert result == _reference_mix([a, b], 1)


def test_16bit_two_segments():
    a = make_16bit(1000, 2000, 3000)
    b = make_16bit(100, 200, 300)
    result = _pydub_core.mix_segments([a, b], 2, [0, 0])
    assert result == _reference_mix([a, b], 2)


def test_32bit_two_segments():
    a = make_32bit(100000, 200000, 300000)
    b = make_32bit(10000, 20000, 30000)
    result = _pydub_core.mix_segments([a, b], 4, [0, 0])
    assert result == _reference_mix([a, b], 4)


def test_three_or_more_segments():
    a = make_16bit(1000, 2000, 3000)
    b = make_16bit(100, 200, 300)
    c = make_16bit(10, 20, 30)
    result = _pydub_core.mix_segments([a, b, c], 2, [0, 0, 0])
    assert result == _reference_mix([a, b, c], 2)


def test_different_lengths():
    a = make_16bit(1000, 2000, 3000, 4000)
    b = make_16bit(100, 200)
    result = _pydub_core.mix_segments([a, b], 2, [0, 0])
    expected = _reference_mix([a, b], 2)
    assert result == expected
    assert len(result) == len(a)


def test_different_lengths_three_segments():
    a = make_16bit(1000, 2000)
    b = make_16bit(100, 200, 300, 400)
    c = make_16bit(10, 20, 30)
    result = _pydub_core.mix_segments([a, b, c], 2, [0, 0, 0])
    expected = _reference_mix([a, b, c], 2)
    assert result == expected
    assert len(result) == len(b)


def test_saturation_clamp_8bit():
    a = make_8bit(120, -120)
    b = make_8bit(120, -120)
    result = _pydub_core.mix_segments([a, b], 1, [0, 0])
    assert result == _reference_mix([a, b], 1)


def test_saturation_clamp_16bit():
    a = make_16bit(32000, -32000)
    b = make_16bit(32000, -32000)
    result = _pydub_core.mix_segments([a, b], 2, [0, 0])
    assert result == _reference_mix([a, b], 2)


def test_saturation_clamp_32bit():
    a = make_32bit(2147483000, -2147483000)
    b = make_32bit(2147483000, -2147483000)
    result = _pydub_core.mix_segments([a, b], 4, [0, 0])
    assert result == _reference_mix([a, b], 4)


def test_empty_list_raise_error():
    with pytest.raises(ValueError):
        _pydub_core.mix_segments([], 2, [])


def test_invalid_sample_width_raise_error():
    a = make_16bit(1000)
    with pytest.raises(ValueError):
        _pydub_core.mix_segments([a], 3, [0])


def test_two_segments_match_overlay():
    a = AudioSegment.silent(duration=100)
    b = AudioSegment.silent(duration=100)
    mixed = AudioSegment.mix(a, b)
    overlaid = a.overlay(b, position=0)
    assert mixed.raw_data == overlaid.raw_data


def test_single_segment_return_same_object():
    a = AudioSegment.silent(duration=100)
    result = AudioSegment.mix(a)
    assert result is a


def test_empty_raise_error():
    with pytest.raises(ValueError):
        AudioSegment.mix()


def test_different_lengths_audiosegment():
    a = AudioSegment.silent(duration=200)
    b = AudioSegment.silent(duration=100)
    result = AudioSegment.mix(a, b)
    assert len(result) == len(a)


def test_mix_with_position_match_overlay_chain():
    a = AudioSegment.silent(duration=200)
    b = AudioSegment.silent(duration=100)
    c = AudioSegment.silent(duration=100)

    mixed = AudioSegment.mix(a, (b, 50), (c, 100))
    overlaid = a.overlay(b, position=50).overlay(c, position=100)

    assert mixed.raw_data == overlaid.raw_data


def test_position_extend_output_beyond_longest_segment():
    a = AudioSegment.silent(duration=100)
    b = AudioSegment.silent(duration=100)

    result = AudioSegment.mix(a, (b, 150))

    assert len(result) == 250


def test_all_zero_positions_match_plain_mix():
    a = AudioSegment.silent(duration=100)
    b = AudioSegment.silent(duration=100)
    c = AudioSegment.silent(duration=50)

    with_positions = AudioSegment.mix((a, 0), (b, 0), (c, 0))
    without_positions = AudioSegment.mix(a, b, c)

    assert with_positions.raw_data == without_positions.raw_data
