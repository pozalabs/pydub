from __future__ import annotations

import array
import audioop
import base64
import contextlib
import dataclasses
import functools
import io
import os
import subprocess
import sys
import wave
from collections.abc import Generator
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryFile
from typing import IO, Any, Literal, Self, TypedDict, Unpack, overload

from . import _compression, _meter, _pydub_core, _wav
from ._subprocess import _ConversionCommand, _PopenParams
from .enums import SampleWidth
from .exceptions import (
    CouldntDecodeError,
    CouldntEncodeError,
    InvalidDuration,
    MissingAudioParameter,
    TooManyMissingFrames,
)
from .logging_utils import log_conversion, log_subprocess_output
from .utils import (
    db_to_float,
    get_encoder_name,
    mediainfo_json,
    ratio_to_db,
)

AUDIO_FILE_EXT_ALIASES = {
    "m4a": "mp4",
    "wave": "wav",
}


def _normalize_format(audio_format: str | None) -> str | None:
    audio_format = audio_format and audio_format.lower()
    return audio_format and AUDIO_FILE_EXT_ALIASES.get(audio_format, audio_format)


def _match_format(audio_format: str | None, filename: str | None, target: str) -> bool:
    target = target.lower()
    if audio_format == target:
        return True
    if filename is not None:
        return filename.lower().endswith(f".{target}")
    return False


def _infer_codec(info: dict[str, Any]) -> str:
    audio_streams = [x for x in info["streams"] if x["codec_type"] == "audio"]
    audio_codec = audio_streams[0].get("codec_name")
    is_fltp = audio_streams[0].get("sample_fmt") == "fltp"
    is_only_16bit_codec = audio_codec in {"mp3", "mp4", "aac", "webm", "ogg"}
    bits_per_sample = 16 if is_fltp and is_only_16bit_codec else audio_streams[0]["bits_per_sample"]
    return "pcm_u8" if bits_per_sample == 8 else f"pcm_s{bits_per_sample}le"


@dataclasses.dataclass
class _AudioParams:
    sample_width: int | None = None
    frame_rate: int | None = None
    channels: int | None = None

    def __post_init__(self):
        data = self.__dict__.values()

        if all(v is None for v in data) or all(v is not None for v in data):
            return

        # prevent partial specification of arguments
        raise MissingAudioParameter("Either all audio parameters or no parameter must be specified")

    @property
    def has_params(self):
        return all(v is not None for v in self.__dict__.values())

    @property
    def frame_width(self) -> int:
        if self.has_params:
            return self.sample_width * self.channels

        raise ValueError("'frame_width' is not available until all audio parameters are specified")

    def is_data_frame_width_valid(self, data: bytes) -> bool:
        return len(data) % self.frame_width == 0


class _AudioSegmentInitDef(TypedDict, total=False):
    sample_width: int
    frame_rate: int
    channels: int


@contextlib.contextmanager
def _ffmpeg_tmp_files(audio_segment: "AudioSegment") -> Generator[tuple[IO[bytes], IO[bytes]]]:
    data = NamedTemporaryFile(mode="wb", delete=False)
    output = NamedTemporaryFile(mode="w+b", delete=False)
    try:
        audio_segment._write_wav(data)
        yield data, output
    finally:
        data.close()
        output.close()
        Path(data.name).unlink()
        Path(output.name).unlink()


class AudioSegment:
    """
    AudioSegments are *immutable* objects representing segments of audio
    that can be manipulated using python code.

    AudioSegments are slicable using milliseconds.
    for example:
        a = AudioSegment.from_mp3(mp3file)
        first_second = a[:1000] # get the first second of an mp3
        slice = a[5000:10000] # get a slice from 5 to 10 seconds of an mp3
    """

    converter = get_encoder_name()  # either ffmpeg or avconv

    DEFAULT_CODECS = {"ogg": "libvorbis"}

    def __init__(
        self,
        data: bytes | array.array | IO,
        **kwargs: Unpack[_AudioSegmentInitDef],
    ):
        if isinstance(data, array.array):
            data = data.tobytes()

        self.sample_width = kwargs.pop("sample_width", None)
        self.frame_rate = kwargs.pop("frame_rate", None)
        self.channels = kwargs.pop("channels", None)

        audio_params = _AudioParams(
            sample_width=self.sample_width,
            frame_rate=self.frame_rate,
            channels=self.channels,
        )

        if audio_params.has_params:
            self._init_with_audio_params(data=data, audio_params=audio_params)
        else:
            self._init_with_data(data)

        if self.sample_width == SampleWidth.PCM24:
            self._extend_24bit_to_32bit()

    def _init_with_audio_params(self, data: bytes, audio_params: _AudioParams) -> None:
        if not audio_params.is_data_frame_width_valid(data):
            raise ValueError("Data length must be a multiple of '(sample_width * channels)'")

        self._data = data

    def _init_with_data(self, data: str | bytes | IO) -> None:
        try:
            data = data if isinstance(data, (str, bytes)) else data.read()
        except OSError:
            d = b""
            while reader := data.read(2**31 - 1):
                d += reader
            data = d

        wav_data = _wav.read_audio(data)

        self.channels = wav_data.channels
        self.sample_width = SampleWidth.from_bit_depth(wav_data.bits_per_sample)
        self.frame_rate = wav_data.sample_rate
        self._data = wav_data.raw_data
        if self.sample_width == SampleWidth.PCM8:
            # convert from unsigned integers in wav
            self._data = audioop.bias(self._data, 1, -128)

    def _extend_24bit_to_32bit(self) -> None:
        if self.sample_width != SampleWidth.PCM24:
            return

        self._data = _pydub_core.extend_24bit_to_32bit(self._data)
        self.sample_width = SampleWidth.PCM32

    @classmethod
    def empty(cls) -> Self:
        return cls(b"", sample_width=SampleWidth.PCM8, frame_rate=1, channels=1)

    @classmethod
    def silent(cls, duration: int = 1000, frame_rate: int = 11025) -> Self:
        """
        Generate a silent audio segment.
        duration specified in milliseconds (default duration: 1000ms, default frame_rate: 11025).
        """
        frames = int(frame_rate * (duration / 1000.0))
        data = b"\0\0" * frames
        return cls(data, sample_width=SampleWidth.PCM16, frame_rate=frame_rate, channels=1)

    @classmethod
    def from_mono_audiosegments(cls, *mono_segments: Self) -> Self:
        if not len(mono_segments):
            raise ValueError("At least one AudioSegment instance is required")

        segs = cls._sync(*mono_segments)

        if segs[0].channels != 1:
            raise ValueError(
                "'from_mono_audiosegments' requires all arguments are mono AudioSegment instances"
            )

        channels = len(segs)
        sample_width = segs[0].sample_width
        frame_rate = segs[0].frame_rate

        frame_count = max(int(seg.frame_count()) for seg in segs)
        data = array.array(segs[0].array_type, b"\0" * (frame_count * sample_width * channels))

        for i, seg in enumerate(segs):
            data[i::channels] = seg.get_array_of_samples()

        return cls(
            data,
            channels=channels,
            sample_width=sample_width,
            frame_rate=frame_rate,
        )

    @classmethod
    def mix(cls, *segs: Self | tuple[Self, int]) -> Self:
        if not segs:
            raise ValueError("At least one AudioSegment is required")

        segments: list[Self] = []
        positions_ms: list[int] = []
        for seg in segs:
            if isinstance(seg, tuple):
                segments.append(seg[0])
                positions_ms.append(seg[1])
            else:
                segments.append(seg)
                positions_ms.append(0)

        if len(segments) == 1 and positions_ms[0] == 0:
            return segments[0]

        synced = cls._sync(*segments)
        ref = synced[0]
        result = _pydub_core.mix_segments(
            [seg.raw_data for seg in synced],
            ref.sample_width,
            [ref._ms_to_byte_offset(ms=pos) for pos in positions_ms],
        )
        return ref._spawn(data=result)

    @classmethod
    def from_file(
        cls,
        file: str | os.PathLike | IO[bytes],
        format: str | None = None,
        codec: str | None = None,
        parameters: list[str] | None = None,
        start_second: int | None = None,
        duration: int | None = None,
        **kwargs: Any,
    ) -> Self:
        orig_file = file
        try:
            filename = os.fsdecode(file)
        except TypeError:
            filename = None

        close_file = False
        if isinstance(file, (str, os.PathLike)):
            file = open(file, "rb")
            close_file = True
        elif isinstance(file, io.BufferedReader):
            close_file = True

        is_compressed, compressor = _compression.is_compressed(file)
        if is_compressed:
            content = file.read()
            if close_file:
                file.close()
            return cls.from_file(
                file=io.BytesIO(_compression.decompress(compressor=compressor, content=content)),
                format=format,
                codec=codec,
                parameters=parameters,
                start_second=start_second,
                duration=duration,
                **kwargs,
            )

        audio_format = _normalize_format(format)
        is_format = functools.partial(_match_format, audio_format=audio_format, filename=filename)

        if is_format(target="wav"):
            try:
                obj = cls._from_safe_wav(file)._segmented(
                    start_second=start_second, duration=duration
                )
                if close_file:
                    file.close()
                return obj
            except:  # noqa: E722
                file.seek(0)
        elif is_format(target="raw") or is_format(target="pcm"):
            obj = cls._from_raw(file=file, start_second=start_second, duration=duration, **kwargs)
            if close_file:
                file.close()
            return obj

        return cls._from_ffmpeg(
            file=file,
            orig_file=orig_file,
            filename=filename,
            close_file=close_file,
            audio_format=audio_format,
            codec=codec,
            parameters=parameters,
            start_second=start_second,
            duration=duration,
            **kwargs,
        )

    @classmethod
    def from_mp3(
        cls, file: str | os.PathLike | IO[bytes], parameters: list[str] | None = None
    ) -> Self:
        return cls.from_file(file, "mp3", parameters=parameters)

    @classmethod
    def from_flv(
        cls, file: str | os.PathLike | IO[bytes], parameters: list[str] | None = None
    ) -> Self:
        return cls.from_file(file, "flv", parameters=parameters)

    @classmethod
    def from_ogg(
        cls, file: str | os.PathLike | IO[bytes], parameters: list[str] | None = None
    ) -> Self:
        return cls.from_file(file, "ogg", parameters=parameters)

    @classmethod
    def from_wav(
        cls, file: str | os.PathLike | IO[bytes], parameters: list[str] | None = None
    ) -> Self:
        return cls.from_file(file, "wav", parameters=parameters)

    @classmethod
    def from_raw(cls, file: str | os.PathLike | IO[bytes], **kwargs: Any) -> Self:
        return cls.from_file(
            file,
            "raw",
            sample_width=kwargs["sample_width"],
            frame_rate=kwargs["frame_rate"],
            channels=kwargs["channels"],
        )

    @classmethod
    def _from_safe_wav(cls, file: IO[bytes]) -> Self:
        file.seek(0)
        return cls(data=file)

    @classmethod
    def _from_raw(
        cls,
        file: IO[bytes],
        start_second: int | None,
        duration: int | None,
        **kwargs: Any,
    ) -> Self:
        return cls(
            data=file.read(),
            sample_width=kwargs["sample_width"],
            frame_rate=kwargs["frame_rate"],
            channels=kwargs["channels"],
        )._segmented(start_second=start_second, duration=duration)

    @classmethod
    def _from_ffmpeg(
        cls,
        file: IO[bytes],
        orig_file: Any,
        filename: str | None,
        close_file: bool,
        audio_format: str | None,
        codec: str | None,
        parameters: list[str] | None,
        start_second: int | None,
        duration: int | None,
        **kwargs: Any,
    ) -> Self:
        try:
            conversion_command = _ConversionCommand.init(cls.converter)
            if audio_format is not None:
                conversion_command = conversion_command.with_format(audio_format)
            if codec is not None:
                conversion_command = conversion_command.with_codec(codec)

            read_ahead_limit = kwargs.get("read_ahead_limit", -1)
            if filename is not None:
                conversion_command = conversion_command.with_filename(filename)
                popen_params = _PopenParams.empty()
            else:
                conversion_command = conversion_command.without_filename(read_ahead_limit)
                popen_params = _PopenParams.pipe(file.read())

            if codec is None:
                info = mediainfo_json(orig_file, read_ahead_limit=read_ahead_limit)
                if info:
                    conversion_command = conversion_command.with_codec(_infer_codec(info))

            conversion_command = conversion_command.remove_video().with_format("wav")

            if start_second is not None:
                conversion_command = conversion_command.with_start_second(start_second)
            if duration is not None:
                conversion_command = conversion_command.with_duration(duration)

            conversion_command = conversion_command.from_stdin()

            if parameters is not None:
                conversion_command = conversion_command.with_parameters(parameters)

            log_conversion(conversion_command)

            p = subprocess.Popen(
                conversion_command,
                stdin=popen_params.stdin,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            p_out, p_err = p.communicate(input=popen_params.data)

            if p.returncode or not p_out:
                error = p_err.decode(errors="ignore")
                raise CouldntDecodeError(
                    f"Decoding failed. {cls.converter} returned error code: {p.returncode}\n\nOutput from {cls.converter}:\n\n{error}"
                )

            p_out = _wav.fix_headers(p_out)
            obj = cls(p_out)

            return obj._segmented(start_second=None, duration=duration)
        finally:
            if close_file:
                file.close()

    def _segmented(self, start_second: int | None = None, duration: int | None = None) -> Self:
        match start_second, duration:
            case None, None:
                return self
            case _, None:
                return self[start_second * 1000 :]
            case None, _:
                return self[: duration * 1000]
            case _, _:
                return self[start_second * 1000 : (start_second + duration) * 1000]

        raise ValueError("Invalid arguments for 'start_second' and 'duration'")

    def __len__(self):
        """
        returns the length of this audio segment in milliseconds
        """
        return round(1000 * (self.frame_count() / self.frame_rate))

    def __eq__(self, other):
        try:
            return self._data == other.raw_data
        except:  # noqa: E722
            return False

    def __hash__(self):
        return hash(AudioSegment) ^ hash(
            (self.channels, self.frame_rate, self.sample_width, self._data)
        )

    def __ne__(self, other):
        return not (self == other)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    @overload
    def __getitem__(self, millisecond: int) -> Self: ...
    @overload
    def __getitem__(self, millisecond: slice) -> Self | Generator[Self, None, None]: ...

    def __getitem__(self, millisecond: int | slice) -> Self | Generator[Self, None, None]:
        if isinstance(millisecond, slice):
            if millisecond.step:
                return (
                    self._get_segment(i, i + millisecond.step)
                    for i in range(*millisecond.indices(len(self)))
                )

            start = millisecond.start if millisecond.start is not None else 0
            end = millisecond.stop if millisecond.stop is not None else len(self)

            start = min(start, len(self))
            end = min(end, len(self))
        else:
            start = millisecond
            end = millisecond + 1

        return self._get_segment(start=start, end=end)

    def __add__(self, arg):
        if isinstance(arg, AudioSegment):
            return self.append(arg, crossfade=0)
        else:
            return self.apply_gain(arg)

    def __radd__(self, rarg):
        """
        Permit use of sum() builtin with an iterable of AudioSegments
        """
        if rarg == 0:
            return self
        raise TypeError("Gains must be the second addend after the AudioSegment")

    def __sub__(self, arg):
        if isinstance(arg, AudioSegment):
            raise TypeError("AudioSegment objects cannot be subtracted from each other")
        else:
            return self.apply_gain(-arg)

    def __mul__(self, arg):
        """
        If the argument is an AudioSegment, overlay the multiplied audio
        segment.

        If it's a number, just use the string multiply operation to repeat the
        audio.

        The following would return an AudioSegment that contains the
        audio of audio_seg eight times

        `audio_seg * 8`
        """
        if isinstance(arg, AudioSegment):
            return self.overlay(arg, position=0, loop=True)
        else:
            return self._spawn(data=self._data * arg)

    def __deepcopy__(self, memo: dict[int, object]) -> AudioSegment:
        return self._spawn(self._data)

    def _repr_html_(self):
        src = """
                    <audio controls>
                        <source src="data:audio/mpeg;base64,{base64}" type="audio/mpeg"/>
                        Your browser does not support the audio element.
                    </audio>
                  """
        fh = self.export()
        data = base64.b64encode(fh.read()).decode("ascii")
        return src.format(base64=data)

    @property
    def raw_data(self) -> bytes:
        """
        public access to the raw audio data as a bytestring
        """
        return self._data

    @property
    def frame_width(self) -> int:
        return self.channels * self.sample_width

    @property
    def array_type(self) -> str:
        return SampleWidth(self.sample_width).array_type

    @property
    def rms(self) -> int:
        return audioop.rms(self._data, self.sample_width)

    @property
    def dBFS(self) -> float:
        rms = self.rms
        if not rms:
            return -float("infinity")
        return ratio_to_db(self.rms / self.max_possible_amplitude)

    @property
    def max(self) -> int:
        return audioop.max(self._data, self.sample_width)

    @property
    def max_possible_amplitude(self) -> float:
        bits = SampleWidth(self.sample_width).bit_depth
        max_possible_val = 2**bits

        # since half is above 0 and half is below the max amplitude is divided
        return max_possible_val / 2

    @property
    def max_dBFS(self) -> float:
        return ratio_to_db(self.max, self.max_possible_amplitude)

    @property
    def duration_seconds(self) -> float:
        return self.frame_rate and self.frame_count() / self.frame_rate or 0.0

    def get_array_of_samples(self, array_type_override=None):
        """
        returns the raw_data as an array of samples
        """
        if array_type_override is None:
            array_type_override = self.array_type
        return array.array(array_type_override, self._data)

    def get_frame(self, index):
        frame_start = index * self.frame_width
        frame_end = frame_start + self.frame_width
        return self._data[frame_start:frame_end]

    def frame_count(self, ms=None):
        """
        returns the number of frames for the given number of milliseconds, or
            if not specified, the number of frames in the whole AudioSegment
        """
        if ms is not None:
            return ms * (self.frame_rate / 1000.0)
        else:
            return float(len(self._data) // self.frame_width)

    def get_sample_slice(self, start_sample=None, end_sample=None):
        """
        Get a section of the audio segment by sample index.

        NOTE: Negative indices do *not* address samples backword
        from the end of the audio segment like a python list.
        This is intentional.
        """
        max_val = int(self.frame_count())

        def bounded(val, default):
            if val is None:
                return default
            if val < 0:
                return 0
            if val > max_val:
                return max_val
            return val

        start_i = bounded(start_sample, 0) * self.frame_width
        end_i = bounded(end_sample, max_val) * self.frame_width

        data = self._data[start_i:end_i]
        return self._spawn(data)

    def _get_segment(self, start: int, end: int) -> Self:
        start = self._ms_to_byte_offset(start)
        end = self._ms_to_byte_offset(end)
        data = self._data[start:end]

        # ensure the output is as long as the requester is expecting
        expected_length = end - start
        missing_frames = (expected_length - len(data)) // self.frame_width
        if missing_frames:
            if missing_frames > self.frame_count(ms=2):
                raise TooManyMissingFrames(
                    f"Missing frames exceed 2 ms of silence: {missing_frames}"
                )
            silence = audioop.mul(data[: self.frame_width], self.sample_width, 0)
            data += silence * missing_frames

        return self._spawn(data)

    def set_sample_width(self, sample_width):
        if sample_width == self.sample_width:
            return self

        return self._spawn(
            audioop.lin2lin(self._data, self.sample_width, sample_width),
            sample_width=sample_width,
        )

    def set_frame_rate(self, frame_rate):
        if frame_rate == self.frame_rate:
            return self

        if self._data:
            converted, _ = audioop.ratecv(
                self._data, self.sample_width, self.channels, self.frame_rate, frame_rate, None
            )
        else:
            converted = self._data

        return self._spawn(data=converted, frame_rate=frame_rate)

    def set_channels(self, channels):
        if channels == self.channels:
            return self

        if channels == 2 and self.channels == 1:
            converted = audioop.tostereo(self._data, self.sample_width, 1, 1)
        elif channels == 1 and self.channels == 2:
            converted = audioop.tomono(self._data, self.sample_width, 0.5, 0.5)
        elif channels == 1:
            channels_data = [seg.get_array_of_samples() for seg in self.split_to_mono()]
            frame_count = int(self.frame_count())
            converted = array.array(
                channels_data[0].typecode, b"\0" * (frame_count * self.sample_width)
            )
            for raw_channel_data in channels_data:
                for i in range(frame_count):
                    converted[i] += raw_channel_data[i] // self.channels
        elif self.channels == 1:
            dup_channels = [self for iChannel in range(channels)]
            return AudioSegment.from_mono_audiosegments(*dup_channels)
        else:
            raise ValueError(
                "'set_channels' only supports mono-to-multi channel "
                "and multi-to-mono channel conversion"
            )

        return self._spawn(data=converted, channels=channels)

    def split_to_mono(self):
        if self.channels == 1:
            return [self]

        samples = self.get_array_of_samples()

        mono_channels = []
        for i in range(self.channels):
            samples_for_current_channel = samples[i :: self.channels]
            mono_data = samples_for_current_channel.tobytes()
            mono_channels.append(self._spawn(mono_data, channels=1))

        return mono_channels

    def apply_gain(self, volume_change):
        return self._spawn(
            data=audioop.mul(self._data, self.sample_width, db_to_float(float(volume_change)))
        )

    def overlay(
        self,
        seg: Self,
        position: int = 0,
        loop: bool = False,
        times: int | None = None,
        gain_during_overlay: int | None = None,
    ) -> Self:
        if loop:
            times = -1
        elif times is None:
            times = 1
        elif times == 0:
            return self._spawn(self._data)

        seg1, seg2 = AudioSegment._sync(self, seg)
        position_in_bytes = self._ms_to_byte_offset(ms=position, frame_width=seg1.frame_width)
        result = _pydub_core.overlay_segments(
            seg1_data=seg1.raw_data,
            seg2_data=seg2.raw_data,
            sample_width=seg1.sample_width,
            position=position_in_bytes,
            times=times,
            gain_during_overlay=gain_during_overlay or 0,
        )

        return seg1._spawn(data=result)

    def append(self, seg, crossfade=100):
        seg1, seg2 = AudioSegment._sync(self, seg)

        if not crossfade:
            return seg1._spawn(seg1.raw_data + seg2.raw_data)
        elif crossfade > len(self):
            raise ValueError(
                f"Crossfade is longer than the original AudioSegment ({crossfade}ms > {len(self)}ms)"
            )
        elif crossfade > len(seg):
            raise ValueError(
                f"Crossfade is longer than the appended AudioSegment ({crossfade}ms > {len(seg)}ms)"
            )

        xf = seg1[-crossfade:].fade(to_gain=-120, start=0, end=float("inf"))
        xf *= seg2[:crossfade].fade(from_gain=-120, start=0, end=float("inf"))

        output = io.BytesIO()

        output.write(seg1[:-crossfade].raw_data)
        output.write(xf.raw_data)
        output.write(seg2[crossfade:].raw_data)

        output.seek(0)
        obj = seg1._spawn(data=output)
        output.close()
        return obj

    def fade(
        self,
        to_gain: float = 0,
        from_gain: float = 0,
        start: int | None = None,
        end: int | None = None,
        duration: int = None,
    ) -> Self:
        """
        Fade the volume of this audio segment.

        to_gain (float):
            resulting volume_change in db

        start (int):
            default = beginning of the segment
            when in this segment to start fading in milliseconds

        end (int):
            default = end of the segment
            when in this segment to start fading in milliseconds

        duration (int):
            default = until the end of the audio segment
            the duration of the fade
        """
        if None not in [duration, end, start]:
            raise TypeError(
                "Only two of the three arguments, 'start', 'end', and 'duration' may be specified"
            )

        # no fade == the same audio
        if to_gain == 0 and from_gain == 0:
            return self

        start = min(len(self), start) if start is not None else None
        end = min(len(self), end) if end is not None else None

        if start is not None and start < 0:
            start += len(self)
        if end is not None and end < 0:
            end += len(self)

        if duration is not None and duration < 0:
            raise InvalidDuration("Duration must be a positive integer")

        if duration:
            if start is not None:
                end = start + duration
            elif end is not None:
                start = end - duration
        else:
            duration = end - start

        start_bytes = self._ms_to_byte_offset(start)
        end_bytes = self._ms_to_byte_offset(end)

        result = _pydub_core.fade_segment(
            data=bytes(self._data),
            sample_width=self.sample_width,
            start_byte=start_bytes,
            end_byte=end_bytes,
            from_power=db_to_float(from_gain),
            to_power=db_to_float(to_gain),
        )

        return self._spawn(data=result)

    def fade_out(self, duration: int) -> Self:
        return self.fade(to_gain=-120, duration=duration, end=len(self))

    def fade_in(self, duration: int) -> Self:
        return self.fade(from_gain=-120, duration=duration, start=0)

    def reverse(self):
        return self._spawn(data=audioop.reverse(self._data, self.sample_width))

    def get_dc_offset(self, channel=1):
        """
        Returns a value between -1.0 and 1.0 representing the DC offset of a
        channel (1 for left, 2 for right).
        """
        if not 1 <= channel <= 2:
            raise ValueError("'channel' must be 1 (left) or 2 (right)")

        if self.channels == 1:
            data = self._data
        elif channel == 1:
            data = audioop.tomono(self._data, self.sample_width, 1, 0)
        else:
            data = audioop.tomono(self._data, self.sample_width, 0, 1)

        return float(audioop.avg(data, self.sample_width)) / self.max_possible_amplitude

    def remove_dc_offset(self, channel=None, offset=None):
        """
        Removes DC offset of given channel. Calculates offset if it's not given.
        Offset values must be in range -1.0 to 1.0. If channel is None, removes
        DC offset from all available channels.
        """
        if channel and not 1 <= channel <= 2:
            raise ValueError("'channel' must be None, 1 (left) or 2 (right)")

        if offset and not -1.0 <= offset <= 1.0:
            raise ValueError("'offset' must be in range -1.0 to 1.0")

        if offset:
            offset = int(round(offset * self.max_possible_amplitude))

        def remove_data_dc(data, off):
            if not off:
                off = audioop.avg(data, self.sample_width)
            return audioop.bias(data, self.sample_width, -off)

        if self.channels == 1:
            return self._spawn(data=remove_data_dc(self._data, offset))

        left_channel = audioop.tomono(self._data, self.sample_width, 1, 0)
        right_channel = audioop.tomono(self._data, self.sample_width, 0, 1)

        if not channel or channel == 1:
            left_channel = remove_data_dc(left_channel, offset)

        if not channel or channel == 2:
            right_channel = remove_data_dc(right_channel, offset)

        left_channel = audioop.tostereo(left_channel, self.sample_width, 1, 0)
        right_channel = audioop.tostereo(right_channel, self.sample_width, 0, 1)

        return self._spawn(data=audioop.add(left_channel, right_channel, self.sample_width))

    def export(
        self,
        out_f: str | os.PathLike | None = None,
        format: str = "mp3",
        codec: str | None = None,
        bitrate: str | None = None,
        parameters: list[str] | None = None,
        tags: dict[str, str] | None = None,
        id3v2_version: str = "4",
        cover: str | None = None,
        compressor: _compression.Compressor | None = None,
    ) -> IO[bytes]:
        result = self._export(
            out_f=out_f,
            format=format,
            codec=codec,
            bitrate=bitrate,
            parameters=parameters,
            tags=tags,
            id3v2_version=id3v2_version,
            cover=cover,
        )
        if compressor is not None:
            result = io.BytesIO(_compression.compress(compressor=compressor, content=result.read()))
        return result

    def _export(
        self,
        out_f=None,
        format="mp3",
        codec=None,
        bitrate=None,
        parameters=None,
        tags=None,
        id3v2_version="4",
        cover=None,
    ):
        """
        Export an AudioSegment to a file with given options

        out_f (string):
            Path to destination audio file. Also accepts os.PathLike objects on
            python >= 3.6

        format (string)
            Format for destination audio file.
            ('mp3', 'wav', 'raw', 'ogg' or other ffmpeg/avconv supported files)

        codec (string)
            Codec used to encode the destination file.

        bitrate (string)
            Bitrate used when encoding destination file. (64, 92, 128, 256, 312k...)
            Each codec accepts different bitrate arguments so take a look at the
            ffmpeg documentation for details (bitrate usually shown as -b, -ba or
            -a:b).

        parameters (list of strings)
            Aditional ffmpeg/avconv parameters

        tags (dict)
            Set metadata information to destination files
            usually used as tags. ({title='Song Title', artist='Song Artist'})

        id3v2_version (string)
            Set ID3v2 version for tags. (default: '4')

        cover (file)
            Set cover for audio file from image file. (png or jpg)
        """
        if format == "raw" and (codec is not None or parameters is not None):
            raise AttributeError(
                "Cannot invoke ffmpeg when export format is 'raw'; "
                "specify an ffmpeg raw format like format='s16le' instead "
                "or call export(format='raw') with no codec or parameters"
            )

        if out_f is None:
            out_f = TemporaryFile(mode="wb+")
        elif isinstance(out_f, (str, os.PathLike)):
            out_f = open(out_f, mode="wb+")
        out_f.seek(0)

        if format == "raw":
            return self._export_raw(out_f)

        if format == "wav" and codec is None and parameters is None:
            return self._export_wav(out_f)

        return self._export_via_ffmpeg(
            out_f,
            format=format,
            codec=codec,
            bitrate=bitrate,
            parameters=parameters,
            tags=tags,
            id3v2_version=id3v2_version,
            cover=cover,
        )

    def _export_raw(self, out_f: IO[bytes]) -> IO[bytes]:
        out_f.write(self._data)
        out_f.seek(0)
        return out_f

    def _write_wav(self, out_f: IO[bytes]) -> None:
        pcm_for_wav = self._data
        if self.sample_width == SampleWidth.PCM8:
            pcm_for_wav = audioop.bias(self._data, 1, 128)

        wave_data = wave.open(out_f, "wb")
        wave_data.setnchannels(self.channels)
        wave_data.setsampwidth(self.sample_width)
        wave_data.setframerate(self.frame_rate)
        wave_data.setnframes(int(self.frame_count()))
        wave_data.writeframesraw(pcm_for_wav)
        wave_data.close()

    def _export_wav(self, out_f: IO[bytes]) -> IO[bytes]:
        self._write_wav(out_f)
        out_f.seek(0)
        return out_f

    def _export_via_ffmpeg(
        self,
        out_f: IO[bytes],
        *,
        format: str,
        codec: str | None,
        bitrate: str | None,
        parameters: list[str] | None,
        tags: dict[str, str] | None,
        id3v2_version: str,
        cover: str | None,
    ) -> IO[bytes]:
        with _ffmpeg_tmp_files(self) as (data, output):
            conversion_command = _ConversionCommand.init(self.converter)
            conversion_command = conversion_command.with_format("wav")
            conversion_command = conversion_command.with_filename(data.name)

            if cover is not None:
                conversion_command = conversion_command.with_cover(cover, format)

            if codec is None:
                codec = self.DEFAULT_CODECS.get(format, None)
            if codec is not None:
                conversion_command = conversion_command.with_codec(codec)

            if bitrate is not None:
                conversion_command = conversion_command.with_bitrate(bitrate)

            if parameters is not None:
                conversion_command = conversion_command.with_parameters(parameters)

            if tags is not None:
                conversion_command = conversion_command.with_tags(tags, format, id3v2_version)

            if sys.platform == "darwin" and codec == "mp3":
                conversion_command = conversion_command.with_parameters(["-write_xing", "0"])

            conversion_command = conversion_command.with_output(format, output.name)

            log_conversion(conversion_command)

            with open(os.devnull, "rb") as devnull:
                p = subprocess.Popen(
                    conversion_command,
                    stdin=devnull,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            p_out, p_err = p.communicate()

            log_subprocess_output(p_out)
            log_subprocess_output(p_err)

            if p.returncode != 0:
                raise CouldntEncodeError(
                    f"Encoding failed. {self.converter} returned error code: "
                    f"{p.returncode}\n\nCommand: {conversion_command}\n\n"
                    f"Output from {self.converter}:\n\n{p_err.decode(errors='ignore')}"
                )

            output.seek(0)
            out_f.write(output.read())

        out_f.seek(0)
        return out_f

    def measure_audio_level(
        self, *names: Unpack[tuple[Literal["rms", "peak", "loudness"], ...]]
    ) -> _meter.AudioLevel:
        meter_to_measurer = {
            "rms": _meter.measure_rms,
            "peak": _meter.measure_peak,
            "loudness": _meter.measure_loudness,
        }
        return {name: meter_to_measurer[name](self) for name in names}

    def get_normalized_amplitudes(self, num_segments: int) -> list[float]:
        if not self.rms:
            raise ValueError("Audio contains no audio data")

        segment_duration = len(self) / num_segments
        amplitudes = [
            self[(i * segment_duration) : ((i + 1) * segment_duration)].rms
            for i in range(num_segments)
        ]
        max_amplitude = max(amplitudes)
        return [(amplitude / max_amplitude) for amplitude in amplitudes]

    def _spawn(
        self,
        data: bytes | list[bytes] | array.array | IO[bytes],
        *,
        sample_width: int | None = None,
        frame_rate: int | None = None,
        channels: int | None = None,
    ) -> Self:
        """
        Creates a new audio segment using the metadata from the current one
        and the data passed in. Should be used whenever an AudioSegment is
        being returned by an operation that would alters the current one,
        since AudioSegment objects are immutable.
        """
        match data:
            case list():
                data = b"".join(data)
            case array.array():
                data = data.tobytes()
            case io.IOBase():
                data.seek(0)
                data = data.read()

        return self.__class__(
            data=data,
            sample_width=sample_width or self.sample_width,
            frame_rate=frame_rate or self.frame_rate,
            channels=channels or self.channels,
        )

    @classmethod
    def _sync(cls, *segs: Self) -> tuple[Self, ...]:
        channels = max(seg.channels for seg in segs)
        frame_rate = max(seg.frame_rate for seg in segs)
        sample_width = max(seg.sample_width for seg in segs)

        return tuple(
            seg.set_channels(channels).set_frame_rate(frame_rate).set_sample_width(sample_width)
            for seg in segs
        )

    def _ms_to_byte_offset(self, ms: float, frame_width: int | None = None) -> int:
        if ms < 0:
            ms = len(self) + ms
        if ms == float("inf"):
            ms = len(self)
        return int(self.frame_count(ms=ms)) * (frame_width or self.frame_width)
