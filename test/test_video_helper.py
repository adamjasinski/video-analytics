import pytest
import io
import math
import src.video_helper as vh

@pytest.mark.parametrize(
    "filename,expected_w,expected_h,expected_codec,expected_frames,expected_duration_secs",
    [pytest.param("./test/data/cars-downsampled.mp4", 320, 240, 'h264', 301, 9)])
def test_video_properties(filename, expected_w, expected_h, expected_codec, expected_frames, expected_duration_secs):
     with io.open(filename, 'rb') as f:
        byte_array = f.read()
        props = vh.get_video_properties(byte_array)
        assert props["width"] == expected_w
        assert props["height"] == expected_h
        assert props["codec"] == expected_codec
        assert props["frames"] == expected_frames
        assert math.floor(props["duration"]) == expected_duration_secs


@pytest.mark.parametrize(
    "filename",
    [pytest.param("./test/data/cars-downsampled.mp4")])
def test_video_conversion_returns_black_and_white(filename):
    with io.open(filename, 'rb') as f:
        byte_array = f.read()
        converted = vh.convert_to_bw(byte_array)
        assert converted is not None
        #TODO - verify colorspace

@pytest.mark.parametrize(
    "filename",
    [pytest.param("./test/data/cars-downsampled.mp4")])
def test_video_conversion_transcode(filename):
    with io.open(filename, 'rb') as f:
        byte_array = f.read()
        converted = vh.encode_bis(byte_array)
        assert converted is not None
