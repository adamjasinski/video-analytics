import pytest
import io
import math
import src.video_helper as vh

@pytest.mark.parametrize(
    "filename,expected_w,expected_h,expected_codec,expected_frames,expected_duration_secs",
    [pytest.param("./test/data/cars-downsampled.mp4", 320, 240, 'h264', 301, 10),
     pytest.param("./test/data/cars-tracking-output.gif", 640, 480, 'gif', 301, 9)])
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
def test_transcode_to_h264(filename):
    converted = None
    input_video_properties = None
    converted_video_properties = None

    with io.open(filename, 'rb') as f:
        byte_array = f.read()
        input_video_properties = vh.get_video_properties(byte_array)
        converted = vh.transcode_to_h264(byte_array)
        converted_video_properties = vh.get_video_properties(converted)

    assert converted is not None
    assert converted_video_properties["width"] == input_video_properties["width"] 
    assert converted_video_properties["height"] == input_video_properties["height"] 
    assert converted_video_properties["codec"] == "h264"
    # NB - compare framecount within a certain tolerance threshold (transcoding may result in some frames being dropped)
    assert abs(converted_video_properties["frames"] - input_video_properties["frames"]) < (converted_video_properties["frames"]/100)
