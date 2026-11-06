import numpy as np
import pytest
from vision import MODES, process_image, ensure_rgb

@pytest.fixture
def image():
    arr = np.zeros((80, 80, 3), dtype=np.uint8)
    arr[20:60, 20:60] = [255, 255, 255]
    return arr

@pytest.mark.parametrize("mode", MODES)
def test_modes_return_rgb(image, mode):
    result = process_image(image, mode)
    assert result.shape == image.shape
    assert result.dtype == np.uint8

def test_original_is_copy(image):
    result = process_image(image, "Original")
    assert np.array_equal(result, image)
    assert result is not image

def test_even_kernel_rejected(image):
    with pytest.raises(ValueError):
        process_image(image, "Gaussian blur", blur_kernel=4)

def test_invalid_canny_rejected(image):
    with pytest.raises(ValueError):
        process_image(image, "Canny edges", canny_low=200, canny_high=50)

def test_bad_image_rejected():
    with pytest.raises(ValueError):
        ensure_rgb(np.zeros((10, 10), dtype=np.uint8))

def test_unknown_mode(image):
    with pytest.raises(ValueError):
        process_image(image, "unknown")
