import io
from unittest.mock import patch

import pytest
from PIL import Image

from fake_or_proof.images import InvalidImage, read_image


def encoded_image(mode="RGB", size=(300, 300), format="PNG"):
    buffer = io.BytesIO()
    Image.new(mode, size).save(buffer, format=format)
    return buffer.getvalue()


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA"])
def test_valid_images_are_fully_decoded_rgb_without_metadata(settings, mode):
    result = read_image(io.BytesIO(encoded_image(mode)), settings)
    result.load()
    assert result.mode == "RGB" and result.size == (300, 300)
    assert not result.info


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA"])
def test_truncated_images_are_rejected(settings, mode):
    raw = encoded_image(mode)
    with pytest.raises(InvalidImage):
        read_image(io.BytesIO(raw[: len(raw) // 2]), settings)


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"not an image",
        encoded_image(size=(10, 10)),
        encoded_image(format="GIF"),
    ],
)
def test_invalid_small_or_unsupported_files_are_rejected(settings, raw):
    with pytest.raises(InvalidImage):
        read_image(io.BytesIO(raw), settings)


def test_upload_size_is_checked_before_image_open(settings):
    with patch("fake_or_proof.images.Image.open") as open_image:
        with pytest.raises(InvalidImage):
            read_image(
                io.BytesIO(b"x" * (settings.max_upload_bytes + 1)), settings
            )
        open_image.assert_not_called()


def test_dimensions_are_checked_before_decode(settings):
    with patch("fake_or_proof.images.Image.open") as open_image:
        image = open_image.return_value.__enter__.return_value
        image.size = (5000, 300)
        with pytest.raises(InvalidImage):
            read_image(io.BytesIO(b"header"), settings)
        image.load.assert_not_called()
        image.convert.assert_not_called()


def test_animated_png_is_rejected(settings):
    buffer = io.BytesIO()
    Image.new("RGB", (100, 100)).save(
        buffer,
        format="PNG",
        save_all=True,
        append_images=[Image.new("RGB", (100, 100), "white")],
    )
    with pytest.raises(InvalidImage, match="Animated"):
        read_image(buffer, settings)
