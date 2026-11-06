"""Decode only bounded, complete JPEG/PNG images before inference."""

import io
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError

from .config import Settings


class InvalidImage(ValueError):
    pass


def read_image(uploaded_file, settings: Settings) -> Image.Image:
    uploaded_file.seek(0)
    contents = uploaded_file.read(settings.max_upload_bytes + 1)
    if not contents or len(contents) > settings.max_upload_bytes:
        raise InvalidImage("Upload a JPEG or PNG image of at most 5 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(
                io.BytesIO(contents), formats=("JPEG", "PNG")
            ) as image:
                width, height = image.size
                # Check headers before allocating the decoded bitmap.
                if (
                    min(width, height) < 50
                    or max(width, height) > settings.max_image_dimension
                    or width * height > settings.max_image_pixels
                ):
                    raise InvalidImage(
                        "Use an image between 50 and 4096 pixels per side, "
                        "at most 12 megapixels."
                    )
                if getattr(image, "n_frames", 1) != 1:
                    raise InvalidImage("Animated images are not supported.")
                image.verify()
            # Reopen after verify() and fully decode inside the try
            # block.
            with Image.open(
                io.BytesIO(contents), formats=("JPEG", "PNG")
            ) as image:
                image.load()
                normalized = ImageOps.exif_transpose(image).convert("RGB")
                return Image.frombytes(
                    "RGB", normalized.size, normalized.tobytes()
                )
    except InvalidImage:
        raise
    except (
        OSError,
        ValueError,
        SyntaxError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as error:
        raise InvalidImage(
            "The file is not a complete, valid JPEG or PNG image."
        ) from error
