import os
from PIL import Image
import io
import pytest
from unittest.mock import patch
from app import validate_image


# In both functions, checking that the image passes or fails validation as expected
# Mocking st.error to not crash the test
@patch("app.st.error")
def test_validate_img_good(mock_error):
    img = Image.new("RGB", (300, 300), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    result = validate_image(buf)
    assert result is not None, "Image should be valid"
    assert result.size == (300, 300)
    mock_error.assert_not_called()


@patch("app.st.error")
def test_validate_img_small(mock_error):
    img = Image.new("RGB", (10, 10))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    result = validate_image(buf)
    assert result is None
    mock_error.assert_called_once_with(
        ("Image dimensions must be between 50×50 and 5000×5000.")
    )
