"""Test aiaia_classifier functions."""

from aiaia_classifier import app


def test_app():
    """Test app.main function."""
    assert app.main("ah ", 3) == "ah ah ah "
