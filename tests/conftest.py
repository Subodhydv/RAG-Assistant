import pytest

from app.config import Settings, settings as global_settings


@pytest.fixture(autouse=True)
def _reset_settings():
    """chunking/config tests mutate the shared settings singleton for
    convenience; reset it after every test so ordering never matters."""
    defaults = Settings()
    yield
    global_settings.chunk_char_len = defaults.chunk_char_len
    global_settings.chunk_overlap = defaults.chunk_overlap
