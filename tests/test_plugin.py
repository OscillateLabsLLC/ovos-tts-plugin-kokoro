"""Basic POC tests for ovos-tts-plugin-kokoro."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _reset_caches():
    from ovos_tts_plugin_kokoro import tts as tts_mod
    tts_mod._pipelines.clear()


@pytest.fixture(autouse=True)
def _clear_caches():
    _reset_caches()
    yield
    _reset_caches()


def test_import_fast():
    """Plugin must import quickly — no kokoro/torch at module load time."""
    import importlib
    import time

    start = time.monotonic()
    importlib.import_module("ovos_tts_plugin_kokoro")
    elapsed = time.monotonic() - start
    assert elapsed < 1.0, f"Import took {elapsed:.2f}s — must be < 1.0s for opm-check"


def test_available_languages():
    from ovos_tts_plugin_kokoro import KokoroTTSPlugin

    # available_languages is a classproperty (matching the OVOS TTS base
    # class), so it is accessed as an attribute, not called.
    langs = KokoroTTSPlugin.available_languages
    assert {"en", "es", "fr", "ja", "zh"}.issubset(langs)


def test_config_dict_has_voices():
    from ovos_tts_plugin_kokoro import KokoroTTSPluginConfig

    assert "en" in KokoroTTSPluginConfig
    en_names = {v["voice"] for v in KokoroTTSPluginConfig["en"]}
    assert "af_bella" in en_names
    assert "bm_george" in en_names


def test_resolve_lang_routes_british_english():
    from ovos_tts_plugin_kokoro import tts as tts_mod

    m = tts_mod._DEFAULT_LANG_MAP
    assert tts_mod._resolve_lang("en-GB", m) == "b"
    assert tts_mod._resolve_lang("en-US", m) == "a"
    # Unknown falls back to American English
    assert tts_mod._resolve_lang("ko-KR", m) == "a"


def _fake_pipeline(audio_chunks):
    """Mock KPipeline that yields (graphemes, phonemes, audio) tuples."""
    pipeline = MagicMock()
    def call(text, voice=None, speed=1.0):
        for chunk in audio_chunks:
            yield ("g", "p", chunk)
    pipeline.side_effect = call
    return pipeline


@patch("ovos_tts_plugin_kokoro.tts._get_pipeline")
def test_get_tts_writes_valid_wav(mock_get_pipeline, tmp_path):
    import wave

    sample_rate = 24000
    fake_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, sample_rate, dtype=np.float32))
    mock_get_pipeline.return_value = _fake_pipeline([fake_audio])

    from ovos_tts_plugin_kokoro import KokoroTTSPlugin

    # Pin to native rate to skip the resample path
    plug = KokoroTTSPlugin(config={"lang": "en", "voice": "af_bella", "sample_rate": sample_rate})
    wav_file = str(tmp_path / "test.wav")
    result, phonemes = plug.get_tts("Hello world", wav_file)

    assert result == wav_file
    assert phonemes is None
    with wave.open(wav_file, "rb") as wf:
        assert wf.getsampwidth() == 2
        assert wf.getnchannels() == 1
        assert wf.getframerate() == sample_rate


@patch("ovos_tts_plugin_kokoro.tts._get_pipeline")
def test_get_tts_voice_overrides_lang(mock_get_pipeline, tmp_path):
    """bm_george must route through the British pipeline even when lang=en-US."""
    fake_audio = np.zeros(24000, dtype=np.float32)
    mock_get_pipeline.return_value = _fake_pipeline([fake_audio])

    from ovos_tts_plugin_kokoro import KokoroTTSPlugin

    plug = KokoroTTSPlugin(config={"lang": "en-US", "voice": "bm_george", "sample_rate": 24000})
    plug.get_tts("Hello", str(tmp_path / "george.wav"))

    args, _ = mock_get_pipeline.call_args
    assert args[0] == "b"


@patch("ovos_tts_plugin_kokoro.tts._get_pipeline")
def test_get_tts_default_voice_sentinel(mock_get_pipeline, tmp_path):
    """OVOS/Neon pass voice="default" when no voice is picked — must fall
    back to configured voice rather than asking Kokoro for a voice named
    "default" (which 404s on HF Hub)."""
    fake_audio = np.zeros(24000, dtype=np.float32)
    pipeline = _fake_pipeline([fake_audio])
    mock_get_pipeline.return_value = pipeline

    from ovos_tts_plugin_kokoro import KokoroTTSPlugin

    plug = KokoroTTSPlugin(config={"lang": "en-US", "voice": "af_bella", "sample_rate": 24000})
    plug.get_tts("Hi", str(tmp_path / "default.wav"), voice="default")

    _args, kwargs = pipeline.call_args
    assert kwargs.get("voice") == "af_bella"
