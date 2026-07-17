"""Kokoro TTS plugin for OVOS — 82M parameter multilingual TTS by hexgrad."""

import wave
from typing import Dict, Optional, Tuple

import numpy as np
from ovos_plugin_manager.templates.tts import TTS
from ovos_utils import classproperty
from ovos_utils.lang import standardize_lang_tag
from ovos_utils.log import LOG


# BCP-47 base subtag -> kokoro language code.
# Full BCP-47 tags can override via `language_aliases` config.
_DEFAULT_LANG_MAP: Dict[str, str] = {
    "en": "a",       # American English (default for plain "en")
    "en-us": "a",
    "en-gb": "b",    # British English
    "es": "e",       # Spanish
    "fr": "f",       # French
    "hi": "h",       # Hindi
    "it": "i",       # Italian
    "ja": "j",       # Japanese — needs misaki[ja]
    "pt": "p",       # Brazilian Portuguese
    "pt-br": "p",
    "zh": "z",       # Mandarin — needs misaki[zh]
}

# Voice catalogue — kept aligned with hexgrad/Kokoro-82M VOICES.md.
# Tuple = (voice_id, gender, kokoro_lang_code).
_VOICES: Tuple[Tuple[str, str, str], ...] = (
    # American English
    ("af_heart", "female", "a"),
    ("af_alloy", "female", "a"),
    ("af_aoede", "female", "a"),
    ("af_bella", "female", "a"),
    ("af_jessica", "female", "a"),
    ("af_kore", "female", "a"),
    ("af_nicole", "female", "a"),
    ("af_nova", "female", "a"),
    ("af_river", "female", "a"),
    ("af_sarah", "female", "a"),
    ("af_sky", "female", "a"),
    ("am_adam", "male", "a"),
    ("am_echo", "male", "a"),
    ("am_eric", "male", "a"),
    ("am_fenrir", "male", "a"),
    ("am_liam", "male", "a"),
    ("am_michael", "male", "a"),
    ("am_onyx", "male", "a"),
    ("am_puck", "male", "a"),
    ("am_santa", "male", "a"),
    # British English
    ("bf_alice", "female", "b"),
    ("bf_emma", "female", "b"),
    ("bf_isabella", "female", "b"),
    ("bf_lily", "female", "b"),
    ("bm_daniel", "male", "b"),
    ("bm_fable", "male", "b"),
    ("bm_george", "male", "b"),
    ("bm_lewis", "male", "b"),
    # Japanese
    ("jf_alpha", "female", "j"),
    ("jf_gongitsune", "female", "j"),
    ("jf_nezumi", "female", "j"),
    ("jf_tebukuro", "female", "j"),
    ("jm_kumo", "male", "j"),
    # Mandarin
    ("zf_xiaobei", "female", "z"),
    ("zf_xiaoni", "female", "z"),
    ("zf_xiaoxiao", "female", "z"),
    ("zf_xiaoyi", "female", "z"),
    ("zm_yunjian", "male", "z"),
    ("zm_yunxi", "male", "z"),
    ("zm_yunxia", "male", "z"),
    ("zm_yunyang", "male", "z"),
    # Spanish
    ("ef_dora", "female", "e"),
    ("em_alex", "male", "e"),
    ("em_santa", "male", "e"),
    # French
    ("ff_siwis", "female", "f"),
    # Hindi
    ("hf_alpha", "female", "h"),
    ("hf_beta", "female", "h"),
    ("hm_omega", "male", "h"),
    ("hm_psi", "male", "h"),
    # Italian
    ("if_sara", "female", "i"),
    ("im_nicola", "male", "i"),
    # Brazilian Portuguese
    ("pf_dora", "female", "p"),
    ("pm_alex", "male", "p"),
    ("pm_santa", "male", "p"),
)

# Reverse lookup: kokoro lang code -> set of BCP-47 base subtags it serves.
# Used by available_languages().
_KOKORO_TO_BCP47: Dict[str, str] = {
    "a": "en", "b": "en", "e": "es", "f": "fr",
    "h": "hi", "i": "it", "j": "ja", "p": "pt", "z": "zh",
}

DEFAULT_VOICE = "af_bella"
KOKORO_NATIVE_RATE = 24000

# Lazy-loaded pipeline cache, keyed by (kokoro_lang, device) — same model
# loaded on different devices is different state. Each KPipeline holds the
# model + a g2p stack for its language.
_pipelines: Dict[tuple, "object"] = {}


def _resolve_lang(lang: Optional[str], lang_map: Dict[str, str]) -> str:
    """Map a BCP-47 language tag to a kokoro single-letter language code.

    Lookup order:
      1. Full BCP-47 tag (e.g. ``en-gb``) — lets us route British English
         to Kokoro's "b" pipeline rather than the American "a".
      2. Base subtag (e.g. ``en``) — the common case.
      3. American English fallback.
    """
    if not lang:
        return lang_map.get("en", "a")
    canonical = standardize_lang_tag(lang).lower()
    if canonical in lang_map:
        return lang_map[canonical]
    base = canonical.split("-")[0]
    return lang_map.get(base, lang_map.get("en", "a"))


def _get_pipeline(kokoro_lang: str, device: Optional[str] = None):
    """Lazy-load and cache the Kokoro KPipeline for a given (lang, device).

    ``device`` accepts ``"cpu"``, ``"cuda"``, ``"mps"``, or ``None`` (let
    Kokoro auto-select). Note that Kokoro's auto-select prefers CUDA and
    falls back to CPU; it does not auto-promote to MPS on Apple Silicon,
    so pass ``"mps"`` explicitly to use the Metal backend.
    """
    key = (kokoro_lang, device)
    if key in _pipelines:
        return _pipelines[key]
    from kokoro import KPipeline

    LOG.info(
        "Loading Kokoro TTS pipeline (lang=%s, device=%s) — first call may download weights",
        kokoro_lang, device or "auto",
    )
    _pipelines[key] = KPipeline(lang_code=kokoro_lang, device=device)
    LOG.info("Kokoro TTS pipeline loaded for %s on %s", kokoro_lang, device or "auto")
    return _pipelines[key]


def _audio_to_int16(audio_np: np.ndarray) -> np.ndarray:
    """Clamp float audio to [-1, 1] and convert to int16 PCM.

    Kokoro typically returns values inside [-1, 1] but we clamp defensively;
    scipy.io.wavfile.write with float32 produces IEEE float WAV (format tag 3)
    which many OVOS playback paths handle poorly. int16 PCM via stdlib `wave`
    is the safe baseline.
    """
    return (np.clip(audio_np, -1, 1) * 32767).astype(np.int16)


def _resample(audio_np: np.ndarray, native_rate: int, target_rate: int) -> np.ndarray:
    """Resample audio if rates differ. OVOS does not resample TTS output."""
    if native_rate == target_rate:
        return audio_np
    from scipy.signal import resample

    num_samples = int(len(audio_np) * target_rate / native_rate)
    return resample(audio_np, num_samples).astype(np.float32)


def _audio_chunk_to_numpy(audio) -> np.ndarray:
    """Coerce a Kokoro audio chunk (torch.Tensor or numpy array) to float32 numpy."""
    if hasattr(audio, "detach"):
        return audio.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(audio, dtype=np.float32)


class KokoroTTSPlugin(TTS):
    """Kokoro TTS — 82M param multilingual TTS by hexgrad.

    Config example (mycroft.conf):
        "tts": {
            "module": "ovos-tts-plugin-kokoro",
            "ovos-tts-plugin-kokoro": {
                "voice": "af_bella",
                "speed": 1.0
            }
        }

    The "voice" field accepts any voice id from the Kokoro-82M voice list
    (e.g. ``af_bella``, ``bm_george``, ``jf_alpha``). The plugin will route
    inference through the matching language pipeline regardless of the
    OVOS active language.
    """

    def __init__(self, config=None):
        super().__init__(config=config, audio_ext="wav")
        for code in self.config.get("preload_languages", []):
            try:
                self._load_language(code)
            except Exception as err:
                LOG.warning("Failed to preload Kokoro language %s: %s", code, err)

    @property
    def lang_map(self) -> Dict[str, str]:
        """Effective BCP-47 -> kokoro language code map (default + user overrides)."""
        return {**_DEFAULT_LANG_MAP, **self.config.get("language_aliases", {})}

    def _voice_lang(self, voice: str) -> Optional[str]:
        """Return the kokoro language code baked into a voice id, if known.

        Voice ids follow ``<lang><gender>_<name>`` (e.g. ``af_bella``,
        ``bm_george``). For built-in voices we trust the prefix and skip
        the BCP-47 dance entirely — picking ``bm_george`` should always
        speak through the British pipeline regardless of the OVOS lang.
        """
        for vid, _gender, lang_code in _VOICES:
            if vid == voice:
                return lang_code
        return None

    @property
    def device(self) -> Optional[str]:
        """Torch device for inference.

        Defaults to ``"cpu"``. CPU is the fastest option for this model on
        Apple Silicon — MPS is ~4-5x slower for typical sentences because
        the vocoder leans heavily on ``torch.stft``/``istft``, which are
        weak spots on the Metal backend. Set ``device`` explicitly to
        ``"cuda"`` on machines with a discrete GPU, or to ``None`` to let
        Kokoro auto-select (prefers CUDA, falls back to CPU — never MPS).
        """
        return self.config.get("device", "cpu")

    def _load_language(self, lang_or_code: str):
        """Resolve a BCP-47 (or kokoro single-letter) code and warm the cache."""
        if len(lang_or_code) == 1 and lang_or_code in _KOKORO_TO_BCP47:
            kokoro_lang = lang_or_code
        else:
            kokoro_lang = _resolve_lang(lang_or_code, self.lang_map)
        return _get_pipeline(kokoro_lang, device=self.device)

    def get_tts(self, sentence: str, wav_file: str,
                lang: str = None, voice: str = None) -> tuple:
        """Synthesize ``sentence`` into ``wav_file`` and return (path, None)."""
        # OVOS/Neon pass voice="default" when the user hasn't picked one
        # explicitly. Kokoro has no voice called "default", so treat that
        # sentinel (and empty/None) as "fall back to configured voice".
        if not voice or voice == "default":
            voice = self.config.get("voice", DEFAULT_VOICE)
        speed = float(self.config.get("speed", 1.0))

        # Voice id wins over OVOS lang — picking bm_george shouldn't render
        # through the American pipeline just because self.lang is en-US.
        kokoro_lang = self._voice_lang(voice) or _resolve_lang(lang or self.lang, self.lang_map)
        pipeline = _get_pipeline(kokoro_lang, device=self.device)

        chunks = []
        for _graphemes, _phonemes, audio in pipeline(sentence, voice=voice, speed=speed):
            chunks.append(_audio_chunk_to_numpy(audio))

        if not chunks:
            LOG.warning("Kokoro returned no audio for sentence: %r", sentence)
            full_audio = np.zeros(0, dtype=np.float32)
        else:
            full_audio = np.concatenate(chunks)

        target_rate = int(self.config.get("sample_rate", 16000))
        full_audio = _resample(full_audio, KOKORO_NATIVE_RATE, target_rate)
        audio_int16 = _audio_to_int16(full_audio)

        with wave.open(wav_file, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(target_rate)
            wf.writeframes(audio_int16.tobytes())

        return wav_file, None

    def shutdown(self):
        """Release cached pipelines to free memory."""
        if _pipelines:
            LOG.info("Shutting down Kokoro TTS — releasing %d pipeline(s)", len(_pipelines))
            _pipelines.clear()

    @classproperty
    def available_languages(cls) -> set:
        """Return the BCP-47 base language codes Kokoro can serve."""
        return set(_KOKORO_TO_BCP47.values())


def _build_plugin_config():
    """Build the OPM KokoroTTSPluginConfig advertisement.

    Emits one entry per voice, grouped under the BCP-47 language code that
    matches the voice's kokoro language. American and British English voices
    both surface under ``en`` so OVOS lang=en setups see the full catalogue.
    """
    config: Dict[str, list] = {}
    for voice_id, gender, kokoro_lang in _VOICES:
        bcp47 = _KOKORO_TO_BCP47[kokoro_lang]
        config.setdefault(bcp47, []).append({
            "voice": voice_id,
            "lang": bcp47,
            "meta": {
                "gender": gender,
                "display_name": f"Kokoro — {voice_id}",
                "offline": True,
                "priority": 60,
            },
        })
    return config


KokoroTTSPluginConfig = _build_plugin_config()
