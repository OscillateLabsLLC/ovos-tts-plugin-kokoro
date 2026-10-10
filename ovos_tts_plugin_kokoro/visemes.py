"""Turn Kokoro's per-token duration predictions into OVOS phoneme timing.

Kokoro's model predicts a duration for every character of its phoneme
string, plus a leading ``<bos>`` and trailing ``<eos>`` token, measured in
600-sample frames at 24 kHz. That gives phoneme-level timing for free, which
is far more accurate than the constant-duration guess the OVOS G2P fallback
produces when a TTS plugin returns no phonemes.

The output is the format ``ovos_plugin_manager.templates.tts.TTS.viseme``
parses: space-separated ``phoneme:end_time`` pairs, using lowercase ARPAbet
keys that match ``ovos_utils.lang.visimes.VISIMES``, with end times
cumulative from the start of the audio in seconds. Enclosures such as the
Mark 1 consume the pairs as "hold this mouth shape until ``end_time``".
"""

from typing import Iterable, List, Optional, Sequence, Tuple


FRAMES_PER_SECOND = 40  # 600 samples per frame at Kokoro's native 24 kHz
PAUSE = "pau"
UNKNOWN_SOUND = "ah"

# Characters that modify the preceding phoneme rather than being one:
# stress, length, nasalisation, aspiration, palatalisation, tone arrows.
MODIFIERS = frozenset("ˈˌː̃ʰʲᵝ→↓↗↘")

# Kokoro phoneme alphabet (misaki IPA plus its single-letter diphthongs)
# -> lowercase ARPAbet key in VISIMES. Non-English sounds map to the
# nearest English mouth shape, which is all the viseme map distinguishes.
_KOKORO_TO_ARPABET = {
    # silence and punctuation
    " ": PAUSE, "!": PAUSE, '"': PAUSE, "(": PAUSE, ")": PAUSE, ",": PAUSE,
    ".": PAUSE, ":": PAUSE, ";": PAUSE, "?": PAUSE, "—": PAUSE, "“": PAUSE,
    "”": PAUSE, "…": PAUSE,
    # misaki diphthong shorthands
    "A": "ey", "I": "ay", "O": "ow", "Q": "ow", "W": "aw", "Y": "oy",
    # vowels
    "a": "ae", "e": "ey", "i": "iy", "o": "ow", "u": "uw", "y": "uw",
    "æ": "ae", "ɐ": "ah", "ɑ": "aa", "ɒ": "ao", "ɔ": "ao", "ə": "ah",
    "ɚ": "er", "ɛ": "eh", "ɜ": "er", "ɨ": "ih", "ɪ": "ih", "ɯ": "uw",
    "ɤ": "ah", "ʊ": "uh", "ʌ": "ah", "ø": "ow", "œ": "ow", "ᵊ": "ah",
    "ᵻ": "ih",
    # consonants
    "b": "b", "c": "k", "d": "d", "f": "f", "h": "hh", "j": "y", "k": "k",
    "l": "l", "m": "m", "n": "n", "p": "p", "q": "k", "r": "r", "s": "s",
    "t": "t", "v": "v", "w": "w", "x": "k", "z": "z", "S": "sh", "T": "t",
    "ç": "hh", "ð": "dh", "ŋ": "ng", "ɕ": "sh", "ɖ": "d", "ɟ": "jh",
    "ɡ": "g", "ɣ": "g", "ɥ": "w", "ɲ": "n", "ɳ": "n", "ɴ": "ng", "ɸ": "b",
    "ɹ": "r", "ɻ": "r", "ɽ": "d", "ɾ": "d", "ʁ": "r", "ʂ": "sh", "ʃ": "sh",
    "ʈ": "t", "ʋ": "v", "ʎ": "l", "ʒ": "zh", "ʔ": "hh", "ʝ": "y", "ʣ": "z",
    "ʤ": "jh", "ʥ": "jh", "ʦ": "s", "ʧ": "ch", "ʨ": "ch", "β": "b",
    "θ": "th", "χ": "k", "ꭧ": "ch",
}

TimedPhoneme = Tuple[str, float]


def phoneme_timing(chunks: Iterable[Tuple[str, Optional[Sequence[int]]]]) -> Optional[str]:
    """Build the OVOS ``phoneme:end_time`` string from Kokoro results.

    Args:
        chunks: ``(phonemes, pred_dur)`` per Kokoro result, in playback
            order. ``pred_dur`` must have one entry per phoneme character
            plus two for ``<bos>``/``<eos>``.

    Returns:
        The timing string, or ``None`` when any chunk lacks usable timing
        so the caller can fall back to the G2P path.
    """
    entries: List[TimedPhoneme] = []
    clock = 0.0
    for phonemes, pred_dur in chunks:
        frames = _frames(phonemes, pred_dur)
        if frames is None:
            return None
        clock = _append_chunk(entries, phonemes, frames, clock)
    if not entries:
        return None
    return " ".join(f"{pho}:{end:.3f}" for pho, end in entries)


def _frames(phonemes: str, pred_dur: Optional[Sequence[int]]) -> Optional[List[int]]:
    if pred_dur is None:
        return None
    try:
        frames = [int(f) for f in pred_dur]
    except TypeError:  # 0-d tensor: nothing to align
        return None
    if len(frames) != len(phonemes) + 2:
        return None
    return frames


def _append_chunk(entries: List[TimedPhoneme], phonemes: str, frames: List[int], clock: float) -> float:
    labels = [PAUSE, *(_label(c) for c in phonemes), PAUSE]
    for label, dur in zip(labels, frames):
        clock += dur / FRAMES_PER_SECOND
        extends_previous = (label is None or label == entries[-1][0]) if entries else False
        if extends_previous:
            entries[-1] = (entries[-1][0], clock)
        else:
            entries.append((label or PAUSE, clock))
    return clock


def _label(char: str) -> Optional[str]:
    """ARPAbet key for a Kokoro phoneme char; None for a modifier."""
    if char in MODIFIERS:
        return None
    return _KOKORO_TO_ARPABET.get(char, UNKNOWN_SOUND)
