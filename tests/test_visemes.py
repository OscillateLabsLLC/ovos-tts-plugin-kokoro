"""Tests for Kokoro duration -> OVOS phoneme timing conversion."""

import pytest

from ovos_tts_plugin_kokoro.visemes import (
    _KOKORO_TO_ARPABET,
    FRAMES_PER_SECOND,
    MODIFIERS,
    phoneme_timing,
)


def _pairs(timing: str):
    return [(p.split(":")[0], float(p.split(":")[1])) for p in timing.split(" ")]


def test_frames_are_25ms():
    assert 1 / FRAMES_PER_SECOND == 0.025


def test_bos_becomes_leading_pause():
    """Kokoro's <bos> frames are real leading silence in the audio; the
    mouth must stay neutral for them or lips lead the voice."""
    timing = phoneme_timing([("h", [18, 2, 1])])
    assert _pairs(timing) == [("pau", 0.45), ("hh", 0.5), ("pau", 0.525)]


def test_end_times_are_cumulative_and_match_audio_length():
    phonemes = "həlˈO wˈɜɹld."
    pred_dur = [18, 2, 2, 2, 2, 2, 2, 1, 2, 3, 4, 3, 14, 8, 1]
    pairs = _pairs(phoneme_timing([(phonemes, pred_dur)]))

    ends = [end for _pho, end in pairs]
    assert ends == sorted(ends)
    assert ends[-1] == pytest.approx(sum(pred_dur) / FRAMES_PER_SECOND)


def test_modifiers_extend_previous_phoneme_instead_of_emitting():
    pairs = _pairs(phoneme_timing([("lˈO", [4, 2, 2, 2, 1])]))
    assert [pho for pho, _end in pairs] == ["pau", "l", "ow", "pau"]
    # 'l' absorbs the stress mark's 2 frames: 0.1 + 0.05 + 0.05
    assert dict(pairs)["l"] == pytest.approx(0.2)


def test_adjacent_identical_labels_merge():
    pairs = _pairs(phoneme_timing([("a. ", [4, 2, 3, 2, 1])]))
    assert [pho for pho, _end in pairs] == ["pau", "ae", "pau"]
    assert pairs[-1][1] == pytest.approx(12 / FRAMES_PER_SECOND)


def test_chunks_continue_the_clock():
    first = ("a", [4, 2, 1])
    second = ("i", [4, 2, 1])
    pairs = _pairs(phoneme_timing([first, second]))
    assert pairs[-1][1] == pytest.approx(14 / FRAMES_PER_SECOND)
    assert [pho for pho, _end in pairs] == ["pau", "ae", "pau", "iy", "pau"]


def test_unknown_char_maps_to_open_mouth():
    pairs = _pairs(phoneme_timing([("ʘ", [1, 2, 1])]))
    assert pairs[1][0] == "ah"


@pytest.mark.parametrize("pred_dur", [None, [1, 2], [1, 2, 3, 4]])
def test_bad_timing_returns_none(pred_dur):
    assert phoneme_timing([("a", pred_dur)]) is None


def test_empty_input_returns_none():
    assert phoneme_timing([]) is None


def test_every_mapped_label_is_a_known_viseme_key():
    from ovos_utils.lang.visimes import VISIMES

    unknown = {v for v in _KOKORO_TO_ARPABET.values() if v not in VISIMES}
    assert not unknown


def test_modifiers_and_phonemes_do_not_overlap():
    assert not (MODIFIERS & set(_KOKORO_TO_ARPABET))
