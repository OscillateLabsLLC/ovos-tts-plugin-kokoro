# ovos-tts-plugin-kokoro

[![Status: Proof of Concept](https://img.shields.io/badge/status-proof%20of%20concept-orange)](https://github.com/OscillateLabsLLC/.github/blob/main/SUPPORT_STATUS.md)

> **POC status — experimental, not for production, may be abandoned.** No API stability promise.

OVOS TTS plugin for [Kokoro](https://github.com/hexgrad/kokoro) — an 82M parameter multilingual TTS model by hexgrad. Same engine used by [VoiceMode](https://github.com/mbailey/voicemode), now wired up for the standard OVOS voice assistant.

## Install

```bash
pip install ovos-tts-plugin-kokoro
```

`espeak-ng` is required for the underlying G2P stack:

```bash
# Debian/Ubuntu
sudo apt-get install espeak-ng
# macOS
brew install espeak-ng
```

English voices also need spaCy's `en_core_web_sm` model. Misaki (the Kokoro G2P library) attempts to download it on first use but does not reload it in the same process, so you'll want to install it ahead of time:

```bash
python -m spacy download en_core_web_sm
```

For Japanese or Chinese voices, install the optional G2P extras:

```bash
pip install "ovos-tts-plugin-kokoro[ja,zh]"
```

### Linux: CPU-only torch (saves ~2GB)

On Linux, pip defaults to the CUDA torch wheel (~2.5GB). If you don't need GPU support, install torch from the CPU index first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install ovos-tts-plugin-kokoro
```

On macOS, this is not needed — PyPI torch is already CPU-only (~60MB). With `uv`, torch automatically resolves to the CPU-only wheel via the `tool.uv.sources` block in `pyproject.toml`.

## Configuration

```json
{
  "tts": {
    "module": "ovos-tts-plugin-kokoro",
    "ovos-tts-plugin-kokoro": {
      "voice": "af_bella"
    }
  }
}
```

### Voice options

Kokoro ships 56 built-in voices across 9 languages. The voice id encodes language + gender:

| Prefix | Language             | Examples                                    |
| ------ | -------------------- | ------------------------------------------- |
| `af_`  | American English (F) | `af_bella`, `af_heart`, `af_nicole`         |
| `am_`  | American English (M) | `am_michael`, `am_onyx`, `am_eric`          |
| `bf_`  | British English (F)  | `bf_alice`, `bf_emma`, `bf_lily`            |
| `bm_`  | British English (M)  | `bm_george`, `bm_fable`, `bm_daniel`        |
| `jf_` / `jm_` | Japanese      | `jf_alpha`, `jm_kumo` *(needs `[ja]` extra)* |
| `zf_` / `zm_` | Mandarin      | `zf_xiaoxiao`, `zm_yunjian` *(needs `[zh]` extra)* |
| `ef_` / `em_` | Spanish       | `ef_dora`, `em_alex`                        |
| `ff_`  | French (F)           | `ff_siwis`                                  |
| `hf_` / `hm_` | Hindi         | `hf_alpha`, `hm_omega`                      |
| `if_` / `im_` | Italian       | `if_sara`, `im_nicola`                      |
| `pf_` / `pm_` | Brazilian Portuguese | `pf_dora`, `pm_alex`                  |

The voice id determines which Kokoro language pipeline is used, regardless of the OVOS active language. Picking `bm_george` will speak through the British pipeline even if `lang` is `en-US`.

See the full [hexgrad/Kokoro-82M VOICES.md](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md) for samples.

## Language support

The plugin maps the active OVOS language (BCP-47, e.g. `fr-FR`) to a Kokoro single-letter language code:

| OVOS lang | Kokoro code | Language             |
| --------- | ----------- | -------------------- |
| `en` / `en-us` | `a`    | American English     |
| `en-gb`   | `b`         | British English      |
| `es`      | `e`         | Spanish              |
| `fr`      | `f`         | French               |
| `hi`      | `h`         | Hindi                |
| `it`      | `i`         | Italian              |
| `ja`      | `j`         | Japanese             |
| `pt` / `pt-br` | `p`    | Brazilian Portuguese |
| `zh`      | `z`         | Mandarin             |

Lookup tries the full BCP-47 tag first (e.g. `en-gb`), then falls back to the base subtag, then to American English. Unknown languages fall back to American English with a log line. **The voice id always wins over the language map** — a voice prefixed `bm_` always uses the British pipeline.

### Override the language map

```json
{
  "tts": {
    "module": "ovos-tts-plugin-kokoro",
    "ovos-tts-plugin-kokoro": {
      "voice": "af_bella",
      "speed": 1.0,
      "language_aliases": {
        "en": "b"
      },
      "preload_languages": ["en", "fr"]
    }
  }
}
```

| Key                  | Type           | Default     | Description                                                              |
| -------------------- | -------------- | ----------- | ------------------------------------------------------------------------ |
| `voice`              | str            | `af_bella`  | Any built-in voice id (see table above).                                 |
| `speed`              | float          | `1.0`       | Playback speed multiplier passed to KPipeline.                           |
| `sample_rate`        | int            | `16000`     | Output sample rate in Hz. Kokoro's native rate is 24000; the plugin resamples. |
| `device`             | str or null    | `"cpu"`     | Torch device — `"cpu"`, `"cuda"`, `"mps"`, or `null` to let Kokoro auto-select. |
| `language_aliases`   | dict           | `{}`        | Override or extend the BCP-47 → Kokoro code map.                          |
| `preload_languages`  | list[str]      | `[]`        | BCP-47 codes to load eagerly during plugin init instead of lazy-loading. |

> **Memory note:** Each loaded language pipeline holds the 82M parameter model + a g2p stack. The plugin caches one pipeline per (language, device) pair, so leaving `preload_languages` empty and letting the cache warm on demand keeps the resident set small.

> **Apple Silicon note:** Despite MPS being available on M-series Macs, **CPU is the fastest device for Kokoro on Apple Silicon**. The vocoder leans heavily on `torch.stft`/`istft`, which are weak spots on the Metal backend — measured RTF on an M3 Max was ~0.08 on CPU vs ~0.40 on MPS. The default of `"cpu"` is intentional; only set `device` to `"cuda"` if you actually have a discrete NVIDIA GPU.

## License

Apache-2.0
