FROM python:3.12-slim AS base

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg espeak-ng && \
    rm -rf /var/lib/apt/lists/*

# Install torch CPU-only first to avoid pulling CUDA wheels (~2GB savings)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install ovos-tts-server and the plugin
COPY . /tmp/plugin
# Pin gradio-client to match gradio 3.x (ovos-tts-server pulls gradio 3.36)
RUN pip install --no-cache-dir "gradio-client<1.0" "ovos-tts-server>=0.1.0" /tmp/plugin && \
    rm -rf /tmp/plugin

# Misaki's G2P stack needs en_core_web_sm. Misaki tries to download it on
# first use but doesn't reload it in the same process, so install ahead.
RUN python3 -m spacy download en_core_web_sm

# Pre-download model weights so first request is fast
RUN python3 -c "\
from kokoro import KPipeline; \
p = KPipeline(lang_code='a'); \
chunks = list(p('warmup', voice='af_bella')); \
print(f'Kokoro pipeline loaded, produced {len(chunks)} chunk(s)')"

# Write default config — voice=af_bella, sample_rate=16000
RUN mkdir -p /root/.config/mycroft && \
    echo '{"tts": {"ovos-tts-plugin-kokoro": {"voice": "af_bella", "sample_rate": 16000}}}' \
    > /root/.config/mycroft/mycroft.conf

EXPOSE 9666

ENTRYPOINT ["ovos-tts-server", "--engine", "ovos-tts-plugin-kokoro", "--cache"]
