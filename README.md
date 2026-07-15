# CosyVoice 3 for RunPod

Self-contained CUDA image for `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` with:

- model weights baked into the image
- German and multilingual zero-shot voice cloning
- normal WAV/PCM synthesis endpoint
- true text-input plus audio-output streaming over WebSocket
- persistent voice registry under `/workspace/cosyvoice-data`
- optional bearer-token protection

Docker image:

```text
manuztl/cosyvoice3:latest
```

## Build on Docker Hub through GitHub Actions

Create this GitHub repository secret:

```text
DOCKERHUB_TOKEN
```

The token must have permission to push to `manuztl/cosyvoice3`. Every push to `main` then builds and publishes:

```text
manuztl/cosyvoice3:latest
manuztl/cosyvoice3:sha-<commit>
```

The build downloads the model weights into the image. RunPod therefore does not need to download the model during startup.

## RunPod settings

Use:

```text
Container image: manuztl/cosyvoice3:latest
Expose HTTP port: 8000
Container disk: 20 GB or more
Volume mount: /workspace
GPU: NVIDIA GPU with at least 12-16 GB VRAM for the first test
```

No Docker command override is required.

Recommended environment variables:

```text
PORT=8000
FP16=1
VOICE_DATA_DIR=/workspace/cosyvoice-data
API_KEY=<optional-secret>
DEFAULT_VOICE=sample
```

The included `sample` voice is only intended for immediate functional tests. Register a clean, authorized reference recording for Jarvis afterward.

## Health check

```bash
curl http://localhost:8000/health
```

Expected fields include `model_loaded`, `gpu`, `sample_rate`, and `voices`.

## Basic German WAV test

Without `API_KEY`:

```bash
curl -sS http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "fun-cosyvoice3-0.5b-2512",
    "voice": "sample",
    "input": "Guten Abend, Manuel. CosyVoice drei läuft jetzt auf dem RunPod.",
    "response_format": "wav",
    "stream": false
  }' \
  --output test.wav
```

With `API_KEY`, add:

```bash
-H "Authorization: Bearer $API_KEY"
```

## Clone and register a voice

The transcript must match the reference recording exactly. A clean 6-15 second WAV is a good starting point.

```bash
curl -sS -X POST http://localhost:8000/v1/voices/jarvis \
  -F 'prompt_text=Ich bin bereit und warte auf deine nächste Anweisung.' \
  -F 'prompt_wav=@jarvis-reference.wav'
```

The generated speaker profile is persisted in:

```text
/workspace/cosyvoice-data/spk2info.pt
```

Because `/workspace` is a RunPod volume, the voice survives container replacement.

List voices:

```bash
curl http://localhost:8000/v1/voices
```

Use the clone:

```bash
curl -sS http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "voice": "jarvis",
    "input": "Verstanden. Ich kümmere mich darum.",
    "instructions": "You are a helpful assistant. Speak German calmly, warmly and confidently. Keep the delivery controlled and natural.",
    "response_format": "wav"
  }' \
  --output jarvis.wav
```

## PCM output streaming with a complete text

CosyVoice cannot alter `speed` while streaming. Streaming output uses signed 16-bit little-endian mono PCM.

```bash
curl -N http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "voice": "sample",
    "input": "Das ist ein Test der kontinuierlichen Audioausgabe.",
    "response_format": "pcm",
    "stream": true
  }' \
  --output stream.pcm

ffplay -f s16le -ar 24000 -ac 1 stream.pcm
```

## True text-input streaming over WebSocket

Endpoint:

```text
ws://HOST:8000/v1/audio/speech/stream
```

Protocol:

```json
{"type":"session.start","voice":"sample"}
{"type":"input.text","text":"Guten Abend, Manuel. "}
{"type":"input.text","text":"Der nächste Textteil trifft erst später ein. "}
{"type":"input.done"}
```

The server responds with a `session.ready` JSON message, then binary `s16le` PCM chunks, followed by `session.done`.

Minimal Python client:

```python
import asyncio
import json
import wave

import websockets


async def main():
    uri = "ws://localhost:8000/v1/audio/speech/stream"
    pcm = bytearray()

    async with websockets.connect(uri, max_size=None) as ws:
        await ws.send(json.dumps({"type": "session.start", "voice": "sample"}))
        print(await ws.recv())

        for text in [
            "Guten Abend, Manuel. ",
            "Dieser Satz wird Stück für Stück übertragen. ",
            "Das Audio kann bereits vorher beginnen.",
        ]:
            await ws.send(json.dumps({"type": "input.text", "text": text}))
            await asyncio.sleep(0.3)

        await ws.send(json.dumps({"type": "input.done"}))

        while True:
            message = await ws.recv()
            if isinstance(message, bytes):
                pcm.extend(message)
                continue
            event = json.loads(message)
            print(event)
            if event.get("type") == "session.done":
                break

    with wave.open("bistream.wav", "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(pcm)


asyncio.run(main())
```

## API documentation

After startup:

```text
http://HOST:8000/docs
```

## Important runtime choice

The image deliberately uses CosyVoice's native PyTorch LLM runtime with `load_vllm=False`. Upstream CosyVoice currently disables streaming text input when its internal vLLM mode is enabled.
