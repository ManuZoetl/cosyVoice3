from __future__ import annotations

import asyncio
import io
import os
import queue
import re
import sys
import threading
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np
import torch
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

COSYVOICE_ROOT = Path(os.getenv("COSYVOICE_ROOT", "/opt/CosyVoice"))
MODEL_DIR = Path(
    os.getenv(
        "MODEL_DIR",
        "/opt/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B",
    )
)
DATA_DIR = Path(os.getenv("VOICE_DATA_DIR", "/workspace/cosyvoice-data"))
VOICE_REGISTRY = DATA_DIR / "spk2info.pt"
MODEL_NAME = os.getenv("MODEL_NAME", "fun-cosyvoice3-0.5b-2512")
API_KEY = os.getenv("API_KEY", "").strip()
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "sample")
DEFAULT_INSTRUCTIONS = os.getenv("DEFAULT_INSTRUCTIONS", "").strip()
ENABLE_SAMPLE_VOICE = os.getenv("ENABLE_SAMPLE_VOICE", "1") == "1"
FP16 = os.getenv("FP16", "1") == "1"

sys.path.insert(0, str(COSYVOICE_ROOT))
sys.path.insert(0, str(COSYVOICE_ROOT / "third_party" / "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402

VOICE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
SAMPLE_VOICE_ID = "sample"
SAMPLE_PROMPT_WAV = COSYVOICE_ROOT / "asset" / "zero_shot_prompt.wav"
SAMPLE_PROMPT_TEXT = (
    "You are a helpful assistant.<|endofprompt|>"
    "希望你以后能够做的比我还好呦。"
)

model: Any | None = None
model_load_error: str | None = None
generation_lock = threading.Lock()
voice_lock = threading.Lock()


class SpeechRequest(BaseModel):
    model: str = MODEL_NAME
    input: str = Field(min_length=1)
    voice: str = DEFAULT_VOICE
    instructions: str | None = None
    response_format: Literal["wav", "pcm"] = "wav"
    stream: bool = False
    speed: float = Field(default=1.0, gt=0.5, lt=2.0)


class VoiceInfo(BaseModel):
    id: str
    builtin: bool = False


def require_http_auth(authorization: str | None) -> None:
    if not API_KEY:
        return
    expected = f"Bearer {API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")


def websocket_is_authorized(websocket: WebSocket) -> bool:
    if not API_KEY:
        return True
    authorization = websocket.headers.get("authorization")
    query_token = websocket.query_params.get("token")
    return authorization == f"Bearer {API_KEY}" or query_token == API_KEY


def ensure_model() -> Any:
    if model is None:
        detail = "CosyVoice model is not loaded"
        if model_load_error:
            detail = f"{detail}: {model_load_error}"
        raise HTTPException(status_code=503, detail=detail)
    return model


def normalize_instruction(instructions: str | None) -> str:
    text = (instructions or DEFAULT_INSTRUCTIONS).strip()
    if not text:
        return ""
    if not text.endswith("<|endofprompt|>"):
        text = f"{text}<|endofprompt|>"
    return text


def float_tensor_to_pcm16(tensor: torch.Tensor) -> bytes:
    audio = tensor.detach().cpu().float().reshape(-1).numpy()
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype("<i2", copy=False).tobytes()


def pcm16_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return buffer.getvalue()


def load_external_voices(instance: Any) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not VOICE_REGISTRY.exists():
        return
    saved = torch.load(VOICE_REGISTRY, map_location=instance.frontend.device, weights_only=True)
    if not isinstance(saved, dict):
        raise RuntimeError(f"Invalid voice registry: {VOICE_REGISTRY}")
    instance.frontend.spk2info.update(saved)


def save_external_voices(instance: Any) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temporary = VOICE_REGISTRY.with_suffix(".tmp")
    torch.save(instance.frontend.spk2info, temporary)
    temporary.replace(VOICE_REGISTRY)


def register_sample_voice(instance: Any) -> None:
    if not ENABLE_SAMPLE_VOICE or SAMPLE_VOICE_ID in instance.frontend.spk2info:
        return
    if not SAMPLE_PROMPT_WAV.exists():
        return
    instance.add_zero_shot_spk(
        SAMPLE_PROMPT_TEXT,
        str(SAMPLE_PROMPT_WAV),
        SAMPLE_VOICE_ID,
    )


def build_inference(
    instance: Any,
    text: str | Iterator[str],
    voice: str,
    instructions: str,
    stream: bool,
    speed: float = 1.0,
) -> Iterator[dict[str, torch.Tensor]]:
    if voice not in instance.frontend.spk2info:
        raise KeyError(f"Unknown voice: {voice}")

    if instructions:
        return instance.inference_instruct2(
            tts_text=text,
            instruct_text=instructions,
            prompt_wav="",
            zero_shot_spk_id=voice,
            stream=stream,
            speed=speed,
            text_frontend=False,
        )

    return instance.inference_zero_shot(
        tts_text=text,
        prompt_text="",
        prompt_wav="",
        zero_shot_spk_id=voice,
        stream=stream,
        speed=speed,
        text_frontend=False,
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model, model_load_error
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        instance = await asyncio.to_thread(
            AutoModel,
            model_dir=str(MODEL_DIR),
            load_vllm=False,
            load_trt=False,
            fp16=FP16,
        )
        await asyncio.to_thread(load_external_voices, instance)
        await asyncio.to_thread(register_sample_voice, instance)
        model = instance
    except Exception as exc:
        model_load_error = f"{type(exc).__name__}: {exc}"
        raise
    yield
    model = None


app = FastAPI(
    title="CosyVoice 3 Streaming API",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> JSONResponse:
    loaded = model is not None
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    payload = {
        "status": "ok" if loaded else "loading",
        "model_loaded": loaded,
        "model": MODEL_NAME,
        "model_dir": str(MODEL_DIR),
        "sample_rate": getattr(model, "sample_rate", None),
        "gpu": gpu_name,
        "cuda_available": torch.cuda.is_available(),
        "voices": sorted(model.frontend.spk2info.keys()) if loaded else [],
        "error": model_load_error,
    }
    return JSONResponse(payload, status_code=200 if loaded else 503)


@app.get("/v1/voices", response_model=list[VoiceInfo])
def list_voices(authorization: str | None = Header(default=None)) -> list[VoiceInfo]:
    require_http_auth(authorization)
    instance = ensure_model()
    return [
        VoiceInfo(id=voice_id, builtin=voice_id == SAMPLE_VOICE_ID)
        for voice_id in sorted(instance.frontend.spk2info.keys())
    ]


@app.post("/v1/voices/{voice_id}")
async def create_voice(
    voice_id: str,
    prompt_text: str = Form(...),
    prompt_wav: UploadFile = File(...),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    require_http_auth(authorization)
    instance = ensure_model()

    if not VOICE_ID_RE.fullmatch(voice_id):
        raise HTTPException(
            status_code=422,
            detail="voice_id may contain letters, numbers, dot, dash and underscore",
        )
    if not prompt_text.strip():
        raise HTTPException(status_code=422, detail="prompt_text is required")

    upload_dir = DATA_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(prompt_wav.filename or "voice.wav").suffix.lower() or ".wav"
    upload_path = upload_dir / f"{voice_id}{suffix}"
    upload_path.write_bytes(await prompt_wav.read())

    def register() -> None:
        with voice_lock:
            instance.add_zero_shot_spk(prompt_text.strip(), str(upload_path), voice_id)
            save_external_voices(instance)

    try:
        await asyncio.to_thread(register)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Voice registration failed: {type(exc).__name__}: {exc}",
        ) from exc

    return {
        "id": voice_id,
        "registered": True,
        "registry": str(VOICE_REGISTRY),
    }


@app.post("/v1/audio/speech")
def create_speech(
    request: SpeechRequest,
    authorization: str | None = Header(default=None),
) -> Response:
    require_http_auth(authorization)
    instance = ensure_model()
    instructions = normalize_instruction(request.instructions)

    if request.voice not in instance.frontend.spk2info:
        raise HTTPException(status_code=404, detail=f"Unknown voice: {request.voice}")
    if request.stream and request.speed != 1.0:
        raise HTTPException(
            status_code=422,
            detail="CosyVoice supports speed changes only for non-streaming generation",
        )

    def pcm_generator() -> Iterator[bytes]:
        with generation_lock:
            outputs = build_inference(
                instance=instance,
                text=request.input,
                voice=request.voice,
                instructions=instructions,
                stream=request.stream,
                speed=request.speed,
            )
            for output in outputs:
                yield float_tensor_to_pcm16(output["tts_speech"])

    if request.stream:
        if request.response_format != "pcm":
            raise HTTPException(
                status_code=422,
                detail="Streaming responses use response_format=pcm",
            )
        return StreamingResponse(
            pcm_generator(),
            media_type=f"audio/L16;rate={instance.sample_rate};channels=1",
            headers={
                "X-Audio-Sample-Rate": str(instance.sample_rate),
                "X-Audio-Sample-Format": "s16le",
            },
        )

    try:
        pcm = b"".join(pcm_generator())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Speech generation failed: {type(exc).__name__}: {exc}",
        ) from exc

    if request.response_format == "pcm":
        return Response(
            content=pcm,
            media_type=f"audio/L16;rate={instance.sample_rate};channels=1",
            headers={
                "X-Audio-Sample-Rate": str(instance.sample_rate),
                "X-Audio-Sample-Format": "s16le",
            },
        )

    return Response(
        content=pcm16_to_wav(pcm, instance.sample_rate),
        media_type="audio/wav",
    )


class AudioWorkerError:
    def __init__(self, exception: Exception):
        self.exception = exception


AUDIO_DONE = object()
TEXT_DONE = object()


@app.websocket("/v1/audio/speech/stream")
async def speech_websocket(websocket: WebSocket) -> None:
    if not websocket_is_authorized(websocket):
        await websocket.close(code=4401)
        return

    await websocket.accept()
    instance = model
    if instance is None:
        await websocket.send_json({"type": "error", "message": "Model is not loaded"})
        await websocket.close(code=1013)
        return

    try:
        start_message = await websocket.receive_json()
    except Exception:
        await websocket.close(code=4400)
        return

    if start_message.get("type") != "session.start":
        await websocket.send_json(
            {"type": "error", "message": "First message must be session.start"}
        )
        await websocket.close(code=4400)
        return

    voice = str(start_message.get("voice") or DEFAULT_VOICE)
    instructions = normalize_instruction(start_message.get("instructions"))
    if voice not in instance.frontend.spk2info:
        await websocket.send_json({"type": "error", "message": f"Unknown voice: {voice}"})
        await websocket.close(code=4404)
        return

    text_queue: queue.Queue[str | object] = queue.Queue(maxsize=64)
    audio_queue: asyncio.Queue[bytes | AudioWorkerError | object] = asyncio.Queue(maxsize=8)
    loop = asyncio.get_running_loop()

    def text_source() -> Iterator[str]:
        while True:
            item = text_queue.get()
            if item is TEXT_DONE:
                return
            yield str(item)

    streamed_text = text_source()

    def put_audio(item: bytes | AudioWorkerError | object) -> None:
        future = asyncio.run_coroutine_threadsafe(audio_queue.put(item), loop)
        future.result()

    def audio_worker() -> None:
        try:
            with generation_lock:
                outputs = build_inference(
                    instance=instance,
                    text=streamed_text,
                    voice=voice,
                    instructions=instructions,
                    stream=True,
                    speed=1.0,
                )
                for output in outputs:
                    put_audio(float_tensor_to_pcm16(output["tts_speech"]))
        except Exception as exc:
            put_audio(AudioWorkerError(exc))
        finally:
            put_audio(AUDIO_DONE)

    worker = threading.Thread(target=audio_worker, name="cosyvoice-stream", daemon=True)
    worker.start()

    await websocket.send_json(
        {
            "type": "session.ready",
            "voice": voice,
            "sample_rate": instance.sample_rate,
            "sample_format": "s16le",
            "channels": 1,
        }
    )

    async def receive_input() -> None:
        try:
            while True:
                message = await websocket.receive_json()
                message_type = message.get("type")
                if message_type == "input.text":
                    text = str(message.get("text") or "")
                    if text:
                        await asyncio.to_thread(text_queue.put, text)
                elif message_type in {"input.done", "session.cancel"}:
                    await asyncio.to_thread(text_queue.put, TEXT_DONE)
                    return
                else:
                    await websocket.send_json(
                        {"type": "error", "message": f"Unknown message type: {message_type}"}
                    )
        except WebSocketDisconnect:
            await asyncio.to_thread(text_queue.put, TEXT_DONE)

    async def send_audio() -> None:
        chunk_index = 0
        while True:
            item = await audio_queue.get()
            if item is AUDIO_DONE:
                await websocket.send_json(
                    {"type": "session.done", "audio_chunks": chunk_index}
                )
                return
            if isinstance(item, AudioWorkerError):
                exc = item.exception
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            await websocket.send_bytes(item)
            chunk_index += 1

    receiver = asyncio.create_task(receive_input())
    sender = asyncio.create_task(send_audio())

    try:
        await asyncio.gather(receiver, sender)
    except WebSocketDisconnect:
        pass
    finally:
        if not receiver.done():
            receiver.cancel()
        if not sender.done():
            sender.cancel()
        if worker.is_alive():
            try:
                text_queue.put_nowait(TEXT_DONE)
            except queue.Full:
                pass
