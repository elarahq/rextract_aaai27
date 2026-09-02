import base64
import logging
import re
from io import BytesIO
from typing import List, Optional

import requests
from PIL import Image

import config

logger = logging.getLogger(__name__)


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def query_vlm(
    prompt: str,
    images: List[Image.Image],
    model: str,
    token: Optional[str] = None,  # unused for Ollama but kept for interface parity
    num_predict: int = config.OLLAMA_NUM_PREDICT,
    think: bool = False,
) -> tuple[str, int, int]:
    encoded_images = []
    for img in images[: config.MAX_IMAGES_TO_VLM]:
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        encoded_images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": encoded_images,
            }
        ],
        "stream": False,
        "think": think,
        "options": {"num_predict": num_predict},
    }

    response = requests.post(config.OLLAMA_API_URL, json=payload, timeout=config.OLLAMA_TIMEOUT)
    response.raise_for_status()
    result = response.json()
    message = result["message"]

    # When think=True Ollama may return a separate "thinking" key; log it and use clean "content"
    thinking = message.get("thinking", "")
    if thinking:
        logger.debug(f"[Ollama] thinking={thinking[:300]}")

    text = message["content"]
    # Strip any inline <think>…</think> blocks that some models embed in content
    text = _THINK_RE.sub("", text).strip()

    input_tokens = result.get("prompt_eval_count", 0)
    output_tokens = result.get("eval_count", 0)
    logger.info(
        f"[Ollama] model={model}  images={len(encoded_images)}"
        f"  tokens={input_tokens}→{output_tokens}\n  out={text[:120]}"
    )
    return text, input_tokens, output_tokens
