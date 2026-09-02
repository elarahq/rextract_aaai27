import base64
import logging
import os
from io import BytesIO
from typing import List, Optional

import requests
from PIL import Image

import config

logger = logging.getLogger(__name__)


def query_vlm(
    prompt: str,
    images: List[Image.Image],
    model: str,
    token: Optional[str] = None,
    num_predict: int = 1000,  # unused for HF but kept for interface parity
    think: bool = False,      # unused for HF but kept for interface parity
) -> tuple[str, int, int]:
    hf_token = token or os.getenv("HF_TOKEN")
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }

    sent_images = images[: config.MAX_IMAGES_TO_VLM]
    content = [{"type": "text", "text": prompt}]
    for img in sent_images:
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
        )

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1000,
        "stream": False,
    }

    response = requests.post(config.HF_API_URL, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    result = response.json()
    text = result["choices"][0]["message"]["content"]
    usage = result.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    logger.info(
        f"[HF] model={model}  images={len(sent_images)}"
        f"  tokens={input_tokens}→{output_tokens}  out={text[:120]}"
    )
    return text, input_tokens, output_tokens
