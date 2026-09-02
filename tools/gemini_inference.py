import logging
from io import BytesIO
from typing import List, Optional

from google import genai
from google.genai import types
from PIL import Image

import config

logger = logging.getLogger(__name__)
logging.getLogger("google.genai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(
            api_key=config.GEMINI_API_KEY,
            http_options=types.HttpOptions(timeout=config.GEMINI_TIMEOUT * 1000),  # SDK expects milliseconds
        )
    return _client


def query_vlm(
    prompt: str,
    images: List[Image.Image],
    model: str,
    token: Optional[str] = None,  # unused for Gemini but kept for interface parity
    num_predict: int = config.GEMINI_NUM_PREDICT,
    think: bool = False,
) -> tuple[str, int, int]:
    sent_images = images[: config.MAX_IMAGES_TO_VLM]

    contents: list = []
    for img in sent_images:
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        contents.append(types.Part.from_bytes(data=buffered.getvalue(), mime_type="image/jpeg"))
    contents.append(prompt)

    generate_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        max_output_tokens=num_predict,
        thinking_config=types.ThinkingConfig(include_thoughts=True) if think else None,
    )

    response = _get_client().models.generate_content(
        model=model,
        contents=contents,
        config=generate_config,
    )

    text = response.text or ""
    usage = response.usage_metadata
    input_tokens = getattr(usage, "prompt_token_count", 0) or 0
    output_tokens = getattr(usage, "candidates_token_count", 0) or 0

    logger.info(
        f"[Gemini] model={model}  images={len(sent_images)}"
        f"  tokens={input_tokens}→{output_tokens}  out={text[:120]}"
    )
    return text, input_tokens, output_tokens
