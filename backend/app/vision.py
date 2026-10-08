import base64
import io
import os
from functools import lru_cache
from pathlib import Path

from groq import Groq
from PIL import Image

from .config import VISION_MODEL


@lru_cache
def groq_client() -> Groq:
    return Groq(api_key=os.environ["GROQ_API_KEY"])


def image_to_data_uri(image_path) -> str:
    with Image.open(Path(image_path)) as img:
        img = img.convert("RGB")
        img.thumbnail((1600, 1600))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


VISUAL_PROMPT = """This visual was extracted from page {page} of a company report.
Describe the useful business information visible in it.
- Chart/graph: list every visible value, highest/lowest, and the main trend.
- Diagram: components, flow, relationships, and any callout text verbatim.
- Scanned text page: transcribe the text and tables faithfully.
- Normal image: only useful factual information (labels, numbers).
Be concise and factual. Do not guess values you cannot read."""


def summarize_visual(image_path, page: int) -> str:
    resp = groq_client().chat.completions.create(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": VISUAL_PROMPT.format(page=page)},
                {"type": "image_url", "image_url": {"url": image_to_data_uri(image_path)}},
            ],
        }],
        temperature=0,
        max_completion_tokens=600,
    )
    return resp.choices[0].message.content.strip()
