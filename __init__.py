"""
LLM Prompt Generator — ComfyUI custom node.

Generates a text prompt via the OpenRouter API, either from scratch (using
user-supplied instructions) or by describing an input image with a
vision-capable model.
"""

import base64
import io
import logging
from pathlib import Path

import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(__name__)

# ComfyUI root = parent of custom_nodes/ (this file is at custom_nodes/comfyui-llm-prompt/__init__.py)
COMFYUI_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_FILE = COMFYUI_ROOT / ".env"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MODES = ["image", "video", "video_ltx2.3"]

# ---------- System prompts -----------------------------------------------

_OUTPUT_RULES = (
    "Output ONLY the final prompt as plain text. No preamble, no quotes, "
    "no markdown, no labels, no commentary."
)

SYSTEM_IMAGE = (
    "You are a prompt engineer for a text-to-image model. Produce ONE "
    "detailed prompt describing the scene as a single still moment: subject, "
    "composition, lighting, color palette, textures, mood, style, lens/camera. "
    "Do NOT describe motion, change over time, camera movement, action verbs "
    "implying motion, sound, or audio. " + _OUTPUT_RULES
)

SYSTEM_VIDEO = (
    "You are a prompt engineer for a text-to-video model. Produce ONE detailed "
    "prompt as a single flowing paragraph in present tense, covering: subject, "
    "action over time, camera movement, lighting, mood, and style. "
    + _OUTPUT_RULES
)

# Derived from the LTX-2.3 prompt guide (ltx.io/model/model-blog/ltx-2-3-prompt-guide).
SYSTEM_VIDEO_LTX = (
    "You are a prompt engineer for the LTX-2.3 text-to-video model. Produce ONE "
    "prompt written as a single flowing paragraph in present tense, following "
    "this field order:\n"
    "  1. Establish the shot — cinematography terms and shot scale "
    "(e.g. wide shot, medium close-up, macro).\n"
    "  2. Set the scene — lighting, color palette, textures, atmosphere.\n"
    "  3. Describe the action — the core sequence from beginning to end.\n"
    "  4. Define the character(s) — age, hair, clothing, distinguishing features.\n"
    "  5. Identify camera movement(s) — how and when the camera moves "
    "(e.g. slow dolly in, handheld tracking, push, pull, pan, tilt). "
    "Describe how subjects appear after the movement so the model can complete "
    "the motion accurately.\n"
    "Rules:\n"
    "  - Match level of detail to shot scale: close-ups get more detail than wide shots.\n"
    "  - Be specific (\"young woman in a red coat\" not \"a person\").\n"
    "  - Use cinematic language (macro lens, tracking shot, golden hour).\n"
    "  - Use physical cues, not emotional labels (show \"clenched jaw\" instead of \"angry\").\n"
    "  - Avoid: readable text or logos, complex physics, chaotic motion, "
    "overloaded scenes with many characters, conflicting lighting, "
    "numerical specifications. Use natural language, not numbers.\n"
    + _OUTPUT_RULES
)

SYSTEM_VIDEO_LTX_AUDIO = (
    "\nAfter the visual paragraph, append a final audio section describing "
    "ambient sound, music, foley, speech, and singing. Specify the acoustic "
    "environment and voice qualities. Place spoken dialogue in quotation marks. "
    "Specify language and accent when relevant. Break dialogue into short "
    "phrases with brief acting directions between lines. Keep the audio "
    "description integrated as the final part of the same prompt."
)

SYSTEM_VIDEO_AUDIO = (
    "\nAlso describe ambient sound, music, and any speech. Place spoken "
    "dialogue in quotation marks."
)

# Extra guidance appended when an image is provided in a video mode (image-to-video).
SYSTEM_IMG2VID_SUFFIX = (
    "\nThe attached image is the starting frame. Focus on the motion and action "
    "that follows: how the subject moves, how the camera follows, what sounds "
    "emerge. Do NOT re-describe static elements already visible in the image."
)


def _build_system_prompt(mode: str, include_audio: bool, has_image: bool) -> str:
    if mode == "image":
        return SYSTEM_IMAGE
    if mode == "video_ltx2.3":
        parts = [SYSTEM_VIDEO_LTX]
        if include_audio:
            parts.append(SYSTEM_VIDEO_LTX_AUDIO)
        if has_image:
            parts.append(SYSTEM_IMG2VID_SUFFIX)
        return "".join(parts)
    # generic video
    parts = [SYSTEM_VIDEO]
    if include_audio:
        parts.append(SYSTEM_VIDEO_AUDIO)
    if has_image:
        parts.append(SYSTEM_IMG2VID_SUFFIX)
    return "".join(parts)


# -------------------------------------------------------------------------

MODELS = [
    "google/gemini-3.1-flash-lite",
    "google/gemini-flash-1.5",
    "google/gemini-pro-1.5",
    "google/gemini-2.0-flash-exp",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-haiku",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
]


def _load_env_key() -> str:
    """Read OPENROUTER_API_KEY from the ComfyUI root .env file, if present.

    Uses python-dotenv when available; otherwise falls back to a tiny parser so
    the node still works without the optional dependency installed.
    """
    if not ENV_FILE.exists():
        return ""
    try:
        from dotenv import dotenv_values  # type: ignore
        values = dotenv_values(ENV_FILE)
        return (values.get("OPENROUTER_API_KEY") or "").strip()
    except ImportError:
        try:
            for line in ENV_FILE.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == "OPENROUTER_API_KEY":
                    return v.strip().strip('"').strip("'")
        except OSError as e:
            logger.warning(f"Could not read {ENV_FILE}: {e}")
        return ""


def _tensor_to_base64_png(image_tensor) -> str:
    """Convert a ComfyUI IMAGE tensor (B,H,W,C float[0..1]) to a base64 PNG string."""
    img = image_tensor[0] if image_tensor.ndim == 4 else image_tensor
    arr = (img.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class LLMPromptGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (
                    MODES,
                    {
                        "default": "video_ltx2.3",
                        "tooltip": (
                            "image: still description, no motion. "
                            "video: generic text-to-video prompt. "
                            "video_ltx2.3: follows the LTX-2.3 prompt guide."
                        ),
                    },
                ),
                "include_audio": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Append an audio description (ambient, music, dialogue). Ignored in image mode.",
                    },
                ),
                "model": (MODELS, {"default": "google/gemini-3.1-flash-lite"}),
                "instructions": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "e.g. make it cinematic, slow dolly-in, golden hour",
                        "tooltip": "Extra instructions. If no image is connected, this is the source for the prompt.",
                    },
                ),
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "leave empty to use OPENROUTER_API_KEY from .env",
                        "multiline": False,
                    },
                ),
                "max_tokens": ("INT", {"default": 512, "min": 32, "max": 4096}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
            },
            "optional": {
                "image": ("IMAGE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "generate"
    CATEGORY = "LTX Tools"
    OUTPUT_NODE = True

    def generate(
        self,
        mode: str,
        include_audio: bool,
        model: str,
        instructions: str,
        api_key: str,
        max_tokens: int,
        temperature: float,
        image=None,
        unique_id=None,
        extra_pnginfo=None,
    ):
        key = (api_key or "").strip() or _load_env_key()
        if not key:
            raise ValueError(
                f"No OpenRouter API key. Provide one in the node, or set "
                f"OPENROUTER_API_KEY in {ENV_FILE}"
            )

        has_image = image is not None
        system_prompt = _build_system_prompt(mode, include_audio, has_image)

        if has_image:
            b64 = _tensor_to_base64_png(image)
            default_user_text = (
                "Describe this image as a still scene per the system instructions."
                if mode == "image"
                else "Write the prompt per the system instructions, starting from this image."
            )
            text = instructions.strip() or default_user_text
            user_content = [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                },
            ]
        else:
            if not instructions.strip():
                raise ValueError(
                    "No image and no instructions — provide at least one."
                )
            user_content = instructions.strip()

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        logger.info(
            f"OpenRouter request: model={model} mode={mode} audio={include_audio} image={has_image}"
        )
        resp = requests.post(
            OPENROUTER_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            timeout=120,
        )

        if resp.status_code != 200:
            raise RuntimeError(
                f"OpenRouter request failed ({resp.status_code}): {resp.text}"
            )

        data = resp.json()
        try:
            prompt = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, AttributeError) as e:
            raise RuntimeError(f"Unexpected OpenRouter response: {data}") from e

        # Persist generated text into the saved workflow JSON so it survives reloads.
        if extra_pnginfo and unique_id is not None:
            for node in (extra_pnginfo.get("workflow", {}) or {}).get("nodes", []) or []:
                if str(node.get("id")) == str(unique_id):
                    node["widgets_values"] = [prompt]
                    break

        return {"ui": {"text": (prompt,)}, "result": (prompt,)}


WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "LLMPromptGenerator": LLMPromptGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMPromptGenerator": "LLM Prompt Generator",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
