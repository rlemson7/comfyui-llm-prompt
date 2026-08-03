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

MODES = ["image", "video", "video_ltx2.3", "video_minimax"]

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

# Derived from the MiniMax H3 video prompt writing guide
# (huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md).
SYSTEM_VIDEO_MINIMAX = (
    "You are a prompt engineer for the MiniMax H3 video+audio generation "
    "model. Produce ONE prompt consisting of exactly three labeled fields, in "
    "this order, each starting on its own line:\n"
    "integrated_multimodal_description: the full audiovisual timeline of the "
    "clip, shot by shot — visuals, actions, camera work, shot transitions, "
    "dialogue, singing, and diegetic audio events in the order they occur.\n"
    "overall_soundscape: 1-4 sentences summarizing ambient sound, physical "
    "action sounds, and non-verbal human sounds (wind, rain, traffic, "
    "footsteps, fabric, impacts, breathing, laughter). No dialogue or singing "
    "here. Write N/A only if complete silence is requested.\n"
    "non_diegetic_music: 1-3 sentences describing background music the "
    "characters cannot hear: instrumentation, tempo, rhythm, dynamic changes. "
    "No abstract mood words and no explanation of the score's emotional "
    "function. Write N/A if there is no music.\n"
    "Rules for integrated_multimodal_description:\n"
    "  - Open [Shot 1] by stating the overall style and initial composition, "
    "e.g. \"[Shot 1] Live-action, cinematic, a medium-wide shot frames...\". "
    "Common styles: live-action, cinematic, 2D-animated, 3D CG, claymation, "
    "watercolor.\n"
    "  - The first shot carries no timestamp. Each later shot starts "
    "\"[Shot N] At MM:SS.SSS, the camera cuts to...\". Use \"the camera cuts "
    "to\" / \"the shot transitions to\" / \"the shot switches to\"; reserve "
    "cross-dissolve, fade, or wipe for explicit user requests.\n"
    "  - Write camera motion as natural prose combining three dimensions: "
    "motion type (push in/out, zoom in/out, pan left/right, truck left/right, "
    "tilt up/down, pedestal up/down, arc shot, tracking shot, static shot, "
    "slight/strong shake, POV, roll clockwise/counterclockwise), amplitude "
    "(with small/large amplitude), and speed (at slow/fast speed) — e.g. "
    "\"The camera pushes in with small amplitude at slow speed toward the "
    "folded letter.\"\n"
    "  - Give each speaking or singing character a stable ID: (S1), (S2); "
    "(S1,S2) when they vocalize together. Keep the same ID across shots; "
    "non-vocalizing characters get no ID.\n"
    "  - Dialogue format: speaker, action, and delivery outside the tag; only "
    "a language tag and the exact words inside it, e.g. the young woman with "
    "a quiet, breathy voice (S1) says: <d>[English] I get off at the next "
    "station.</d> Keep user-provided lines and punctuation verbatim. For "
    "voiceover write \"says in an off-screen voiceover\" and state that the "
    "on-screen character's lips remain completely closed.\n"
    "  - Put visible on-screen text (signs, banners, labels, subtitles, neon) "
    "in English double quotation marks, verbatim, untranslated.\n"
    "Output ONLY the prompt in exactly this structure, as plain text. No "
    "preamble, no surrounding quotes, no markdown, no commentary."
)

SYSTEM_MINIMAX_NO_AUDIO = (
    "\nThe user wants a silent clip: include no dialogue, singing, or "
    "diegetic audio events in the description, and write N/A for both "
    "overall_soundscape and non_diegetic_music."
)

# Used when the image only informs the LLM and is NOT fed to the video model.
SYSTEM_MINIMAX_IMG_REFERENCE_ONLY = (
    "\nThe attached image(s) are visual reference for YOU only — the video "
    "model will NOT receive them. Write a fully self-contained text-to-video "
    "prompt: describe the style, subjects, clothing, colors, and scene "
    "completely in words. Do NOT include an image alignment line, and do NOT "
    "mention Picture 1, reference pictures, or the attached image in any way."
)

SYSTEM_IMG_REFERENCE_ONLY = (
    "\nThe attached image is visual reference for YOU only — the video model "
    "will NOT receive it. Write a fully self-contained prompt: describe the "
    "subjects, style, and scene completely in words, and do not mention the "
    "image or assume the video model can see it."
)


def _minimax_task_suffix(has_first: bool, has_last: bool, clip_length: float) -> str:
    """Image-alignment instructions for the MiniMax keyframe tasks.

    T2VA (no images) needs no alignment line, hence the empty string.
    """
    if clip_length > 0:
        end_ts = f"{clip_length:.2f}"
        end_note = ""
    else:
        end_ts = "S.SS"
        end_note = (
            " (replace S.SS with a plausible clip duration in seconds, two "
            "decimal places)"
        )

    if has_first and has_last:
        return (
            "\nTwo reference images are attached: the FIRST is the first frame "
            "of the clip (Picture 1), the SECOND is the last frame (Picture 2). "
            "Begin the prompt with this alignment line, followed by one blank "
            "line, then the three fields:\n"
            "How the reference pictures align with the target video — Picture 1 "
            "(from Shot 1) aligns with the 0.00-second mark of the target "
            "video; Picture 2 (from Shot N) aligns with the "
            f"{end_ts}-second mark of the target video.{end_note}\n"
            "Keep the whole clip a single continuous shot so the model can "
            "interpolate from the first frame to the last, unless the user "
            "explicitly asks for multiple shots. Describe: the first-frame "
            "state, the observable intermediate changes (movement, pose, "
            "object manipulation, composition), the progressively narrowing "
            "differences, and finally the last-frame state."
        )
    if has_first:
        return (
            "\nThe attached image is the first frame of the clip (Picture 1). "
            "Begin the prompt with exactly this line, followed by one blank "
            "line, then the three fields:\n"
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced.\n"
            "Anchor style, subjects, composition, and scene from the image — "
            "keep character identity, clothing, colors, key objects, and "
            "spatial relationships consistent — then describe what happens "
            "next: action onset, continuous development, result or reaction. "
            "Do not merely re-describe the static image."
        )
    if has_last:
        return (
            "\nThe attached image is the LAST frame of the clip (Picture 1). "
            "Begin the prompt with this alignment line, followed by one blank "
            "line, then the three fields:\n"
            "How the reference pictures align with the target video — "
            "<Picture 1> (from [Shot N]) aligns with the "
            f"{end_ts}-second mark of the target video.{end_note}\n"
            "Infer a plausible earlier state from the user's intent and the "
            "last frame, then describe how the characters, objects, camera, "
            "and scene gradually approach the reference image: preceding "
            "state, explicit action and transition path, gradual convergence, "
            "and the final landing on the last frame."
        )
    return ""


def _clip_length_suffix(mode: str, clip_length: float) -> str:
    """Tell the LLM how long the clip is so it scales the action to fit."""
    if clip_length <= 0 or mode == "image":
        return ""
    n = f"{clip_length:g}"
    s = (
        f"\nThe target clip is exactly {n} seconds long. Scale the content to "
        f"that duration: include only as much action as can plausibly unfold "
        f"in {n} seconds. Do not compress a whole storyline into a short "
        "clip — prefer one continuous action or a few clear beats that fit "
        "comfortably."
    )
    if mode == "video_minimax":
        s += (
            f" All shot timestamps must fall between 00:00.000 and the clip "
            f"end at {n} seconds."
        )
    return s

# Extra guidance appended when an image is provided in a video mode (image-to-video).
SYSTEM_IMG2VID_SUFFIX = (
    "\nThe attached image is the starting frame. Focus on the motion and action "
    "that follows: how the subject moves, how the camera follows, what sounds "
    "emerge. Do NOT re-describe static elements already visible in the image."
)


def _build_system_prompt(
    mode: str,
    include_audio: bool,
    has_first: bool,
    has_last: bool,
    clip_length: float,
    image_as_keyframe: bool,
) -> str:
    if mode == "image":
        return SYSTEM_IMAGE
    if mode == "video_minimax":
        parts = [SYSTEM_VIDEO_MINIMAX]
        if not include_audio:
            parts.append(SYSTEM_MINIMAX_NO_AUDIO)
        if has_first or has_last:
            if image_as_keyframe:
                parts.append(
                    _minimax_task_suffix(has_first, has_last, clip_length)
                )
            else:
                parts.append(SYSTEM_MINIMAX_IMG_REFERENCE_ONLY)
        parts.append(_clip_length_suffix(mode, clip_length))
        return "".join(parts)
    if mode == "video_ltx2.3":
        parts = [SYSTEM_VIDEO_LTX]
    else:  # generic video
        parts = [SYSTEM_VIDEO]
    if include_audio:
        parts.append(
            SYSTEM_VIDEO_LTX_AUDIO if mode == "video_ltx2.3" else SYSTEM_VIDEO_AUDIO
        )
    if has_first or has_last:
        parts.append(
            SYSTEM_IMG2VID_SUFFIX if image_as_keyframe else SYSTEM_IMG_REFERENCE_ONLY
        )
    parts.append(_clip_length_suffix(mode, clip_length))
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
                            "video_ltx2.3: follows the LTX-2.3 prompt guide. "
                            "video_minimax: follows the MiniMax H3 prompt guide "
                            "(supports first/last reference frames)."
                        ),
                    },
                ),
                "include_audio": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Append an audio description (ambient, music, dialogue). "
                            "Ignored in image mode. In video_minimax mode, off = "
                            "silent clip (soundscape/music fields become N/A)."
                        ),
                    },
                ),
                "clip_length": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 600.0,
                        "step": 0.5,
                        "tooltip": (
                            "Clip duration in seconds (video modes only). Tells the "
                            "LLM how much action fits — no film scripts in a 5 s "
                            "clip. 0 = unspecified."
                        ),
                    },
                ),
                "image_as_keyframe": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "On: connected image(s) are real keyframes the video "
                            "model will also receive (alignment header / "
                            "starting-frame wording). Off: image(s) only inspire "
                            "the LLM — the prompt is written fully self-contained "
                            "for a pure text-to-video run, with no reference to "
                            "attached pictures."
                        ),
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
                "image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Reference image. In video_minimax mode this is the "
                            "FIRST frame of the clip."
                        ),
                    },
                ),
                "image_last": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "video_minimax only: LAST frame of the clip. Combine "
                            "with 'image' for first+last-frame interpolation, or "
                            "connect alone to converge toward this frame."
                        ),
                    },
                ),
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
        clip_length: float,
        image_as_keyframe: bool,
        model: str,
        instructions: str,
        api_key: str,
        max_tokens: int,
        temperature: float,
        image=None,
        image_last=None,
        unique_id=None,
        extra_pnginfo=None,
    ):
        key = (api_key or "").strip() or _load_env_key()
        if not key:
            raise ValueError(
                f"No OpenRouter API key. Provide one in the node, or set "
                f"OPENROUTER_API_KEY in {ENV_FILE}"
            )

        if image_last is not None and mode != "video_minimax":
            logger.warning(
                "image_last is only used in video_minimax mode; ignoring it."
            )
            image_last = None

        has_first = image is not None
        has_last = image_last is not None
        system_prompt = _build_system_prompt(
            mode, include_audio, has_first, has_last, clip_length, image_as_keyframe
        )

        # Order matters: Picture 1 first. For last-frame-only (L2VA), the last
        # frame IS Picture 1.
        images = [img for img in (image, image_last) if img is not None]
        if images:
            if mode == "image":
                default_user_text = (
                    "Describe this image as a still scene per the system instructions."
                )
            elif not image_as_keyframe:
                default_user_text = (
                    "Write the prompt per the system instructions. The attached "
                    "image(s) show the desired subject and style."
                )
            elif mode == "video_minimax":
                default_user_text = (
                    "Write the prompt per the system instructions, using the "
                    "attached reference frame(s)."
                )
            else:
                default_user_text = (
                    "Write the prompt per the system instructions, starting from this image."
                )
            text = instructions.strip() or default_user_text
            user_content = [{"type": "text", "text": text}]
            for img in images:
                b64 = _tensor_to_base64_png(img)
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    }
                )
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
            f"OpenRouter request: model={model} mode={mode} audio={include_audio} "
            f"images={len(images)} clip_length={clip_length}"
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
