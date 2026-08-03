# comfyui-llm-prompt

A ComfyUI custom node that generates text prompts via the [OpenRouter](https://openrouter.ai) API.

It can describe an input image, write a prompt from scratch, or do both — and switches between **image**, generic **video**, **LTX-2.3 video**, and **MiniMax H3 video** styles via a single dropdown.

The generated text is shown inline inside the node *and* returned as a `STRING` output, so it can feed straight into your sampler / text-encode chain.

---

## Features

- Four modes:
  - `image` — describes a single still moment. No motion, no camera movement, no audio.
  - `video` — generic text-to-video prompt. Single paragraph, present tense, covering subject / action / camera / lighting / mood / style.
  - `video_ltx2.3` — follows the [LTX-2.3 prompt guide](https://ltx.io/model/model-blog/ltx-2-3-prompt-guide) field order (shot → scene → action → characters → camera → audio), with the guide's do's and don'ts baked into the system prompt.
  - `video_minimax` — follows the [MiniMax H3 video prompt writing guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md): three labeled fields (`integrated_multimodal_description`, `overall_soundscape`, `non_diegetic_music`), `[Shot N]` timeline with timestamps, three-dimensional camera-motion phrasing, `(S1)` speaker IDs with `<d>[Language] …</d>` dialogue tags, and the guide's image-alignment header for keyframe tasks.
- Optional audio description (ambient sound, music, dialogue in quotation marks, language/accent) for video modes.
- Optional `clip_length` (seconds) — tells the LLM how long the clip is so it scales the action to fit (no film scripts in a 5-second clip). In `video_minimax` mode it also bounds the shot timestamps and fills in the last-frame alignment timestamp. `0` = unspecified.
- Optional image input — when connected in a video mode, the node prompts the LLM to focus on what happens *next*, not re-describe the static frame.
- Optional second image input (`image_last`, `video_minimax` only) for keyframe tasks:

  | Connected | MiniMax task |
  | --------- | ------------ |
  | *(none)* | T2VA — text to video+audio |
  | `image` | I2VA — image is the first frame |
  | `image` + `image_last` | FL2VA — interpolate first → last frame |
  | `image_last` | L2VA — converge toward the last frame |
- API key read from a node field, or falls back to `OPENROUTER_API_KEY` in a `.env` file at the ComfyUI root.
- Generated prompt is displayed inside the node (read-only multiline widget) and persists across workflow reloads.

---

## Installation

Clone into your `custom_nodes` directory:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone git@github.com:rlemson7/comfyui-llm-prompt.git
```

Install dependencies into your ComfyUI Python environment:

```bash
pip install -r comfyui-llm-prompt/requirements.txt
```

Restart ComfyUI and hard-refresh the browser tab so the new frontend JS loads.

The node appears under **LTX Tools → LLM Prompt Generator**.

---

## Configuration

Set your OpenRouter API key either:

1. **In the node** — paste it into the `api_key` field, or
2. **In `.env`** (recommended) — create `/path/to/ComfyUI/.env` with:

   ```
   OPENROUTER_API_KEY=sk-or-v1-...
   ```

   The node uses `python-dotenv` when available and falls back to a tiny parser otherwise. The key is read from the ComfyUI root (the directory containing `custom_nodes/`).

---

## Inputs

| Name            | Type    | Notes |
| --------------- | ------- | ----- |
| `mode`          | dropdown | `image`, `video`, `video_ltx2.3` (default), `video_minimax` |
| `include_audio` | BOOLEAN | Adds an audio description section in video modes. Ignored when `mode = image`. In `video_minimax` mode, off means a silent clip (`overall_soundscape` / `non_diegetic_music` become `N/A`). |
| `clip_length`   | FLOAT   | Clip duration in seconds for video modes. The LLM is told to scale the action to this length. `0` (default) = unspecified. |
| `model`         | dropdown | OpenRouter model id. Default `google/gemini-3.1-flash-lite`. Edit `MODELS` in `__init__.py` to add more. |
| `instructions`  | STRING  | Extra guidance ("make it cinematic", "slow dolly-in"). Required if no image is connected. |
| `api_key`       | STRING  | Leave empty to use the `.env` fallback. |
| `max_tokens`    | INT     | Response cap. Default 512. Consider ≥1024 for `video_minimax` — its three-field prompts run longer. |
| `temperature`   | FLOAT   | Default 0.7. |
| `image`         | IMAGE *(optional)* | When connected, sent as a base64 PNG to a vision-capable model. First frame in `video_minimax` mode. |
| `image_last`    | IMAGE *(optional)* | `video_minimax` only: the clip's last frame (ignored in other modes). See the task table above. |

### Output

| Name     | Type   |
| -------- | ------ |
| `prompt` | STRING |

The same text also renders inside the node after `Queue Prompt`.

---

## How the system prompt is built

The node assembles the system prompt server-side from `mode` + `include_audio` + whether an image is connected — you don't (and can't) edit the system prompt from the graph. To tweak it, edit the prompt strings near the top of [`__init__.py`](__init__.py).

The LTX-2.3 system prompt encodes the guide's rules:

- Single flowing paragraph, present tense.
- Field order: shot → scene → action → characters → camera movement → (audio).
- Specific, cinematic language ("macro lens", "tracking shot", "golden hour").
- Physical cues over emotional labels ("clenched jaw" instead of "angry").
- Avoid: readable text/logos, complex physics, chaotic motion, overloaded scenes, conflicting lighting, numerical specifications.

When `include_audio` is on, an audio section is appended: ambient sound, music, foley, speech. Dialogue goes in quotation marks; language and accent are specified when relevant.

When an image is connected in a video mode, an image-to-video instruction is added so the LLM focuses on motion and what follows the starting frame rather than re-describing static elements.

The MiniMax system prompt encodes the H3 guide's conventions:

- Three labeled fields: `integrated_multimodal_description`, `overall_soundscape` (1–4 sentences, no dialogue), `non_diegetic_music` (1–3 sentences, instrumentation/tempo/rhythm/dynamics — no mood words).
- `[Shot 1]` opens with overall style + composition; later shots start `[Shot N] At MM:SS.SSS, the camera cuts to...`.
- Camera motion written as prose combining motion type + amplitude + speed ("pushes in with small amplitude at slow speed").
- Stable speaker IDs `(S1)`, dialogue as `<d>[English] exact words.</d>`, on-screen text in double quotes verbatim.
- When reference frames are connected, the prompt begins with the guide's image-alignment line (I2VA / FL2VA / L2VA variants); `clip_length` supplies the last-frame timestamp.

When `clip_length` > 0 in any video mode, the LLM is told the exact duration and instructed to scale the action to fit — one continuous action or a few beats, not a compressed storyline. In `video_minimax` mode shot timestamps are additionally bounded to the clip's duration.

---

## Tips

- **Run only this node** without firing your whole workflow: select the downstream nodes (KSampler, VAE Decode, etc.) and press **Ctrl+B** to bypass them. Only the LLM call runs.
- **See the result** — the prompt appears in the read-only text widget inside the node after each execution.
- **Switch models** — vision-capable OpenRouter models work for image inputs (e.g. Gemini, Claude 3.5 Sonnet, GPT-4o). Text-only models are fine when no image is connected.

---

## Files

- [`__init__.py`](__init__.py) — node class, system prompts, OpenRouter call.
- [`web/llm_prompt.js`](web/llm_prompt.js) — frontend widget that displays the generated text.
- [`requirements.txt`](requirements.txt) — `requests`, `Pillow`, `numpy`, `python-dotenv`.

---

## License

MIT.
