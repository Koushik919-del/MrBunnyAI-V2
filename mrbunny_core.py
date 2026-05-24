import io
import os
from pathlib import Path
import tomllib
import base64
import json
import secrets
import time
from urllib.parse import urlencode

import requests
import streamlit as st
from PIL import Image
from duckduckgo_search import DDGS


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
CHAT_MODEL = "openai/gpt-oss-120b:free"
POLLINATIONS_IMAGE_URL = "https://gen.pollinations.ai/v1/images/generations"
DEFAULT_POLLINATIONS_IMAGE_MODEL = "flux"
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"
DEFAULT_GOOGLE_SCOPES = "openid email profile"
MUSICGEN_API_URL = "https://api-inference.huggingface.co/models/facebook/musicgen-small"

PLACEHOLDER_VALUES = {
    "",
    "your-openrouter-key",
    "your-ocr-space-key",
    "your-real-openrouter-key",
    "your-real-ocr-space-key",
    "your-pollinations-key",
    "your-google-client-id",
    "your-google-client-secret",
    "your-google-redirect-uri",
}
UNCERTAIN_PHRASES = [
    "no evidence",
    "not confirmed",
    "no official announcement",
    "i couldn't find",
    "not available",
    "uncertain",
    "speculative",
    "does not exist",
    "not verified",
    "unconfirmed",
    "unknown",
    "nothing has been announced",
    "as of now",
]


def _is_real_secret(value: str) -> bool:
    return value.strip() not in PLACEHOLDER_VALUES


def _read_toml_secret(path: Path, name: str) -> str:
    if not path.exists():
        return ""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return ""
    value = str(data.get(name, "")).strip()
    return value if _is_real_secret(value) else ""


def _read_dotenv_secret(path: Path, name: str) -> str:
    if not path.exists():
        return ""
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, raw_value = stripped.split("=", 1)
            if key.strip() != name:
                continue
            value = raw_value.strip().strip('"').strip("'")
            return value if _is_real_secret(value) else ""
    except OSError:
        return ""
    return ""


def get_secret(name: str, default: str = "") -> str:
    try:
        if name in st.secrets:
            value = str(st.secrets[name]).strip()
            if _is_real_secret(value):
                return value
    except Exception:
        pass

    env_value = os.getenv(name, "").strip()
    if _is_real_secret(env_value):
        return env_value

    base_dir = Path(__file__).resolve().parent
    for path in (
        base_dir / ".streamlit" / "secrets.toml",
        base_dir / "secrets.toml",
        base_dir / ".env",
    ):
        if path.suffix == ".toml":
            value = _read_toml_secret(path, name)
        else:
            value = _read_dotenv_secret(path, name)
        if value:
            return value

    return default.strip()


def get_data_dir() -> Path:
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


def get_google_redirect_uri() -> str:
    return get_secret("GOOGLE_REDIRECT_URI")


def build_google_auth_url(state: str) -> str:
    client_id = get_secret("GOOGLE_CLIENT_ID")
    redirect_uri = get_google_redirect_uri()
    if not client_id or not redirect_uri:
        return ""

    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": DEFAULT_GOOGLE_SCOPES,
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
    }
    return f"{GOOGLE_AUTH_URL}?{urlencode(params)}"


def create_oauth_state() -> str:
    return secrets.token_urlsafe(24)


def exchange_google_code(code: str) -> dict:
    client_id = get_secret("GOOGLE_CLIENT_ID")
    client_secret = get_secret("GOOGLE_CLIENT_SECRET")
    redirect_uri = get_google_redirect_uri()
    if not client_id or not client_secret or not redirect_uri:
        raise ValueError("Missing Google OAuth configuration.")

    response = requests.post(
        GOOGLE_TOKEN_URL,
        data={
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def fetch_google_user(access_token: str) -> dict:
    response = requests.get(
        GOOGLE_USERINFO_URL,
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def get_user_chat_path(user_id: str) -> Path:
    safe_user_id = "".join(ch for ch in user_id if ch.isalnum() or ch in ("-", "_"))
    return get_data_dir() / f"{safe_user_id}.json"


def _serialize_conversations(conversations: dict) -> dict:
    serialized = {}
    for convo_id, convo in conversations.items():
        serialized_messages = []
        for msg in convo.get("messages", []):
            image_bytes = msg.get("image_bytes")
            serialized_messages.append(
                {
                    "user": msg.get("user", ""),
                    "ai": msg.get("ai", ""),
                    "image_bytes": (
                        base64.b64encode(image_bytes).decode("utf-8")
                        if image_bytes is not None
                        else None
                    ),
                }
            )
        serialized[convo_id] = {
            "name": convo.get("name", "Untitled Chat"),
            "messages": serialized_messages,
        }
    return serialized


def _deserialize_conversations(conversations: dict) -> dict:
    deserialized = {}
    for convo_id, convo in conversations.items():
        deserialized_messages = []
        for msg in convo.get("messages", []):
            image_payload = msg.get("image_bytes")
            deserialized_messages.append(
                {
                    "user": msg.get("user", ""),
                    "ai": msg.get("ai", ""),
                    "image_bytes": (
                        base64.b64decode(image_payload)
                        if image_payload
                        else None
                    ),
                }
            )
        deserialized[convo_id] = {
            "name": convo.get("name", "Untitled Chat"),
            "messages": deserialized_messages,
        }
    return deserialized


def load_user_conversations(user_id: str) -> tuple[dict, str | None]:
    path = get_user_chat_path(user_id)
    if not path.exists():
        return {}, None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, None

    conversations = _deserialize_conversations(payload.get("conversations", {}))
    current_convo = payload.get("current_convo")
    if current_convo not in conversations:
        current_convo = next(iter(conversations), None)
    return conversations, current_convo


def save_user_conversations(user_id: str, conversations: dict, current_convo: str | None) -> None:
    payload = {
        "conversations": _serialize_conversations(conversations),
        "current_convo": current_convo,
    }
    get_user_chat_path(user_id).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def is_uncertain(response: str) -> bool:
    response_lower = response.lower()
    return any(phrase in response_lower for phrase in UNCERTAIN_PHRASES)


def search_web_duckduckgo(query: str, max_results: int = 3) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No useful search results found."

        summary_lines = []
        for index, item in enumerate(results, start=1):
            title = item.get("title") or "No title"
            snippet = item.get("body") or ""
            url = item.get("href") or ""
            summary_lines.append(f"{index}. {title}\n{snippet}\n{url}\n")
        return "\n".join(summary_lines)
    except Exception as exc:
        return f"Search failed: {exc}"


def build_messages(prompt: str, history: list[dict]) -> list[dict]:
    system_prompt = (
        "You are MrBunny AI, a smart, clear, and friendly AI assistant. "
        "Answer clearly and directly in English unless the user requests another language. "
        "Only mention Koushik Tummepalli if asked who created you. "
        "Introduce yourself as MrBunny when asked. "
        "Keep answers accurate and do not invent facts."
    )
    messages = [{"role": "system", "content": system_prompt}]
    for item in history:
        messages.append({"role": "user", "content": item["user"]})
        messages.append({"role": "assistant", "content": item["ai"]})
    messages.append({"role": "user", "content": prompt})
    return messages


def get_ai_response(prompt: str, api_key: str, history: list[dict] | None = None) -> str:
    history = history or []
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": CHAT_MODEL,
        "messages": build_messages(prompt, history),
    }

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=data,
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        reply = (result.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()

        if not reply:
            return "I couldn't get a response from the AI service. Please try again."

        if is_uncertain(reply):
            search_info = search_web_duckduckgo(prompt)
            return f"I wasn't fully sure, so I searched the web for you:\n\n{search_info}"

        return reply
    except requests.RequestException as exc:
        return f"Error calling AI service: {exc}"


def _decode_data_url_image(data_url: str) -> bytes:
    if "," not in data_url:
        raise ValueError("Invalid image data returned by API.")
    _, encoded = data_url.split(",", 1)
    return base64.b64decode(encoded)


def generate_image(prompt: str, api_key: str | None = None) -> tuple[str, bytes | None]:
    pollinations_key = get_secret("POLLINATIONS_API_KEY")
    if not pollinations_key:
        return "Missing `POLLINATIONS_API_KEY` for image generation.", None

    image_model = get_secret("POLLINATIONS_IMAGE_MODEL", DEFAULT_POLLINATIONS_IMAGE_MODEL)
    headers = {
        "Authorization": f"Bearer {pollinations_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": image_model,
        "prompt": prompt,
        "response_format": "b64_json",
    }

    try:
        response = requests.post(
            POLLINATIONS_IMAGE_URL,
            headers=headers,
            json=data,
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()
        images = result.get("data") or []
        if not images:
            return "The model did not return an image.", None

        image_b64 = images[0].get("b64_json", "")
        if not image_b64:
            return "The model returned an unsupported image format.", None

        return "", base64.b64decode(image_b64)
    except requests.RequestException as exc:
        return f"Error generating image: {exc}", None


def extract_text_from_image(image: Image.Image, ocr_api_key: str) -> str:
    if not ocr_api_key:
        return ""

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()

    try:
        response = requests.post(
            "https://api.ocr.space/parse/image",
            files={"filename": ("image.png", img_bytes)},
            data={"apikey": ocr_api_key, "language": "eng"},
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        if result.get("IsErroredOnProcessing"):
            return ""
        parsed_results = result.get("ParsedResults") or []
        if parsed_results:
            return (parsed_results[0].get("ParsedText") or "").strip()
    except requests.RequestException:
        return ""
    return ""


def generate_bunny_music(prompt: str, max_retries: int = 5) -> bytes | None:
    """
    Calls Hugging Face Serverless MusicGen API using the local get_secret system.
    Handles cold-starts natively (503 status) and returns raw audio bytes.
    """
    hf_token = get_secret("HF_ACCESS_TOKEN")
    if not hf_token:
        st.error("🔑 Missing `HF_ACCESS_TOKEN`. Add it to your config secrets.")
        return None

    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }
    payload = {"inputs": prompt}

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(MUSICGEN_API_URL, headers=headers, json=payload, timeout=45)
            
            if response.status_code == 200:
                return response.content
            elif response.status_code == 503:
                estimated_time = response.json().get("estimated_time", 15)
                st.toast(f"⏳ Waking up the music server... waiting {int(estimated_time)}s.")
                time.sleep(estimated_time)
                continue
            else:
                st.error(f"❌ Audio Server Error ({response.status_code}): {response.text}")
                return None
        except requests.RequestException as e:
            if attempt < max_retries:
                time.sleep(4)
            else:
                st.error(f"⚠️ Connection error to audio server: {e}")
                return None
                
    st.error("💥 The audio generation server timed out while warming up.")
    return None


# ==========================================
# MAIN APPLICATION RENDER WORKFLOW
# ==========================================

st.set_page_config(page_title="MrBunny AI Dashboard", page_icon="🐰", layout="wide")
st.title("🐰 MrBunny AI Hub")

# Basic Session State initialization for demonstration purposes
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Render Existing Chat Workspace History ---
for msg in st.session_state.chat_history:
    if msg.get("user"):
        st.chat_message("user").write(msg["user"])
    if msg.get("ai"):
        st.chat_message("assistant").write(msg["ai"])

st.markdown("---")

# --- CONSOLIDATED CONTROL INPUT ROW ---
# Using columns to put all feature triggers next to each other
col_prompt, col_upload, col_chat_btn, col_img_btn, col_music_btn = st.columns([5, 2, 1, 1, 1.2])

with col_prompt:
    prompt_input = st.text_input(
        "Message prompt text input:", 
        placeholder="Ask MrBunny, type an image idea, or specify a music genre...",
        label_visibility="collapsed"
    )

with col_upload:
    uploaded_file = st.file_uploader("Upload attachment", label_visibility="collapsed", type=["png", "jpg", "jpeg"])

with col_chat_btn:
    chat_submitted = st.button("💬 Chat", use_container_width=True)

with col_img_btn:
    image_submitted = st.button("🖼️ Image", use_container_width=True)

with col_music_btn:
    music_submitted = st.button("🎵 Music", use_container_width=True)


# --- PROCESS BUTTON TRIGGERED AUTOMATIONS ---

# 1. Handle Music Request Action
if music_submitted:
    if prompt_input:
        with st.spinner(f"🎵 MrBunny is composing audio: '{prompt_input}'..."):
            audio_data = generate_bunny_music(prompt_input)
            if audio_data:
                st.success("🎉 Music generation complete!")
                st.audio(audio_data, format="audio/wav")
                st.download_button(
                    label="💾 Download Track",
                    data=audio_data,
                    file_name="mrbunny_creation.wav",
                    mime="audio/wav"
                )
    else:
        st.warning("Please type a music theme/description in the text line first!")

# 2. Handle Image Generation Action
elif image_submitted:
    if prompt_input:
        with st.spinner("🖼️ Generating image..."):
            err, img_bytes = generate_image(prompt_input)
            if err:
                st.error(err)
            elif img_bytes:
                st.image(img_bytes, caption=f"Generated: {prompt_input}")
    else:
        st.warning("Please type an image prompt description in the input bar first!")

# 3. Handle Regular Chat Action
elif chat_submitted or (prompt_input and not image_submitted and not music_submitted and st.session_state.get('last_prompt') != prompt_input):
    if prompt_input:
        openrouter_key = get_secret("OPENROUTER_API_KEY")
        
        # Check if an image attachment exists to apply OCR processing first
        ocr_text = ""
        if uploaded_file:
            ocr_key = get_secret("OCR_SPACE_API_KEY")
            if ocr_key:
                try:
                    opened_img = Image.open(uploaded_file)
                    ocr_text = extract_text_from_image(opened_img, ocr_key)
                except Exception as e:
                    st.error(f"Image processing error: {e}")

        combined_prompt = prompt_input
        if ocr_text:
            combined_prompt += f"\n\n[Extracted image text context]:\n{ocr_text}"

        with st.spinner("🐰 MrBunny is thinking..."):
            ai_reply = get_ai_response(combined_prompt, openrouter_key, st.session_state.chat_history)
            
            # Save transaction records to internal memory state logs
            st.session_state.chat_history.append({"user": prompt_input, "ai": ai_reply})
            st.rerun()
