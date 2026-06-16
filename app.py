import hashlib
import math
import random
import re
import struct
import wave
from io import BytesIO
from uuid import uuid4

import streamlit as st
from gtts import gTTS
from PIL import Image
from streamlit_local_storage import LocalStorage

from mrbunny_core import (
    extract_text_from_image,
    generate_image,
    generate_treblo_music,
    get_ai_response,
    get_secret,
    load_user_conversations,
    save_user_conversations,
)


st.set_page_config(page_title="MrBunny AI", page_icon="🐰", layout="wide")

BROWSER_DEVICE_KEY = "mrbunny_device_id_v1"
SAMPLE_RATE = 22050

def init_session_state() -> None:
    st.session_state.setdefault("conversations", {})
    st.session_state.setdefault("current_convo", None)
    st.session_state.setdefault("show_image_uploader", False)
    st.session_state.setdefault("rename_mode", set())
    st.session_state.setdefault("feedback", {})
    st.session_state.setdefault("pending_audio", "")
    st.session_state.setdefault("device_id", None)
    st.session_state.setdefault("device_storage_loaded", False)
    st.session_state.setdefault("device_storage_attempts", 0)
    st.session_state.setdefault("ghost_conversations", set())


def get_local_storage() -> LocalStorage:
    return LocalStorage()


def load_device_state() -> None:
    if st.session_state.device_storage_loaded:
        return

    local_storage = get_local_storage()
    device_id = local_storage.getItem(BROWSER_DEVICE_KEY)
    if device_id in (None, "") and st.session_state.device_storage_attempts < 1:
        st.session_state.device_storage_attempts += 1
        st.rerun()

    if not device_id:
        device_id = uuid4().hex
        local_storage.setItem(BROWSER_DEVICE_KEY, device_id, key="browser_device_id_saver")

    st.session_state.device_id = device_id
    conversations, current_convo = load_user_conversations(device_id)
    st.session_state.conversations = conversations
    st.session_state.current_convo = current_convo
    st.session_state.ghost_conversations = set()
    st.session_state.device_storage_attempts = 0
    st.session_state.device_storage_loaded = True


def save_device_chats() -> None:
    device_id = st.session_state.device_id
    if not device_id:
        return

    persisted_conversations = {
        convo_id: convo
        for convo_id, convo in st.session_state.conversations.items()
        if convo_id not in st.session_state.ghost_conversations
    }
    persisted_current = st.session_state.current_convo
    if persisted_current not in persisted_conversations:
        persisted_current = next(iter(persisted_conversations), None)
    save_user_conversations(device_id, persisted_conversations, persisted_current)


def clear_saved_chats() -> None:
    st.session_state.conversations = {}
    st.session_state.current_convo = None
    st.session_state.rename_mode = set()
    st.session_state.feedback = {}
    st.session_state.pending_audio = ""
    st.session_state.ghost_conversations = set()
    save_device_chats()
    st.rerun()


def remove_emojis(text: str) -> str:
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)


def wants_image_generation(text: str) -> bool:
    lowered = text.lower().strip()
    image_phrases = (
        "draw ",
        "draw me",
        "make an image",
        "generate an image",
        "create an image",
        "make a picture",
        "generate a picture",
        "create a picture",
        "make art",
        "generate art",
        "create art",
        "illustrate",
        "image of",
        "picture of",
    )
    return any(phrase in lowered for phrase in image_phrases)


def note_frequency(note_name: str, octave: int) -> float:
    notes = {
        "C": -9,
        "C#": -8,
        "D": -7,
        "D#": -6,
        "E": -5,
        "F": -4,
        "F#": -3,
        "G": -2,
        "G#": -1,
        "A": 0,
        "A#": 1,
        "B": 2,
    }
    return 440.0 * (2 ** ((notes[note_name] + (octave - 4) * 12) / 12))


def envelope(position: int, length: int, attack: float = 0.04, release: float = 0.18) -> float:
    if length <= 0:
        return 0.0
    progress = position / length
    if progress < attack:
        return progress / attack
    if progress > 1 - release:
        return max(0.0, (1 - progress) / release)
    return 1.0


def sine(freq: float, t: float) -> float:
    return math.sin(2 * math.pi * freq * t)


def soft_square(freq: float, t: float) -> float:
    return math.tanh(2.4 * sine(freq, t))


def choose_music_style(prompt: str) -> dict:
    text = prompt.lower()
    style = {
        "name": "dreamy synth loop",
        "bpm": 96,
        "scale": ["C", "D", "E", "G", "A"],
        "chords": [["C", "E", "G"], ["A", "C", "E"], ["F", "A", "C"], ["G", "B", "D"]],
        "wave": "sine",
        "swing": 0.0,
    }

    if any(word in text for word in ("lofi", "lo-fi", "chill", "calm", "study", "sleep")):
        style.update(
            {
                "name": "lo-fi chill loop",
                "bpm": 82,
                "scale": ["A", "B", "C", "E", "G"],
                "chords": [["A", "C", "E"], ["F", "A", "C"], ["C", "E", "G"], ["G", "B", "D"]],
                "swing": 0.12,
            }
        )
    elif any(word in text for word in ("game", "8bit", "8-bit", "arcade", "retro")):
        style.update(
            {
                "name": "retro game loop",
                "bpm": 128,
                "scale": ["C", "D", "E", "G", "A"],
                "chords": [["C", "E", "G"], ["G", "B", "D"], ["A", "C", "E"], ["F", "A", "C"]],
                "wave": "square",
            }
        )
    elif any(word in text for word in ("sad", "dark", "moody", "cinematic", "epic")):
        style.update(
            {
                "name": "moody cinematic loop",
                "bpm": 72,
                "scale": ["D", "F", "G", "A", "C"],
                "chords": [["D", "F", "A"], ["A", "C", "E"], ["B", "D", "F"], ["G", "B", "D"]],
            }
        )
    elif any(word in text for word in ("happy", "dance", "pop", "upbeat", "party")):
        style.update(
            {
                "name": "bright pop loop",
                "bpm": 118,
                "scale": ["G", "A", "B", "D", "E"],
                "chords": [["G", "B", "D"], ["D", "F#", "A"], ["E", "G", "B"], ["C", "E", "G"]],
            }
        )
    return style


def synth_note(samples: list[float], start: int, length: int, freq: float, volume: float, wave_name: str = "sine") -> None:
    for offset in range(max(0, length)):
        index = start + offset
        if index >= len(samples):
            break
        t = offset / SAMPLE_RATE
        tone = soft_square(freq, t) if wave_name == "square" else sine(freq, t)
        samples[index] += tone * volume * envelope(offset, length)


def synth_kick(samples: list[float], start: int, length: int, volume: float) -> None:
    for offset in range(length):
        index = start + offset
        if index >= len(samples):
            break
        progress = offset / length
        freq = 95 - 55 * progress
        samples[index] += sine(freq, offset / SAMPLE_RATE) * volume * ((1 - progress) ** 2)


def synth_hat(samples: list[float], start: int, length: int, volume: float, rng: random.Random) -> None:
    for offset in range(length):
        index = start + offset
        if index >= len(samples):
            break
        progress = offset / length
        samples[index] += rng.uniform(-1, 1) * volume * ((1 - progress) ** 3)


def samples_to_wav(samples: list[float]) -> bytes:
    output = BytesIO()
    peak = max(max(abs(sample) for sample in samples), 0.01)
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        for sample in samples:
            value = int(max(-1.0, min(1.0, sample / peak * 0.85)) * 32767)
            wav_file.writeframes(struct.pack("<h", value))
    output.seek(0)
    return output.read()


def generate_music(prompt: str, duration: int = 16) -> tuple[str, bytes | None]:
    """Generate a free local WAV loop from the prompt. Returns (reply, wav_bytes)."""
    seed = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12], 16)
    rng = random.Random(seed)
    style = choose_music_style(prompt)
    total_samples = SAMPLE_RATE * duration
    samples = [0.0] * total_samples
    beat_seconds = 60 / style["bpm"]
    beat_samples = int(SAMPLE_RATE * beat_seconds)
    step_samples = max(1, beat_samples // 2)
    steps = max(1, total_samples // step_samples)

    melody = [rng.choice(style["scale"]) for _ in range(steps)]
    for step in range(steps):
        start = step * step_samples
        chord = style["chords"][(step // 8) % len(style["chords"])]

        if step % 8 == 0:
            for note in chord:
                synth_note(samples, start, beat_samples * 4, note_frequency(note, 4), 0.11, style["wave"])

        if step % 2 == 0:
            synth_note(samples, start, int(beat_samples * 0.9), note_frequency(chord[0], 2), 0.22, "square")

        if step % 4 == 0:
            synth_kick(samples, start, int(beat_samples * 0.55), 0.9)
        elif step % 4 == 2:
            synth_hat(samples, start, int(beat_samples * 0.25), 0.22, rng)

        if step % 2 == 1 or "game" in style["name"]:
            offset = int(step_samples * style["swing"] if step % 2 else 0)
            synth_note(
                samples,
                start + offset,
                int(step_samples * 0.82),
                note_frequency(melody[step], rng.choice([4, 5])),
                0.16,
                style["wave"],
            )

    return f"MrBunny made a free {style['name']} from your prompt.", samples_to_wav(samples)


def speak(text: str) -> None:
    clean_text = remove_emojis(text).strip()
    if not clean_text:
        st.warning("There is no readable text to play.")
        return

    try:
        audio_buffer = BytesIO()
        gTTS(clean_text).write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        st.audio(audio_buffer.read(), format="audio/mp3")
    except Exception as exc:
        st.warning(f"Audio generation failed: {exc}")


def render_generated_image(image_bytes: bytes | None) -> None:
    if not image_bytes:
        return

    try:
        generated_image = Image.open(BytesIO(image_bytes))
        st.image(generated_image, use_container_width=True)
    except Exception as exc:
        st.warning(f"Generated image could not be displayed: {exc}")


def render_generated_music(music_bytes: bytes | None) -> None:
    if not music_bytes:
        return
    st.audio(music_bytes, format="audio/wav")


def render_generated_music_url(music_url: str | None) -> None:
    if not music_url:
        return
    st.audio(music_url)


def add_convo(name: str) -> None:
    clean_name = name.strip()
    if not clean_name:
        return
    convo_id = str(uuid4())
    st.session_state.conversations[convo_id] = {"name": clean_name, "messages": []}
    st.session_state.current_convo = convo_id
    st.session_state.ghost_conversations.discard(convo_id)
    save_device_chats()


def delete_convo(convo_id: str) -> None:
    if convo_id not in st.session_state.conversations:
        return

    del st.session_state.conversations[convo_id]
    st.session_state.rename_mode.discard(convo_id)
    st.session_state.ghost_conversations.discard(convo_id)

    if st.session_state.current_convo == convo_id:
        remaining = list(st.session_state.conversations.keys())
        st.session_state.current_convo = remaining[0] if remaining else None
    save_device_chats()


def rename_convo(convo_id: str, new_name: str) -> None:
    clean_name = new_name.strip()
    if convo_id in st.session_state.conversations and clean_name:
        st.session_state.conversations[convo_id]["name"] = clean_name
        save_device_chats()


def toggle_ghost_mode(convo_id: str) -> None:
    if convo_id in st.session_state.ghost_conversations:
        st.session_state.ghost_conversations.remove(convo_id)
    else:
        st.session_state.ghost_conversations.add(convo_id)
    save_device_chats()


def render_sidebar() -> None:
    with st.sidebar:
        st.title("💬 Conversations")
        st.caption("Chats are saved for this device without sign-in.")
        if st.button("Clear saved chats", use_container_width=True):
            clear_saved_chats()
        st.markdown("---")

        current_convo = st.session_state.current_convo
        ghost_enabled = current_convo in st.session_state.ghost_conversations if current_convo else False
        ghost_label = "👻 Ghost On" if ghost_enabled else "👻 Ghost Off"
        if st.button(ghost_label, use_container_width=True, help="Toggle whether the current chat is saved"):
            if current_convo:
                toggle_ghost_mode(current_convo)
                st.rerun()

        if ghost_enabled:
            st.caption("This conversation will not be saved to browser storage.")

        with st.form("new_convo_form", clear_on_submit=True):
            new_convo_name = st.text_input("Create New Conversation")
            create_clicked = st.form_submit_button("Create")
            if create_clicked and new_convo_name.strip():
                add_convo(new_convo_name)
                st.rerun()

        for convo_id, convo in list(st.session_state.conversations.items()):
            is_current = convo_id == st.session_state.current_convo
            row = st.container()
            cols = row.columns([0.72, 0.14, 0.14])
            label = f"👉 {convo['name']}" if is_current else convo["name"]

            if cols[0].button(label, key=f"select_{convo_id}", use_container_width=True):
                st.session_state.current_convo = convo_id
                save_device_chats()
                st.rerun()

            if cols[1].button("✍️", key=f"rename_btn_{convo_id}", use_container_width=True, help="Rename"):
                if convo_id in st.session_state.rename_mode:
                    st.session_state.rename_mode.remove(convo_id)
                else:
                    st.session_state.rename_mode.add(convo_id)
                st.rerun()

            if cols[2].button("🗑️", key=f"del_{convo_id}", use_container_width=True, help="Delete"):
                delete_convo(convo_id)
                st.rerun()

            if convo_id in st.session_state.rename_mode:
                new_name = st.text_input(
                    "Rename to",
                    value=convo["name"],
                    key=f"rename_input_{convo_id}",
                )
                if st.button("Save name", key=f"save_rename_{convo_id}"):
                    rename_convo(convo_id, new_name)
                    st.session_state.rename_mode.discard(convo_id)
                    st.rerun()


def render_feedback(idx: int) -> None:
    feedback = st.session_state.feedback
    current = feedback.get(idx)
    col1, col2, col3 = st.columns([0.14, 0.14, 0.72])

    if col1.button("Play", key=f"speak_{idx}", use_container_width=True):
        st.session_state.pending_audio = str(idx)

    if col2.button("👍", key=f"like_{idx}", use_container_width=True):
        feedback[idx] = "liked"

    if col3.button("👎", key=f"dislike_{idx}", use_container_width=True):
        feedback[idx] = "disliked"

    if current == "liked":
        st.caption("Liked")
    elif current == "disliked":
        st.caption("Disliked")


def render_main() -> None:
    st.title("🐰 MrBunny AI")
    st.caption("Your friendly AI assistant")

    api_key = get_secret("OPENROUTER_API_KEY")
    ocr_api_key = get_secret("OCR_API_KEY")

    if not api_key:
        st.error(
            "Missing `OPENROUTER_API_KEY`. Add a real key in Streamlit Cloud app secrets, "
            "`.streamlit/secrets.toml`, `secrets.toml`, or `.env`."
        )
        st.stop()

    if st.session_state.current_convo is None:
        st.info("Create or select a conversation to begin chatting with MrBunny.")
        return

    convo = st.session_state.conversations[st.session_state.current_convo]
    ghost_enabled = st.session_state.current_convo in st.session_state.ghost_conversations

    if ghost_enabled:
        st.info("Ghost mode is on for this chat. Messages here will not be saved.")

    for idx, msg in enumerate(convo["messages"]):
        with st.chat_message("user"):
            st.write(msg["user"])
        with st.chat_message("assistant"):
            if msg["ai"]:
                st.write(msg["ai"])
            render_generated_image(msg.get("image_bytes"))
            render_generated_music(msg.get("music_bytes"))
            render_generated_music_url(msg.get("music_url"))
            render_feedback(idx)

        if st.session_state.pending_audio == str(idx):
            speak(msg["ai"])
            st.session_state.pending_audio = ""

    uploaded_file = None
    if st.session_state.show_image_uploader:
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["png", "jpg", "jpeg"],
            key="chat_image_upload",
        )

    with st.form("chat_form", clear_on_submit=True):
        input_col, send_col, upload_col, image_col, music_col = st.columns([5, 1, 1, 1, 1])
        user_text = input_col.text_input("Type your message:")
        send_clicked = send_col.form_submit_button("Chat")
        upload_clicked = upload_col.form_submit_button("📥 Upload")
        image_clicked = image_col.form_submit_button("🎨 Image")
        music_clicked = music_col.form_submit_button("🎵 Music")

        st.caption("Use `Chat` for replies, `🎨 Image` for pictures, and `🎵 Music` to generate music.")

        if upload_clicked:
            st.session_state.show_image_uploader = not st.session_state.show_image_uploader
            st.rerun()

        if music_clicked:
            clean_text = user_text.strip()
            if not clean_text:
                st.warning("Describe the music you want to generate.")
                return
            treblo_api_key = get_secret("TREBLO_API_KEY")
            with st.spinner("MrBunny is composing with Treblo... This can take a couple of minutes."):
                reply, music_url = generate_treblo_music(clean_text, treblo_api_key)
            convo["messages"].append({
                "user": clean_text,
                "ai": reply,
                "image_bytes": None,
                "music_bytes": None,
                "music_url": music_url,
            })
            if not ghost_enabled:
                save_device_chats()
            st.rerun()

        if send_clicked or image_clicked:
            clean_text = user_text.strip()
            if not clean_text:
                st.warning("Type a message before sending.")
                return

            should_generate_image = image_clicked or wants_image_generation(clean_text)

            if should_generate_image:
                with st.spinner("MrBunny is drawing..."):
                    reply, image_bytes = generate_image(clean_text, api_key)
                convo["messages"].append(
                    {"user": clean_text, "ai": reply, "image_bytes": image_bytes, "music_bytes": None}
                )
                if not ghost_enabled:
                    save_device_chats()
                st.rerun()

            combined_prompt = clean_text
            if uploaded_file is not None:
                try:
                    image = Image.open(uploaded_file).convert("RGB")
                    ocr_text = extract_text_from_image(image, ocr_api_key)
                    if ocr_text:
                        combined_prompt = f"[Image text extracted: {ocr_text}]\n\n{clean_text}"
                except Exception as exc:
                    st.warning(f"Failed to process uploaded image: {exc}")

            with st.spinner("MrBunny is thinking..."):
                reply = get_ai_response(combined_prompt, api_key, convo["messages"])

            convo["messages"].append({"user": clean_text, "ai": reply, "image_bytes": None, "music_bytes": None})
            if not ghost_enabled:
                save_device_chats()
            st.rerun()


def main() -> None:
    init_session_state()
    load_device_state()
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
