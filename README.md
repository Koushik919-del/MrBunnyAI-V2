# Old-MrBunny-AI

MrBunny AI is a Streamlit assistant with:

- Multi-conversation chat
- Browser-saved chats on the same device
- Optional OCR text extraction from uploaded images
- Voice playback with gTTS
- Free local music loop generation without a paid music API
- OpenRouter-backed responses

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Add secrets in Streamlit app settings, `.streamlit/secrets.toml`, or environment variables:

```toml
OPENROUTER_API_KEY = "your-openrouter-key"
OCR_API_KEY = "your-ocr-space-key"
POLLINATIONS_API_KEY = "your-pollinations-key"
```

3. Run the app:

```bash
streamlit run app.py
```

## Notes

- Do not commit real API keys.
- Music generation is built in and does not need ElevenLabs or any paid music API.
- OCR is optional. If `OCR_API_KEY` is missing, image upload still works but text extraction is skipped.
- Chats are stored in the browser on the same device, so clearing browser storage can remove them.
