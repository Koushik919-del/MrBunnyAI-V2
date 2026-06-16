# MrBunny AI V2

MrBunny AI is a Streamlit assistant with:

- Multi-conversation chat
- Browser-saved chats on the same device
- Optional OCR text extraction from uploaded images
- Voice playback with gTTS
- Treblo Melodia v3 music generation
- OpenRouter-backed responses
- AI that can make images

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
TREBLO_API_KEY = "your-treblo-key"
```

3. Run the app:

```bash
streamlit run app.py
```

## Notes

- Do not commit real API keys.
- Music generation uses Treblo's API. Generated Treblo song URLs may expire, so download anything you want to keep.
- OCR is optional. If `OCR_API_KEY` is missing, image upload still works but text extraction is skipped.
- Chats are stored in the browser on the same device, so clearing browser storage can remove them.
