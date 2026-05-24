import time
import requests
import streamlit as st

# 1. Grab your secret key securely using Streamlit's built-in secrets tool
HF_TOKEN = st.secrets["HF_ACCESS_TOKEN"]

# 2. Point to Meta's free serverless MusicGen endpoint
API_URL = "https://api-inference.huggingface.co/models/facebook/musicgen-small"

def generate_bunny_music(prompt: str, output_path: str = "bunny_track.wav", max_retries: int = 5) -> str:
    """
    Sends a text prompt to Hugging Face's free serverless MusicGen instance.
    Includes a built-in loop to wait out 'Cold Start' 503 model-loading delays.
    """
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {"inputs": prompt}
    
    # Simple UI feedback indicator for Streamlit users
    with st.spinner(f"🥕 MrBunny is composing: '{prompt}'..."):
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
                
                # Success! Write the raw audio bytes to a file
                if response.status_code == 200:
                    with open(output_path, "wb") as f:
                        f.write(response.content)
                    return output_path
                
                # Model is asleep (Cold Start 503 error)
                elif response.status_code == 503:
                    estimated_time = response.json().get("estimated_time", 15)
                    st.toast(f"⏳ Waking up the music server... waiting {int(estimated_time)}s.")
                    time.sleep(estimated_time)
                    continue
                
                # Any other unexpected server error
                else:
                    st.error(f"❌ Error {response.status_code}: {response.text}")
                    return None
                    
            except requests.exceptions.RequestException as e:
                st.error(f"⚠️ Network error occurred: {e}")
                if attempt < max_retries:
                    time.sleep(5)
                else:
                    return None
                    
        st.error("💥 Max retries reached. The server took too long to wake up.")
        return None
