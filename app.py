# python -m streamlit run app.py
import streamlit as st
import json
import time
import torch
import types
import html
import threading
import tempfile
import os
import asyncio
import edge_tts

from streamlit_autorefresh import st_autorefresh
from rag_system import rag_pipeline

if not hasattr(torch.distributed, 'is initialized'):
    torch.distributed.is_initialized = types.MethodType(lambda self = None: 0, torch.distributed)
if not hasattr(torch.distributed, 'get_rank'):
    torch.distributed.get_rank= types.MethodType(lambda self = None: 0, torch.distributed)

st.set_page_config(page_title="RAG Weather Dashboard",
                   page_icon="🌦️", 
                   layout="wide")

st.title("🌦️ RAG Weather Analysis Dashboard")
st.caption("Powered by Ollama and Streamlit")

st_autorefresh(interval=280 * 1000, key = "weather_refresh")

response_file = "responses.json"

def run_edge_tts(text: str):
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

    async def _generate():
        communicate = edge_tts.Communicate(
            text,
            voice="en-US-AriaNeural"  # You can change the voice here
        )
        await communicate.save(temp_wav)

    try:
        asyncio.run(_generate())
    except RuntimeError:
        # If Streamlit already has an event loop
        loop = asyncio.get_event_loop()
        loop.run_until_complete(_generate())

    return temp_wav


def get_responses():
    try:
        with open(response_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

if 'rag_thread_started' not in st.session_state:

    st.session_state.rag_thread_started = True

    def update_rag_data(interval = 200 ):
        while True:
            try:
                rag_pipeline()
            except Exception as e:
                print("Error running RAG:", e)
            time.sleep(interval)
    
    threading.Thread(target=update_rag_data, daemon=True).start()


#if not st.session_state["initial_data_ready"]:
 #   st.info("Fetching initial RAG data...Please wait.")
  #  st.stop()


responses = get_responses()

if not responses:
    st.info("No weather analyses available yet. Run the backend to generate some!")
else:
    for entry in reversed(responses):
        timestamp = entry.get('timestamp', 'No timestamp')
        weather_data = entry.get('weather_data', {})
        llm_responses = entry.get('llm_responses', {})

        st.markdown(f"### Updated: {timestamp}")

        st.markdown("Weather Metrics")
        cols = st.columns(3)
        cols[0].metric("Temperature (°C)", weather_data.get("temperature", "N/A"))
        cols[0].metric("Humidity (%)", weather_data.get("humidity", "N/A"))
        cols[0].metric("Rainfall (mm)", weather_data.get("rainfall", "N/A"))

        cols[1].metric("Wind Speed (km/h)", weather_data.get("wind_speed", "N/A"))
        cols[1].metric("Pressure (mbar)", weather_data.get("pressure", "N/A"))
        cols[1].metric("Light Intensity (lux)", weather_data.get("light_intensity", "N/A"))

        st.markdown("---")

        for label, text in llm_responses.items():
            if label in ["Alert", "Safety"]:
                block_color = "#FFCCCC"  # light red
            elif label == "Prediction":
                block_color = "#FFF2CC"  # light yellow
            else:
                block_color = "#CCFFCC"  # light green

            safe_text = html.escape(text).replace('\n','<br>')

            st.markdown(
                f'<div style=" background-color: {block_color}; padding: 15px; border-radius: 10px; font-size: 25px; line-height: 1.5; margin-bottom: 10px;">'
                f'<strong>{label}:</strong><br>{safe_text}</div>',
                
                unsafe_allow_html=True
            )


    latest_entry = responses[-1]
    weather_data = latest_entry.get("weather_data", {})
    llm_responses = latest_entry.get("llm_responses", {})

    tts_text = f"Hello! I am your weather assistant. The weather update as of {latest_entry.get('timestamp', 'Unknown')} is as follows: "

    tts_text += (
        f"The temperature is {weather_data.get('temperature', 'unknown')} degrees Celsius. "
        f"Humidity is {weather_data.get('humidity', 'unknown')} percent. "
        f"Rainfall is {weather_data.get('rainfall', 'unknown')} millimeters. "
        f"Wind speed is {weather_data.get('wind_speed', 'unknown')} kilometers per hour. "
        f"Pressure is {weather_data.get('pressure', 'unknown')} millibars. "
        f"Light intensity is {weather_data.get('light_intensity', 'unknown')} lux. "
    )

    for section, text in llm_responses.items():
        tts_text += f"{section} report: {text}. "

    try:
        wav_file = run_edge_tts(tts_text)
        if wav_file and os.path.exists(wav_file):
            st.audio(open(wav_file, "rb").read(), format="audio/wav")
    except Exception as e:
        st.error(f"TTS failed: {e}")



    st.markdown("----")

    last_updated = responses[-1].get("timestamp", "Unknown")
    st.markdown(
        f"""
        <div style="text-align:center; color:gray; font-size:14px; margin-top:20px;">
            Last updated: {last_updated} | Refreshes every 5 minutes
        </div>
        """,
        unsafe_allow_html=True
    )