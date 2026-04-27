# RAG Weather Analysis Dashboard

A real-time weather monitoring and analysis system that combines a physical weather station, a Retrieval-Augmented Generation (RAG) pipeline powered by a local LLM, and a Streamlit web dashboard with text-to-speech output.

---

## Overview

This project reads live sensor data from a Modbus weather station, augments it with contextual knowledge from weather documents, and uses a locally hosted LLM (via Ollama) to generate human-readable weather summaries, alerts, clothing suggestions, and safety advisories — all displayed on an auto-refreshing web dashboard with voice readout. 

> ⚠️ **Note:** This project is optimized to run on the Jetson Orin Nano, but can be run on a PC with the recommended RAM and/or CPU requirements of the LLM used 

---

## Features

- **Live weather data ingestion** via Modbus RTU serial connection
- **RAG pipeline** using ChromaDB vector storage and `all-MiniLM-L6-v2` sentence embeddings
- **Local LLM inference** via Ollama (`llama3.1:8b` with a custom Modelfile)
- **Multi-category LLM responses**: General summary, Clothing advice, Weather prediction, Alerts, and Safety Advisories
- **Streamlit dashboard** with auto-refresh every 5 minutes
- **Text-to-speech readout** of the latest weather report using Microsoft Edge TTS
- **CSV dataset fallback** for offline testing with logged weather data

---

## Project Structure

```
├── app.py                  # Streamlit dashboard and TTS integration
├── rag_system.py           # Core RAG pipeline: retrieval, prompting, LLM inference
├── document_processor.py   # PDF ingestion, chunking, and vector DB population
├── weather_station.py      # Modbus RTU interface for live sensor data
├── dataset.py              # CSV-based weather data loader for testing
├── TTS_rag.py              # Standalone Edge TTS test script
├── weather_log.csv         # Logged historical weather data
├── requirements.txt        # Python dependencies
├── weather_documents/      # PDF knowledge base for RAG context
└── backend/
    └── vector_databases/
        └── weatherInfo_db_path/   # ChromaDB persistent vector store
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Dashboard | Streamlit |
| LLM Backend | Ollama (`llama3.1:8b-instruct-q4_K_M`) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector Database | ChromaDB |
| Text Splitting | LangChain NLTK Text Splitter |
| PDF Parsing | PyMuPDF (`fitz`) |
| Weather Hardware | Modbus RTU via `pymodbus` |
| Text-to-Speech | Microsoft Edge TTS (`edge-tts`) |
| ML Framework | PyTorch |

---

##  Setup & Installation

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally
- A CUDA-compatible GPU (recommended) or CPU
- A Modbus RTU weather station (Ultrasonic 7-in-1 RS485 Weather Sensor (SEN0657) used for this project) connected on `/dev/ttyUSB0` (for live data)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/rag-weather-dashboard.git
cd rag-weather-dashboard
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **Note:** The `torch` package in `requirements.txt` is pinned to an ARM/Jetson-specific wheel. Replace it with the appropriate PyTorch build for your system from [pytorch.org](https://pytorch.org/).

### 3. Pull and Configure the Ollama Model

```bash
ollama pull llama3.1:8b-instruct-q4_K_M
```

Then create the custom model using the included Modelfile:

```bash
ollama create weather-agent_8b -f Modelfile
```

### 4. Populate the Vector Database

Place your weather reference PDFs in the `weather_documents/` folder, then run:

```bash
python document_processor.py
```

This chunks, embeds, and stores the documents in ChromaDB.

### 5. Run the Dashboard

```bash
python -m streamlit run app.py
```

The dashboard will open in your browser and begin fetching RAG-generated weather analyses automatically.

---

## 🧪 Testing Without Hardware

To test without a physical weather station, edit `rag_system.py` and switch the data source to the CSV loader:

```python
# Comment out live data:
# weather_data = get_weather_data()

# Use CSV data instead:
weather_data = weather_aq(87, "weather_log.csv")
```

---

## 📡 Weather Station Configuration

The system reads from a Modbus RTU sensor at the following default settings:

| Parameter | Value |
|---|---|
| Port | `/dev/ttyUSB0` |
| Baud Rate | 4800 |
| Sensor ID | 1 |

Modify these constants at the top of `weather_station.py` to match your hardware.

---

## 📊 LLM Response Categories

The RAG pipeline generates responses across five categories for each weather reading:

| Label | Description |
|---|---|
| **General** | Overall weather summary (4–6 sentences) |
| **Clothing** | Clothing and outdoor comfort advice |
| **Prediction** | Forecast for later in the day or next day |
| **Alert** | Public advisory if any parameters exceed safe limits |
| **Safety** | Driving and walking safety recommendation |

---

## 🔧 Known Issues & Notes

- The `torch` wheel in `requirements.txt` is specific to an NVIDIA Jetson (ARM, CUDA 12.4). You must replace it with a compatible build for your system.
- NLTK `punkt` tokenizer data must be downloaded before first use. Uncomment the `nltk.download` lines in `document_processor.py` on first run.
- The dashboard currently saves only the latest RAG response to `responses.json` (single-entry list). Historical entries are overwritten on each pipeline run.

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## 👤 Author

**Thaeel Chetram**  
[GitHub](https://github.com/Tha33l) · [LinkedIn](www.linkedin.com/in/thaeel-chetram-3a451b251)
