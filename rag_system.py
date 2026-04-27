import json
import chromadb
import ollama
import torch
import types
import sys
import time
import math
import edge_tts
import asyncio
import subprocess

sys.path.append("Tests")

from sentence_transformers import SentenceTransformer
from weather_station import get_weather_data
from dataset import weather_aq
from datetime import datetime

if not hasattr(torch.distributed, 'is initialized'):
    torch.distributed.is_initialized = types.MethodType(lambda self = None: 0, torch.distributed)
if not hasattr(torch.distributed, 'get_rank'):
    torch.distributed.get_rank= types.MethodType(lambda self = None: 0, torch.distributed)

torch.cuda.set_per_process_memory_fraction(0.8, device=0)
torch.cuda.empty_cache()

start_time = time.time()
embedding_model = "all-MiniLM-L6-v2"
weather_infoDB = "weatherInfo_db"
top_k = 10
ollama_model = "weather-agent_8b" #llama3.1:8b-instruct-q4_K_M with Modelfile - weather-agent_8b
json_response_file = "responses.json"

model = ollama.chat

docs_db = chromadb.PersistentClient(
    path="backend/vector_databases/weatherInfo_db_path")
docs_collection = docs_db.get_collection("weather_chunks")

prompts = {
    "General": "Give me a general weather summary on the weather conditions today. Be concise and clear",
    "Clothing": "Suggest appropriate clothing for people around the building and weather it is comfortable to be outside. Be concise and clear.",
    "Prediction": "Give the most accurate prediction of what the weather will be later in the day or the next day",
    "Alert": "If if any weather data exceeds safe limits, generate a short public advisory. If not respond with no severe weather conditions",
    "Safety": "Advise whether it is safe to leave driving or walking"
}

'''
weather_data = {
    "timestamp" : "2025-11-15 10:52:28",
    "temperature": 20,
    "humidity": 50,
    "rainfall": 50,
    "wind_speed": 15,
    "pressure": 1008,
    "light_intensity": 135      #Manual Weather Testing
}
'''
#weather_data = weather_aq(87, "weather_log.csv")       #Data captured

weather_data = get_weather_data()                      #Live weather data

T = weather_data.get("temperature")
RH = weather_data.get("humidity")
P = weather_data.get("pressure")

'''
HI = -8.784695 + 1.61139411*T + 2.338549*RH \
    - 0.14611605*T*RH - 0.012308094*T**2 \
    -0.01642828*RH**2 + 0.002211732*T**2*RH \
    + 0.00072546*T*RH**2 - 0.000003582*T**2*RH**2
weather_data['heat_index'] = round(HI, 2)

Es = 6.112 * math.exp((17.67 * T)/(T + 243.5)) #Saturation Vapour Pressure
E = RH/100 * Es #Actual Vapour Pressure
AH = 216.7 * (E / (T + 273.15)) #Absolute Humidity

weather_data["absolute_humidity"] = round(AH, 2)
'''

weather_text = (
    f"The temperature is {weather_data['temperature']}°C,"
    f"humidity is {weather_data['humidity']}%, "
    f"rainfall is {weather_data['rainfall']}mm, "
    f"windspeed is {weather_data['wind_speed']}m/s, "
    f"atmosperic pressure is {weather_data['pressure']}hPa, "
    f"light intensity is {weather_data['light_intensity']}lux, "
)


def retrieval(prompt_text, collection, top_k):

    embedder = SentenceTransformer(embedding_model, device="cpu")
    query_embedding = embedder.encode([prompt_text]).tolist()
    del embedder
    torch.cuda.empty_cache()
    result = collection.query(
        query_embeddings=query_embedding, n_results=top_k)
    return result["documents"][0] if result["documents"] else []


def augmented_prompt(prompt_text, weather_data, chunks, label):
    if label in ["General", "Prediction"]:
        general_info = "Output should be 4-6 sentences in a paragraph format. Also include a confidence level from 0-100%. Mention the weather parameters that justifies the above statment in the paragraph"
    else:
        general_info = "Output should be 2-4 sentences in a short paragraph format. Also include a confidence level from 0-100%. Mention the weather parameters that justifies the above statement in the paragraph"
    
    formatted_chunks = "\n".join([f"{i+1}. {chunks}" for i, chunk in enumerate(chunks)])
    #print(f"{formatted_chunks}")
    final_prompt = f"""
Based on the weather data and chunks provide responses to the following instruction/question. Take time of day into consideration as well:

Weather Data:
{weather_text}

Information:
{formatted_chunks}

Instructions:
{general_info}

Prompt:
{prompt_text}

"""
    #print(f"{final_prompt}")
    return final_prompt


def rag_pipeline():
    responses = {}
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for label, prompt_text in prompts.items():
        chunks = retrieval(prompt_text, docs_collection, top_k)
        final_prompt = augmented_prompt(
            prompt_text, weather_data, chunks, label)
        
        torch.cuda.empty_cache()
        response = model(
            model=ollama_model,
            messages=[{"role": "user", "content": final_prompt}],
            options={"temperature": 0.3,
                     "device_map": "auto",
                     "offload_folder": "/tmp"}
        )

        llm_response = response['message']['content']
        responses[label] = llm_response

    for label, output in responses.items():
        print(f"{label}:\n{output}\n{'-'*50}")

    data_to_save = {
        "timestamp": timestamp,
        "weather_data": weather_data,
        "llm_responses": responses
    }


    with open(json_response_file, "w") as f:
        json.dump([data_to_save], f, indent=2)

    print(f"\nResponses saved successfully for {timestamp}")

    return responses



def main():
    rag_pipeline()
    end_time = time.time()
    latency = end_time - start_time

    print(f"Response latency: {latency:.2f} seconds")
    print(torch.cuda.memory_summary)
    

main()