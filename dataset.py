import pandas as pd

csv_path = "weather_log.csv"

def weather_aq(index, path = csv_path):
    df = pd.read_csv(path)
    
    act_index = index - 2
    if act_index < 0 or act_index >= len(df):
        raise IndexError("Out of Bounds")
    
    
    row = df.iloc[act_index]

    weather_data = {
        "timestamp": str(row["timestamp"]),
        "temperature": float(row["temperature (°C)"]),
        "humidity": float(row["humidity (%)"]),
        "rainfall": float(row["rainfall (mm)"]),
        "wind_speed": float(row["wind_speed (m/s)"]),
        "pressure": float(row["pressure (hPa)"]),
        "light_intensity": float(row["light_intensity (lux)"]),
        "wind_direction": float(row["wind_direction (°)"])
    }
    
    print(f"{weather_data}")
    
    return weather_data