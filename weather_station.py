from pymodbus.client.sync import ModbusSerialClient
from datetime import datetime
import sys

sys.stdout.reconfigure(encoding='utf-8')

PORT = "/dev/ttyUSB0"         
BAUD_RATE = 4800      
SENSOR_ID = 1         

client = ModbusSerialClient(
    method='rtu',
    port=PORT,
    baudrate=BAUD_RATE,
    bytesize=8,
    parity='N',
    stopbits=1,
    timeout=1
)

if not client.connect():
    print("Failed to connect!")
    exit()


def read_register(address, scale=1.0, signed=False):
    """Read one Modbus register."""
    result = client.read_holding_registers(
        address=address, count=1, unit=SENSOR_ID)
    if result.isError():
        return None
    raw = result.registers[0]
    if signed and raw & 0x8000:
        raw = raw - 0x10000
    return raw / scale


def get_weather_data():
    """Reads all weather parameters and returns a dictionary in RAG format."""
    weather_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "temperature": read_register(0x01F9, scale=10.0, signed=True),
        "humidity": read_register(0x01F8, scale=10.0),
        "rainfall": read_register(0x0201, scale=10.0),
        "wind_speed": read_register(0x01F4, scale=100.0),
        "pressure": read_register(0x01FD),
        "light_intensity": read_register(0x0200),
        "wind_direction": read_register(0x01F7)
    }
    return weather_data


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    weather_data = get_weather_data()

    print(f"\n=== Weather Station Data @ {timestamp} ===")
    for k, v in weather_data.items():
        if k == "temperature":
            print(f"{k.replace('_', ' ').title()}: {v} °C")
        elif k == "humidity":
            print(f"{k.replace('_', ' ').title()}: {v} %")
        elif k == "rainfall":
            print(f"{k.replace('_', ' ').title()}: {v} mm")
        elif k == "pressure":
            print(f"{k.replace('_', ' ').title()}: {v} hPa")
        elif k == "wind_speed":
            print(f"{k.replace('_', ' ').title()}: {v} m/s")
        elif k == "wind_direction":
            print(f"{k.replace('_', ' ').title()}: {v} °")
        elif k == "light_intensity":
            print(f"{k.replace('_', ' ').title()}: {v} lux")
    print("=" * 50)

    client.close()
