import edge_tts
import asyncio
import subprocess

async def main():

    text = """
            Hello! This is a test of Edge-TTS with lower volume, faster speed and higher pitch.
"""
    communicate = edge_tts.Communicate(text,"en-US-AriaNeural")
    await communicate.save("out.wav")

asyncio.run(main())
subprocess.run(["amixer", "set", "Master", "50%"])
subprocess.run(["ffplay", "-nodisp", "-autoexit", "out.wav"])