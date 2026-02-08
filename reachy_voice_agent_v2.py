#!/usr/bin/env python3
"""
Voice agent using Reachy Mini's mic + speaker.
High quality audio + max volume.
"""
import sys
import signal
import json
import random
import threading
import time
import pyaudio
import struct
import numpy as np

from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation, AudioInterface
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

API_KEY = "sk_995fb89693eec0150baa8f37e91be1783a4c1e407d15b437"

with open("agents.json") as f:
    AGENTS = json.load(f)

is_speaking = False
robot = None

# Volume boost factor (1.0 = normal, 2.0 = 2x louder, etc.)
VOLUME_BOOST = 3.0


class ReachyAudioInterface(AudioInterface):
    """Audio interface using Reachy Mini's mic and speaker - HIGH QUALITY"""
    
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None
        self.running = False
        self.input_callback = None
        
        # Find Reachy audio device
        self.reachy_device = 0
        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            if "Reachy Mini Audio" in info['name']:
                self.reachy_device = i
                print(f"Using Reachy audio device [{i}]: {info['name']}")
                # Print supported sample rates
                print(f"  Max input channels: {info['maxInputChannels']}")
                print(f"  Max output channels: {info['maxOutputChannels']}")
                print(f"  Default sample rate: {info['defaultSampleRate']}")
                break
    
    def start(self, input_callback, sample_rate=16000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.input_callback = input_callback
        self.running = True
        
        print(f"Starting audio at {sample_rate}Hz...")
        
        # Input stream (mic) - use higher quality
        self.input_stream = self.p.open(
            format=pyaudio.paInt16,
            channels=2,  # Reachy has stereo mic
            rate=sample_rate,
            input=True,
            input_device_index=self.reachy_device,
            frames_per_buffer=512  # Smaller buffer for lower latency
        )
        
        # Output stream (speaker)  
        self.output_stream = self.p.open(
            format=pyaudio.paInt16,
            channels=2,  # Reachy has stereo output
            rate=sample_rate,
            output=True,
            output_device_index=self.reachy_device,
            frames_per_buffer=512
        )
        
        # Start input thread
        self.input_thread = threading.Thread(target=self._read_input, daemon=True)
        self.input_thread.start()
        print("Audio started!")
    
    def _read_input(self):
        """Read from mic and send to callback"""
        while self.running:
            try:
                data = self.input_stream.read(512, exception_on_overflow=False)
                # Convert stereo to mono by taking every other sample
                samples = struct.unpack(f'{len(data)//2}h', data)
                mono = samples[::2]  # Take left channel
                mono_data = struct.pack(f'{len(mono)}h', *mono)
                if self.input_callback:
                    self.input_callback(mono_data)
            except Exception as e:
                if self.running:
                    print(f"Input error: {e}")
                break
    
    def stop(self):
        self.running = False
        time.sleep(0.1)
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
        self.p.terminate()
    
    def output(self, audio_data: bytes):
        if self.output_stream and audio_data:
            try:
                # Convert mono to stereo AND boost volume
                samples = struct.unpack(f'{len(audio_data)//2}h', audio_data)
                
                # Boost volume
                boosted = []
                for s in samples:
                    boosted_sample = int(s * VOLUME_BOOST)
                    # Clip to prevent distortion
                    boosted_sample = max(-32768, min(32767, boosted_sample))
                    boosted.append(boosted_sample)
                
                # Convert to stereo
                stereo = []
                for s in boosted:
                    stereo.extend([s, s])  # Duplicate for L and R
                
                stereo_data = struct.pack(f'{len(stereo)}h', *stereo)
                self.output_stream.write(stereo_data)
            except Exception as e:
                print(f"Output error: {e}")
    
    def interrupt(self):
        pass


def animate_speaking():
    global is_speaking, robot
    while is_speaking and robot:
        try:
            roll = random.uniform(-5, 5)
            z = random.uniform(-3, 3)
            robot.goto_target(
                head=create_head_pose(z=z, roll=roll, degrees=True, mm=True),
                duration=0.3
            )
            time.sleep(0.25)
            if random.random() < 0.3:
                robot.goto_target(antennas=[0.2, -0.2], duration=0.15)
                robot.goto_target(antennas=[0, 0], duration=0.15)
        except:
            pass


def on_agent_response(response):
    global is_speaking
    print(f"\n🤖 Agent: {response}")
    is_speaking = True
    threading.Thread(target=animate_speaking, daemon=True).start()


def on_user_transcript(transcript):
    global is_speaking
    print(f"\n👤 You: {transcript}")
    is_speaking = False


def run_conversation(voice_name: str):
    global robot, is_speaking
    
    if voice_name not in AGENTS:
        print(f"Unknown voice: {voice_name}")
        return
    
    agent_id = AGENTS[voice_name]
    print(f"\n{'='*50}")
    print(f"🤖 FuelOS Demo with {voice_name}")
    print(f"   HIGH QUALITY + LOUD MODE")
    print(f"   Volume boost: {VOLUME_BOOST}x")
    print(f"{'='*50}")
    
    print("\nConnecting to Reachy Mini...")
    robot = ReachyMini(media_backend="no_media")
    robot.__enter__()
    robot.enable_motors()
    print("Robot connected!")
    
    # Greeting pose
    robot.goto_target(
        head=create_head_pose(z=0, roll=0, degrees=True, mm=True),
        antennas=[0.5, -0.5],
        duration=0.5
    )
    robot.goto_target(antennas=[0, 0], duration=0.3)
    
    print("\n🎤 Speak into the ROBOT's microphone!")
    print("   Press Ctrl+C to end.\n")
    
    client = ElevenLabs(api_key=API_KEY)
    audio_interface = ReachyAudioInterface()
    
    conversation = Conversation(
        client,
        agent_id,
        requires_auth=True,
        audio_interface=audio_interface,
        callback_agent_response=on_agent_response,
        callback_user_transcript=on_user_transcript,
    )
    
    def cleanup(sig, frame):
        global is_speaking
        is_speaking = False
        conversation.end_session()
    
    signal.signal(signal.SIGINT, cleanup)
    
    conversation.start_session()
    conv_id = conversation.wait_for_session_end()
    
    # Goodbye wave
    if robot:
        is_speaking = False
        robot.goto_target(antennas=[0.7, 0.7], duration=0.3)
        robot.goto_target(antennas=[0, 0], duration=0.3)
        robot.__exit__(None, None, None)
    
    print(f"\n\nConversation ended. ID: {conv_id}")


if __name__ == "__main__":
    voice = sys.argv[1] if len(sys.argv) > 1 else "Callum"
    run_conversation(voice)
