#!/usr/bin/env python3
"""
Run voice agent with Reachy Mini animation.
Audio through Mac, robot moves expressively while speaking.
"""
import os
import sys
import signal
import json
import random
import threading
import time

from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
import numpy as np

API_KEY = "sk_995fb89693eec0150baa8f37e91be1783a4c1e407d15b437"

with open("agents.json") as f:
    AGENTS = json.load(f)

# Robot animation state
is_speaking = False
robot = None

def animate_speaking():
    """Subtle movements while speaking"""
    global is_speaking, robot
    while is_speaking and robot:
        try:
            # Small head movements
            roll = random.uniform(-5, 5)
            z = random.uniform(-3, 3)
            robot.goto_target(
                head=create_head_pose(z=z, roll=roll, degrees=True, mm=True),
                duration=0.3
            )
            time.sleep(0.25)
            
            # Occasional antenna wiggle
            if random.random() < 0.3:
                robot.goto_target(antennas=[0.2, -0.2], duration=0.15)
                robot.goto_target(antennas=[0, 0], duration=0.15)
        except:
            pass

def animate_listening():
    """Attentive pose while listening"""
    global robot
    if robot:
        try:
            robot.goto_target(
                head=create_head_pose(z=5, degrees=True, mm=True),
                antennas=[0.3, 0.3],
                duration=0.5
            )
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
    animate_listening()

def run_conversation(voice_name: str):
    global robot, is_speaking
    
    if voice_name not in AGENTS:
        print(f"Unknown voice: {voice_name}")
        return
    
    agent_id = AGENTS[voice_name]
    print(f"\n{'='*50}")
    print(f"🤖 FuelOS Demo with {voice_name}")
    print(f"{'='*50}")
    
    # Connect to robot
    print("\nConnecting to Reachy Mini...")
    robot = ReachyMini(media_backend="no_media")
    robot.__enter__()
    robot.enable_motors()
    print("Robot connected!")
    
    # Initial greeting pose
    robot.goto_target(
        head=create_head_pose(z=0, roll=0, degrees=True, mm=True),
        antennas=[0.5, -0.5],
        duration=0.5
    )
    robot.goto_target(antennas=[0, 0], duration=0.3)
    
    print("\nSpeak into your mic. Press Ctrl+C to end.\n")
    
    client = ElevenLabs(api_key=API_KEY)
    
    conversation = Conversation(
        client,
        agent_id,
        requires_auth=True,
        audio_interface=DefaultAudioInterface(),
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
        robot.goto_target(antennas=[0.7, 0.7], duration=0.3)
        robot.goto_target(antennas=[0, 0], duration=0.3)
        robot.__exit__(None, None, None)
    
    print(f"\n\nConversation ended. ID: {conv_id}")

if __name__ == "__main__":
    voice = sys.argv[1] if len(sys.argv) > 1 else "Callum"
    run_conversation(voice)
