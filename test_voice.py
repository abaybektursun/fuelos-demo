#!/usr/bin/env python3
"""
Test ElevenLabs voice agents one by one.
Usage: python test_voice.py [voice_name]

Available voices: Callum, Laura, Jessica, Will, Chris
"""
import os
import sys
import signal
import json

from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface

API_KEY = "sk_995fb89693eec0150baa8f37e91be1783a4c1e407d15b437"

# Load agents
with open("agents.json") as f:
    AGENTS = json.load(f)

def run_conversation(voice_name: str):
    if voice_name not in AGENTS:
        print(f"Unknown voice: {voice_name}")
        print(f"Available: {', '.join(AGENTS.keys())}")
        return
    
    agent_id = AGENTS[voice_name]
    print(f"\n{'='*50}")
    print(f"Testing voice: {voice_name}")
    print(f"Agent ID: {agent_id}")
    print(f"{'='*50}")
    print("\nSpeak into your mic. Press Ctrl+C to end.\n")
    
    client = ElevenLabs(api_key=API_KEY)
    
    conversation = Conversation(
        client,
        agent_id,
        requires_auth=True,
        audio_interface=DefaultAudioInterface(),
        callback_agent_response=lambda r: print(f"\n🤖 {voice_name}: {r}"),
        callback_user_transcript=lambda t: print(f"\n👤 You: {t}"),
    )
    
    conversation.start_session()
    signal.signal(signal.SIGINT, lambda s, f: conversation.end_session())
    
    conv_id = conversation.wait_for_session_end()
    print(f"\n\nConversation ended. ID: {conv_id}")

def main():
    if len(sys.argv) < 2:
        print("FuelOS Demo Voice Tester")
        print("=" * 40)
        print("\nAvailable voices:")
        for name in AGENTS:
            print(f"  - {name}")
        print(f"\nUsage: python {sys.argv[0]} <voice_name>")
        print(f"Example: python {sys.argv[0]} Callum")
        return
    
    voice_name = sys.argv[1]
    run_conversation(voice_name)

if __name__ == "__main__":
    main()
