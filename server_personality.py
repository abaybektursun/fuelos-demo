#!/usr/bin/env python3
"""
FuelOS Demo Companion — Personality Server + Voice

Character: Curious young creature discovering humans
Philosophy: Stillness is presence. Movement is earned. Attention is a gift.
"""

import asyncio
import threading
import time
import math
import json
import random
import queue
import signal
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Callable, Tuple, List
from collections import deque

import numpy as np
import cv2
import struct
import pyaudio
from ultralytics import YOLO
from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation, AudioInterface, ConversationInitiationData

# =============================================================================
# VOICE CONFIG
# =============================================================================

ELEVENLABS_API_KEY = "sk_995fb89693eec0150baa8f37e91be1783a4c1e407d15b437"
VOICE_AGENTS = {
    "Callum": "agent_1201kgz84rm2fr080e4t12zsp3m8",
    "Laura": "agent_4601kgz84sg1evgt11vdcejzf3wq",
    "Jessica": "agent_2701kgz84tcvejw90b8qt1mpemdk",
}
DEFAULT_VOICE = "Callum"
VOLUME_BOOST = 3.0

# =============================================================================
# PERSONALITY CONSTANTS
# =============================================================================

# Character timing (seconds)
BREATH_CYCLE = 4.0          # Full breath in-out
GLANCE_INTERVAL = (8, 15)   # Random interval for idle glances
DROWSY_ONSET = 30.0         # Seconds before getting drowsy
NOTICE_PAUSE = 0.3          # Anticipation before tracking
LOST_PATIENCE = 2.0         # How long to search before giving up
SURPRISED_DURATION = 0.4    # Startle reaction time

# Movement amounts (radians)
BREATH_AMOUNT = math.radians(0.5)     # Subtle
GLANCE_AMOUNT = math.radians(15)      # Small look around
DROWSY_DROOP = math.radians(8)        # Head droops when sleepy
CURIOSITY_TILT = math.radians(5)      # Head tilt while tracking
STARTLE_BACK = math.radians(3)        # Micro backward movement
SEARCH_RANGE = math.radians(20)       # How far to look when searching

# Antenna positions (radians) - [right, left]
# 0 = perky/upright, negative = droop right, positive = droop left
ANTENNA_PERKY = [0.3, -0.3]           # Alert, interested
ANTENNA_NEUTRAL = [0.0, 0.0]          # Calm, present
ANTENNA_RELAXED = [-0.3, 0.3]         # At ease
ANTENNA_DROOPY = [-1.0, 1.0]          # Sleepy/sad
ANTENNA_SURPRISED = [0.6, -0.6]       # Very alert

# Tracking
SMOOTHING_ALPHA = 0.15      # Detection smoothing (lower = smoother)
MOVE_THRESHOLD = 0.08       # Radians before moving (~4.5°)
TRACKING_LAG = 0.85         # Intentional lag for organic feel

# Zone thresholds (face box height in pixels at 320x180)
# Only track close faces - ignore distant ones
TRACKING_MIN_SIZE = 35.0
ENGAGEMENT_MIN_SIZE = 55.0


# =============================================================================
# VOICE INTERFACE
# =============================================================================

class ReachyAudioInterface(AudioInterface):
    """Audio interface for Reachy Mini with pause/resume support"""
    
    # Small buffers for minimum latency
    INPUT_FRAMES = 512   # 32ms @ 16kHz - fast response
    OUTPUT_FRAMES = 512  # 32ms @ 16kHz
    
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None
        self.running = False
        self.paused = False  # Pause flag - stops sending audio without closing
        self.input_callback = None
        
        # Output queue for interrupt support (small max size for low latency)
        self.output_queue = queue.Queue(maxsize=10)
        self.output_thread = None
        
        # Find Reachy audio device
        self.reachy_device = 0
        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            if "Reachy Mini Audio" in info['name']:
                self.reachy_device = i
                print(f"[Voice] Found Reachy audio [{i}]: {info['name']}")
                break
    
    def start(self, input_callback):
        self.input_callback = input_callback
        self.running = True
        self.paused = False
        
        print("[Voice] Starting audio streams...")
        
        # Input stream (mic) - stereo from hardware
        self.input_stream = self.p.open(
            format=pyaudio.paInt16,
            channels=2,
            rate=16000,
            input=True,
            input_device_index=self.reachy_device,
            frames_per_buffer=self.INPUT_FRAMES
        )
        
        # Output stream (speaker) - stereo to hardware
        self.output_stream = self.p.open(
            format=pyaudio.paInt16,
            channels=2,
            rate=16000,
            output=True,
            output_device_index=self.reachy_device,
            frames_per_buffer=self.OUTPUT_FRAMES
        )
        
        # Start input thread
        self.input_thread = threading.Thread(target=self._read_input, daemon=True)
        self.input_thread.start()
        
        # Start output thread (for interrupt support)
        self.output_thread = threading.Thread(target=self._write_output, daemon=True)
        self.output_thread.start()
        
        print("[Voice] Audio started!")
    
    def _read_input(self):
        """Read from mic and send to callback (unless paused)"""
        while self.running:
            try:
                data = self.input_stream.read(self.INPUT_FRAMES, exception_on_overflow=False)
                # Only send if not paused and callback exists
                if not self.paused and self.input_callback:
                    # Convert stereo to mono by taking left channel
                    samples = struct.unpack(f'{len(data)//2}h', data)
                    mono = samples[::2]
                    mono_data = struct.pack(f'{len(mono)}h', *mono)
                    self.input_callback(mono_data)
            except Exception as e:
                if self.running:
                    print(f"[Voice] Input error: {e}")
                break
    
    def _write_output(self):
        """Write queued audio to speaker immediately"""
        while self.running:
            try:
                audio = self.output_queue.get(timeout=0.05)  # Fast polling
                if self.output_stream and not self.paused:
                    self.output_stream.write(audio)
            except queue.Empty:
                pass
            except Exception as e:
                if self.running:
                    print(f"[Voice] Output error: {e}")
    
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
        """Queue audio for immediate playback"""
        if audio_data and not self.paused:
            try:
                # Convert mono int16 to stereo with volume boost
                samples = struct.unpack(f'{len(audio_data)//2}h', audio_data)
                boosted = []
                for s in samples:
                    b = int(s * VOLUME_BOOST)
                    b = max(-32768, min(32767, b))
                    boosted.append(b)
                # Convert to stereo
                stereo = []
                for s in boosted:
                    stereo.extend([s, s])
                stereo_data = struct.pack(f'{len(stereo)}h', *stereo)
                # Non-blocking put - drop if queue full (prevents backup)
                try:
                    self.output_queue.put_nowait(stereo_data)
                except queue.Full:
                    pass  # Drop frame to prevent latency buildup
            except Exception as e:
                print(f"[Voice] Output error: {e}")
    
    def interrupt(self):
        """Clear output queue to stop playback immediately"""
        try:
            while True:
                self.output_queue.get_nowait()
        except queue.Empty:
            pass
    
    def pause(self):
        """Pause audio streaming (keeps streams open)"""
        self.paused = True
        self.interrupt()  # Clear any pending output
    
    def resume(self):
        """Resume audio streaming"""
        self.paused = False


class VoiceManager:
    """Manages voice conversations with 11Labs - persistent session with pause/resume"""
    
    def __init__(self, behavior_engine: 'BehaviorEngine', voice_name: str = DEFAULT_VOICE):
        self.behavior = behavior_engine
        self.client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        self.voice_name = voice_name
        self.agent_id = VOICE_AGENTS.get(voice_name)
        
        # Audio interface (created once, never destroyed)
        self.audio_interface = ReachyAudioInterface()
        
        # Conversation object
        self.conversation: Optional[Conversation] = None
        
        # State
        self.session_active = False  # WebSocket connected
        self.is_engaged = False      # Currently talking (not paused)
        self.is_speaking = False
        self.is_listening = False
        self._lock = threading.Lock()
        
        # Keep-alive thread for paused state
        self._keepalive_thread = None
        self._keepalive_stop = threading.Event()
        
        print(f"[Voice] Initialized with {voice_name}")
    
    def _create_conversation(self):
        """Create conversation object with low-latency config"""
        if not self.agent_id:
            print(f"[Voice] Unknown voice: {self.voice_name}")
            return
        
        # Low-latency configuration overrides
        config = ConversationInitiationData(
            conversation_config_override={
                "tts": {
                    "model_id": "eleven_flash_v2_5",  # Fastest TTS (~75ms)
                },
                "turn": {
                    "turn_timeout": 10,  # Shorter timeout
                    "turn_eagerness": "eager",  # Respond quickly
                },
            },
            extra_body={
                "temperature": 0.7,
                "max_tokens": 150,  # Keep responses short
            },
        )
        
        self.conversation = Conversation(
            self.client,
            self.agent_id,
            requires_auth=True,
            audio_interface=self.audio_interface,
            config=config,
            callback_agent_response=self._on_agent_response,
            callback_user_transcript=self._on_user_transcript,
        )
        print("[Voice] Created conversation with low-latency config (Flash TTS, eager turns)")
    
    def start_session(self):
        """Start the persistent session (call once at startup)"""
        with self._lock:
            if self.session_active:
                print("[Voice] Session already active")
                return
        
        self._create_conversation()
        
        try:
            print("[Voice] Starting persistent session...")
            self.conversation.start_session()
            self.session_active = True
            # Start paused - will resume when engaged
            self.audio_interface.pause()
            self._start_keepalive()
            print("[Voice] Session started (paused, waiting for engagement)")
        except Exception as e:
            print(f"[Voice] Session start error: {e}")
            self.session_active = False
    
    def _start_keepalive(self):
        """Start background thread to send user_activity during pause"""
        self._keepalive_stop.clear()
        self._keepalive_thread = threading.Thread(target=self._keepalive_loop, daemon=True)
        self._keepalive_thread.start()
    
    def _keepalive_loop(self):
        """Send user_activity every 5 seconds to prevent timeout"""
        while not self._keepalive_stop.is_set():
            if self.session_active and not self.is_engaged:
                try:
                    self.conversation.register_user_activity()
                except Exception as e:
                    print(f"[Voice] Keepalive error: {e}")
            self._keepalive_stop.wait(5.0)  # Send every 5 seconds
    
    def engage(self):
        """Resume audio streaming - user is close enough to talk"""
        with self._lock:
            if not self.session_active:
                print("[Voice] No active session")
                return
            if self.is_engaged:
                return
            self.is_engaged = True
        
        self.audio_interface.resume()
        print("[Voice] 🎤 Engaged - listening")
    
    def disengage(self):
        """Pause audio streaming - user walked away"""
        with self._lock:
            if not self.is_engaged:
                return
            self.is_engaged = False
            self.is_speaking = False
            self.is_listening = False
        
        self.audio_interface.pause()
        print("[Voice] ⏸️ Disengaged - paused (session kept alive)")
    
    def end_session(self):
        """Fully end the session (call at shutdown)"""
        self._keepalive_stop.set()
        with self._lock:
            if not self.session_active:
                return
            self.session_active = False
            self.is_engaged = False
        
        try:
            if self.conversation:
                self.conversation.end_session()
        except Exception as e:
            print(f"[Voice] End error: {e}")
        
        print("[Voice] Session ended")
    
    # Legacy aliases for compatibility
    def start_conversation(self, voice_name: str = None):
        """Alias for engage()"""
        if not self.session_active:
            self.start_session()
        self.engage()
    
    def end_conversation(self):
        """Alias for disengage() - does NOT end the session"""
        self.disengage()
    
    def _on_agent_response(self, response):
        print(f"[Voice] 🤖 Agent: {response}")
        self.is_speaking = True
        self.is_listening = False
    
    def _on_user_transcript(self, transcript):
        print(f"[Voice] 👤 User: {transcript}")
        self.is_speaking = False
        self.is_listening = True


# =============================================================================
# STATE MACHINE
# =============================================================================

class State(Enum):
    IDLE = auto()       # No one around
    NOTICING = auto()   # Just detected someone
    TRACKING = auto()   # Following face (far)
    ENGAGED = auto()    # Close conversation distance
    LOST = auto()       # Face disappeared
    SURPRISED = auto()  # New face appeared suddenly


@dataclass
class Personality:
    """Current emotional/behavioral state"""
    state: State = State.IDLE
    state_entered: float = field(default_factory=time.time)
    
    # Idle behaviors
    last_glance: float = 0
    next_glance: float = 0
    is_drowsy: bool = False
    drowsy_amount: float = 0.0
    
    # Tracking state
    target_yaw: float = 0.0
    target_pitch: float = 0.0
    current_yaw: float = 0.0
    current_pitch: float = 0.0
    last_seen_yaw: float = 0.0
    last_seen_pitch: float = 0.0
    
    # Search behavior
    search_phase: int = 0
    search_positions: List[Tuple[float, float]] = field(default_factory=list)
    
    # Antenna state
    antenna_target: List[float] = field(default_factory=lambda: [0.0, 0.0])
    antenna_current: List[float] = field(default_factory=lambda: [0.0, 0.0])
    
    # Surprise
    surprise_offset: float = 0.0
    
    # Detection
    smoothed_detection: Optional[Tuple[float, float, float]] = None  # yaw, pitch, size
    
    def state_time(self) -> float:
        """Seconds in current state"""
        return time.time() - self.state_entered
    
    def transition(self, new_state: State):
        if new_state != self.state:
            print(f"[Personality] {self.state.name} → {new_state.name}")
            self.state = new_state
            self.state_entered = time.time()


# =============================================================================
# BEHAVIOR ENGINE
# =============================================================================

class BehaviorEngine:
    """Computes personality-driven movements"""
    
    def __init__(self):
        self.personality = Personality()
        self._schedule_next_glance()
    
    def _schedule_next_glance(self):
        """Schedule random glance during idle"""
        self.personality.next_glance = time.time() + random.uniform(*GLANCE_INTERVAL)
    
    def _ease_in_out(self, t: float) -> float:
        """Smooth S-curve easing"""
        return t * t * (3 - 2 * t)
    
    def _smooth_toward(self, current: float, target: float, alpha: float) -> float:
        """Exponential smoothing"""
        return current + alpha * (target - current)
    
    def update(self, detection: Optional[Tuple[float, float, float]]) -> Tuple[float, float, float, List[float]]:
        """
        Main update - computes head pose and antenna positions.
        
        Args:
            detection: (yaw, pitch, face_size) or None if no face
            
        Returns:
            (roll, pitch, yaw, [right_antenna, left_antenna])
        """
        p = self.personality
        now = time.time()
        
        # Smooth detection to reduce jitter
        if detection:
            if p.smoothed_detection is None:
                p.smoothed_detection = detection
            else:
                p.smoothed_detection = (
                    self._smooth_toward(p.smoothed_detection[0], detection[0], SMOOTHING_ALPHA),
                    self._smooth_toward(p.smoothed_detection[1], detection[1], SMOOTHING_ALPHA),
                    self._smooth_toward(p.smoothed_detection[2], detection[2], SMOOTHING_ALPHA),
                )
            det = p.smoothed_detection
        else:
            det = None
        
        # State transitions
        self._update_state(det, now)
        
        # Compute behaviors for current state
        roll, pitch, yaw = self._compute_head_pose(det, now)
        antennas = self._compute_antennas(now)
        
        return roll, pitch, yaw, antennas
    
    def _update_state(self, det: Optional[Tuple[float, float, float]], now: float):
        """Handle state transitions"""
        p = self.personality
        
        if det is None:
            # No face detected
            if p.state in (State.TRACKING, State.ENGAGED, State.NOTICING):
                p.last_seen_yaw = p.current_yaw
                p.last_seen_pitch = p.current_pitch
                p.search_phase = 0
                p.search_positions = [
                    (p.last_seen_yaw, p.last_seen_pitch),  # Hold
                    (p.last_seen_yaw - SEARCH_RANGE * 0.5, p.last_seen_pitch),  # Look left
                    (p.last_seen_yaw + SEARCH_RANGE * 0.5, p.last_seen_pitch),  # Look right  
                    (p.last_seen_yaw, p.last_seen_pitch),  # Back to last
                    (0, 0),  # Return to center
                ]
                p.transition(State.LOST)
            elif p.state == State.LOST:
                if p.state_time() > LOST_PATIENCE:
                    p.transition(State.IDLE)
            elif p.state == State.SURPRISED:
                if p.state_time() > SURPRISED_DURATION:
                    p.transition(State.IDLE)
        else:
            yaw, pitch, size = det
            is_engaged = size >= ENGAGEMENT_MIN_SIZE
            is_tracking = size >= TRACKING_MIN_SIZE
            
            if p.state == State.IDLE:
                if is_tracking:
                    # New face! Check if sudden (surprised) or gradual (noticing)
                    p.transition(State.NOTICING)
                    p.is_drowsy = False
                    p.drowsy_amount = 0.0
                    
            elif p.state == State.NOTICING:
                if p.state_time() > NOTICE_PAUSE:
                    if is_engaged:
                        p.transition(State.ENGAGED)
                    else:
                        p.transition(State.TRACKING)
                        
            elif p.state == State.TRACKING:
                if is_engaged:
                    p.transition(State.ENGAGED)
                elif not is_tracking:
                    p.transition(State.LOST)
                    
            elif p.state == State.ENGAGED:
                if not is_engaged and is_tracking:
                    p.transition(State.TRACKING)
                elif not is_tracking:
                    p.transition(State.LOST)
                    
            elif p.state == State.LOST:
                # Found them again!
                if is_engaged:
                    p.transition(State.ENGAGED)
                elif is_tracking:
                    p.transition(State.TRACKING)
                    
            elif p.state == State.SURPRISED:
                if p.state_time() > SURPRISED_DURATION:
                    if is_engaged:
                        p.transition(State.ENGAGED)
                    elif is_tracking:
                        p.transition(State.TRACKING)
            
            # Update target
            if p.state in (State.TRACKING, State.ENGAGED, State.NOTICING):
                p.target_yaw = yaw
                p.target_pitch = pitch
    
    def _compute_head_pose(self, det: Optional[Tuple[float, float, float]], now: float) -> Tuple[float, float, float]:
        """Compute roll, pitch, yaw based on state"""
        p = self.personality
        roll = 0.0
        
        if p.state == State.IDLE:
            # Breathing
            breath_phase = (now % BREATH_CYCLE) / BREATH_CYCLE
            breath = math.sin(breath_phase * 2 * math.pi) * BREATH_AMOUNT
            
            # Drowsiness builds over time
            idle_time = p.state_time()
            if idle_time > DROWSY_ONSET:
                drowsy_progress = min(1.0, (idle_time - DROWSY_ONSET) / 20.0)
                p.drowsy_amount = self._smooth_toward(p.drowsy_amount, drowsy_progress, 0.02)
            
            # Return to center first, then random glances
            target_yaw = 0.0
            target_pitch = 0.0
            
            # Only do glances once near center
            at_center = abs(p.current_yaw) < 0.1 and abs(p.current_pitch) < 0.1
            if at_center and now > p.next_glance and not p.is_drowsy:
                # Do a glance
                target_yaw = random.uniform(-GLANCE_AMOUNT, GLANCE_AMOUNT)
                target_pitch = random.uniform(-GLANCE_AMOUNT * 0.5, GLANCE_AMOUNT * 0.5)
                self._schedule_next_glance()
            
            # Smooth movement - faster return to center (0.15 vs 0.05)
            smooth_factor = 0.15 if not at_center else 0.05
            p.current_yaw = self._smooth_toward(p.current_yaw, target_yaw, smooth_factor)
            p.current_pitch = self._smooth_toward(p.current_pitch, target_pitch, smooth_factor)
            
            pitch = p.current_pitch + breath - (p.drowsy_amount * DROWSY_DROOP)
            yaw = p.current_yaw
            
        elif p.state == State.NOTICING:
            # Hold current position with slight perk
            pitch = p.current_pitch
            yaw = p.current_yaw
            
            # Start orienting toward target
            progress = min(1.0, p.state_time() / NOTICE_PAUSE)
            eased = self._ease_in_out(progress) * 0.3  # Only 30% of the way
            yaw = self._smooth_toward(yaw, p.target_yaw * eased, 0.1)
            pitch = self._smooth_toward(pitch, p.target_pitch * eased, 0.1)
            
        elif p.state in (State.TRACKING, State.ENGAGED):
            # Follow face with intentional lag
            lag = TRACKING_LAG if p.state == State.TRACKING else 0.9
            p.current_yaw = self._smooth_toward(p.current_yaw, p.target_yaw, 1.0 - lag)
            p.current_pitch = self._smooth_toward(p.current_pitch, p.target_pitch, 1.0 - lag)
            
            yaw = p.current_yaw
            pitch = p.current_pitch
            
            # Curiosity tilt during tracking (not engaged)
            if p.state == State.TRACKING:
                tilt_phase = (now * 0.3) % 1.0
                roll = math.sin(tilt_phase * 2 * math.pi) * CURIOSITY_TILT * 0.3
            
        elif p.state == State.LOST:
            # Search behavior
            search_time = p.state_time()
            phase_duration = LOST_PATIENCE / len(p.search_positions)
            current_phase = min(int(search_time / phase_duration), len(p.search_positions) - 1)
            
            if current_phase < len(p.search_positions):
                target = p.search_positions[current_phase]
                speed = 0.08 if current_phase == len(p.search_positions) - 1 else 0.12
                p.current_yaw = self._smooth_toward(p.current_yaw, target[0], speed)
                p.current_pitch = self._smooth_toward(p.current_pitch, target[1], speed)
            
            yaw = p.current_yaw
            pitch = p.current_pitch
            
            # Confused tilt
            roll = math.sin(search_time * 2) * CURIOSITY_TILT * 0.5
            
        elif p.state == State.SURPRISED:
            # Quick startle then orient
            progress = p.state_time() / SURPRISED_DURATION
            
            if progress < 0.3:
                # Startle back
                p.surprise_offset = STARTLE_BACK * (1 - progress / 0.3)
            else:
                p.surprise_offset = 0
            
            p.current_yaw = self._smooth_toward(p.current_yaw, p.target_yaw, 0.2)
            p.current_pitch = self._smooth_toward(p.current_pitch, p.target_pitch, 0.2)
            
            yaw = p.current_yaw
            pitch = p.current_pitch - p.surprise_offset
        
        else:
            yaw = p.current_yaw
            pitch = p.current_pitch
        
        return roll, pitch, yaw
    
    def _compute_antennas(self, now: float) -> List[float]:
        """Compute antenna positions based on state"""
        p = self.personality
        
        if p.state == State.IDLE:
            if p.drowsy_amount > 0.3:
                target = ANTENNA_DROOPY
            else:
                target = ANTENNA_NEUTRAL
                
        elif p.state == State.NOTICING:
            # Perk up
            progress = min(1.0, p.state_time() / NOTICE_PAUSE)
            target = [
                ANTENNA_NEUTRAL[0] + (ANTENNA_PERKY[0] - ANTENNA_NEUTRAL[0]) * progress,
                ANTENNA_NEUTRAL[1] + (ANTENNA_PERKY[1] - ANTENNA_NEUTRAL[1]) * progress,
            ]
            
        elif p.state == State.TRACKING:
            target = ANTENNA_PERKY
            
        elif p.state == State.ENGAGED:
            # Relaxed but attentive
            target = ANTENNA_RELAXED
            
        elif p.state == State.LOST:
            # Gradual droop
            progress = min(1.0, p.state_time() / LOST_PATIENCE)
            target = [
                ANTENNA_PERKY[0] + (ANTENNA_DROOPY[0] - ANTENNA_PERKY[0]) * progress * 0.5,
                ANTENNA_PERKY[1] + (ANTENNA_DROOPY[1] - ANTENNA_PERKY[1]) * progress * 0.5,
            ]
            
        elif p.state == State.SURPRISED:
            target = ANTENNA_SURPRISED
            
        else:
            target = ANTENNA_NEUTRAL
        
        # Smooth antenna movement
        p.antenna_current[0] = self._smooth_toward(p.antenna_current[0], target[0], 0.1)
        p.antenna_current[1] = self._smooth_toward(p.antenna_current[1], target[1], 0.1)
        
        return list(p.antenna_current)
    
    def get_state_info(self) -> dict:
        """Get current state for UI"""
        p = self.personality
        return {
            "state": p.state.name,
            "state_time": round(p.state_time(), 1),
            "drowsy": round(p.drowsy_amount, 2),
            "yaw": round(math.degrees(p.current_yaw), 1),
            "pitch": round(math.degrees(p.current_pitch), 1),
            "antenna_r": round(math.degrees(p.antenna_current[0]), 1),
            "antenna_l": round(math.degrees(p.antenna_current[1]), 1),
        }


# =============================================================================
# CAMERA & DETECTION
# =============================================================================

class CameraWorker:
    """Handles camera capture and face detection"""
    
    def __init__(self, reachy):
        self.reachy = reachy
        self.behavior = BehaviorEngine()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Output
        self._output = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # x, y, z, roll, pitch, yaw
        self._antennas = [0.0, 0.0]
        self._frame: Optional[np.ndarray] = None
        self._state_info: dict = {}
        
        # Detection - download model from HuggingFace
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(
            repo_id="AdamCodd/YOLOv11n-face-detection",
            filename="model.pt"
        )
        self._model = YOLO(model_path)
        print("[Camera] YOLO model loaded")
        
        # OpenCV camera (Reachy camera is device 0)
        self._cap = cv2.VideoCapture(0)
        if self._cap.isOpened():
            print("[Camera] OpenCV camera opened")
        else:
            print("[Camera] WARNING: Could not open camera")
        
        # Camera intrinsics (Sony IMX708, 120° diagonal FOV)
        self._hfov = 120  # degrees horizontal
        self._fx = 160 / math.tan(math.radians(self._hfov / 2))
        self._fy = self._fx
        self._cx = 160  # center x
        self._cy = 90   # center y
        
        # Callbacks
        self._engagement_callback: Optional[Callable[[bool], None]] = None
        self._was_engaged = False
        self._engaged_since = 0.0
        self._disengaged_since = 0.0
        self._voice_active = False
    
    def _capture_frame(self) -> Optional[np.ndarray]:
        """Capture frame from OpenCV camera"""
        if self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret:
                return frame
        return None
    
    def set_engagement_callback(self, callback: Callable[[bool], None]):
        """Set callback for engagement state changes"""
        self._engagement_callback = callback
    
    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[Camera] Started")
    
    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()
    
    def get_output(self) -> Tuple[float, ...]:
        with self._lock:
            return self._output
    
    def get_antennas(self) -> List[float]:
        with self._lock:
            return list(self._antennas)
    
    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None
    
    def get_state_info(self) -> dict:
        with self._lock:
            return dict(self._state_info)
    
    def _loop(self):
        period = 1.0 / 30.0  # 30 Hz
        
        while not self._stop.is_set():
            start = time.perf_counter()
            
            try:
                # Capture frame from OpenCV camera
                frame = self._capture_frame()
                if frame is None:
                    time.sleep(period)
                    continue
                
                # Resize for processing
                small = cv2.resize(frame, (320, 180))
                
                # Detect faces
                results = self._model(small, verbose=False)
                detection = None
                
                if len(results) > 0 and len(results[0].boxes) > 0:
                    # Get largest face
                    boxes = results[0].boxes
                    areas = [(b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]) for b in boxes]
                    best_idx = int(np.argmax(areas))
                    box = boxes[best_idx].xyxy[0].cpu().numpy()
                    
                    # Face center and size
                    cx = (box[0] + box[2]) / 2
                    cy = (box[1] + box[3]) / 2
                    size = box[3] - box[1]  # Height
                    
                    # Convert to angles
                    yaw = -math.atan2(cx - self._cx, self._fx)
                    pitch = math.atan2(cy - self._cy, self._fy)
                    
                    detection = (yaw, pitch, float(size))
                    
                    # Draw on frame
                    color = (0, 255, 0) if size >= ENGAGEMENT_MIN_SIZE else (0, 255, 255)
                    cv2.rectangle(small, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                
                # Update behavior
                roll, pitch, yaw, antennas = self.behavior.update(detection)
                
                # Check engagement callback with hysteresis
                is_engaged = self.behavior.personality.state == State.ENGAGED
                now = time.time()
                
                if is_engaged:
                    self._engaged_since = now
                    # Start voice after 0.5s of sustained engagement
                    if not self._voice_active and (now - self._disengaged_since) > 0.5:
                        self._voice_active = True
                        if self._engagement_callback:
                            try:
                                self._engagement_callback(True)
                            except Exception as e:
                                print(f"[Callback] Error: {e}")
                else:
                    self._disengaged_since = now
                    # Only stop voice after 3s of no engagement (prevents jitter)
                    if self._voice_active and (now - self._engaged_since) > 3.0:
                        self._voice_active = False
                        if self._engagement_callback:
                            try:
                                self._engagement_callback(False)
                            except Exception as e:
                                print(f"[Callback] Error: {e}")
                
                # Store output
                with self._lock:
                    self._output = (0.0, 0.0, 0.0, roll, pitch, yaw)
                    self._antennas = antennas
                    self._frame = small
                    self._state_info = self.behavior.get_state_info()
                
            except Exception as e:
                print(f"[Camera] Error: {e}")
            
            elapsed = time.perf_counter() - start
            if elapsed < period:
                time.sleep(period - elapsed)


# =============================================================================
# MOVEMENT MANAGER
# =============================================================================

class MovementManager:
    """Applies computed movements to robot — body moves organically"""
    
    def __init__(self, reachy, camera_worker: CameraWorker):
        self.reachy = reachy
        self.camera_worker = camera_worker
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        
        from reachy_mini.utils import create_head_pose
        from reachy_mini.utils.interpolation import compose_world_offset
        self._create = create_head_pose
        self._compose = compose_world_offset
        self._neutral = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
        
        # Body state — lazy, organic "settling" movement
        self.body_yaw = 0.0
        self.attention_accumulator = 0.0  # Builds up when looking one direction
        self.last_settle_time = time.time()
    
    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[Movement] Started (30Hz)")
    
    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()
    
    def _loop(self):
        period = 1.0 / 30.0
        
        while not self._stop.is_set():
            start = time.perf_counter()
            
            try:
                off = self.camera_worker.get_output()
                antennas = self.camera_worker.get_antennas()
                head_yaw = off[5]
                
                # Head moves directly
                pose = self._create(
                    x=off[0], y=off[1], z=off[2],
                    roll=off[3], pitch=off[4], yaw=head_yaw,
                    degrees=False, mm=False
                )
                combined = self._compose(self._neutral, pose, reorthonormalize=True)
                
                # Body: organic "settling" — like creature shifting weight
                # Accumulate attention direction (very slowly)
                self.attention_accumulator += head_yaw * 0.001
                self.attention_accumulator *= 0.998  # Slow decay
                self.attention_accumulator = max(-0.8, min(0.8, self.attention_accumulator))
                
                # Body follows accumulated attention (glacier slow)
                self.body_yaw += (self.attention_accumulator - self.body_yaw) * 0.005
                
                self.reachy.set_target(head=combined, antennas=antennas, body_yaw=self.body_yaw)
                
            except Exception as e:
                pass
            
            elapsed = time.perf_counter() - start
            if elapsed < period:
                time.sleep(period - elapsed)


# =============================================================================
# WEB UI
# =============================================================================

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import uvicorn
import base64

app = FastAPI()
camera_worker: Optional[CameraWorker] = None
voice_manager: Optional[VoiceManager] = None

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>FuelOS Personality</title>
    <style>
        body { font-family: system-ui; background: #111; color: #fff; margin: 0; padding: 20px; }
        h1 { color: #0cf; margin-bottom: 5px; }
        .subtitle { color: #666; font-size: 14px; margin-bottom: 20px; font-style: italic; }
        .container { display: flex; gap: 20px; flex-wrap: wrap; }
        .video { flex: 1; min-width: 400px; }
        .status { flex: 1; min-width: 300px; max-width: 400px; }
        img { width: 100%; border-radius: 8px; }
        .group { background: #222; border-radius: 8px; padding: 15px; margin-bottom: 15px; }
        .group h3 { margin: 0 0 10px 0; color: #0cf; font-size: 14px; }
        .state { font-size: 32px; font-weight: bold; margin: 10px 0; }
        .row { display: flex; justify-content: space-between; margin: 5px 0; font-size: 13px; }
        .row .label { color: #888; }
        .row .value { color: #0cf; }
        .states-legend { display: flex; gap: 8px; flex-wrap: wrap; margin: 10px 0; }
        .state-item { padding: 5px 10px; border-radius: 4px; font-size: 11px; opacity: 0.5; }
        .state-item.active { opacity: 1; }
        .philosophy { background: #1a1a1a; border-left: 3px solid #0cf; padding: 10px; font-size: 11px; color: #666; }
        .antenna-vis { display: flex; justify-content: center; gap: 40px; margin: 20px 0; }
        .antenna { width: 4px; height: 40px; background: #0cf; border-radius: 2px; transform-origin: bottom center; }
    </style>
</head>
<body>
    <h1>🌟 FuelOS Personality</h1>
    <p class="subtitle">Curious young creature discovering humans</p>
    <div class="philosophy">
        <strong>Philosophy:</strong> Stillness is presence. Movement is earned. Attention is a gift.
    </div>
    <div class="states-legend">
        <span class="state-item" id="s-IDLE" style="background:#808080">IDLE</span>
        <span class="state-item" id="s-NOTICING" style="background:#FFA500">NOTICING</span>
        <span class="state-item" id="s-TRACKING" style="background:#FFFF00;color:#000">TRACKING</span>
        <span class="state-item" id="s-ENGAGED" style="background:#00FF00;color:#000">ENGAGED</span>
        <span class="state-item" id="s-LOST" style="background:#FF6666">LOST</span>
        <span class="state-item" id="s-SURPRISED" style="background:#FF00FF">SURPRISED</span>
    </div>
    <div class="container">
        <div class="video">
            <img id="frame" src="" alt="Camera feed">
        </div>
        <div class="status">
            <div class="group">
                <h3>State</h3>
                <div class="state" id="current-state">IDLE</div>
                <div class="row"><span class="label">Time in state</span><span class="value" id="state-time">0s</span></div>
                <div class="row"><span class="label">Drowsiness</span><span class="value" id="drowsy">0%</span></div>
            </div>
            <div class="group">
                <h3>Head</h3>
                <div class="row"><span class="label">Yaw</span><span class="value" id="yaw">0°</span></div>
                <div class="row"><span class="label">Pitch</span><span class="value" id="pitch">0°</span></div>
            </div>
            <div class="group">
                <h3>Antennas</h3>
                <div class="antenna-vis">
                    <div class="antenna" id="antenna-r"></div>
                    <div class="antenna" id="antenna-l"></div>
                </div>
                <div class="row"><span class="label">Right</span><span class="value" id="ant-r">0°</span></div>
                <div class="row"><span class="label">Left</span><span class="value" id="ant-l">0°</span></div>
            </div>
        </div>
    </div>
    <script>
        const ws = new WebSocket(`ws://${location.host}/ws`);
        ws.onmessage = (e) => {
            const d = JSON.parse(e.data);
            if (d.frame) document.getElementById('frame').src = 'data:image/jpeg;base64,' + d.frame;
            if (d.state) {
                const s = d.state;
                document.getElementById('current-state').textContent = s.state;
                document.getElementById('current-state').style.color = getStateColor(s.state);
                document.getElementById('state-time').textContent = s.state_time + 's';
                document.getElementById('drowsy').textContent = Math.round(s.drowsy * 100) + '%';
                document.getElementById('yaw').textContent = s.yaw + '°';
                document.getElementById('pitch').textContent = s.pitch + '°';
                document.getElementById('ant-r').textContent = s.antenna_r + '°';
                document.getElementById('ant-l').textContent = s.antenna_l + '°';
                
                // Update antenna visualization
                document.getElementById('antenna-r').style.transform = `rotate(${-s.antenna_r}deg)`;
                document.getElementById('antenna-l').style.transform = `rotate(${-s.antenna_l}deg)`;
                
                // Highlight active state
                document.querySelectorAll('.state-item').forEach(el => el.classList.remove('active'));
                const stateEl = document.getElementById('s-' + s.state);
                if (stateEl) stateEl.classList.add('active');
            }
        };
        
        function getStateColor(state) {
            const colors = {
                'IDLE': '#808080',
                'NOTICING': '#FFA500',
                'TRACKING': '#FFFF00',
                'ENGAGED': '#00FF00',
                'LOST': '#FF6666',
                'SURPRISED': '#FF00FF'
            };
            return colors[state] || '#fff';
        }
    </script>
</body>
</html>
"""

@app.get("/")
async def index():
    return HTMLResponse(HTML)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Client connected")
    
    try:
        while True:
            frame = camera_worker.get_frame() if camera_worker else None
            state = camera_worker.get_state_info() if camera_worker else {}
            
            data = {"state": state}
            
            if frame is not None:
                _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                data["frame"] = base64.b64encode(jpg).decode('utf-8')
            
            await websocket.send_json(data)
            await asyncio.sleep(0.05)
            
    except Exception as e:
        print(f"[WS] Error: {e}")


@app.on_event("startup")
async def startup():
    global camera_worker, voice_manager
    
    print("=" * 60)
    print("FuelOS PERSONALITY SERVER + VOICE")
    print("=" * 60)
    print("Character: Curious young creature discovering humans")
    print("Philosophy: Stillness is presence. Movement is earned.")
    print("=" * 60)
    
    from reachy_mini import ReachyMini
    
    # Use no_media so pyaudio can access the audio device
    reachy = ReachyMini(media_backend="no_media")
    print("[Main] Reachy connected (no_media for voice)")
    
    camera_worker = CameraWorker(reachy)
    camera_worker.start()
    
    movement = MovementManager(reachy, camera_worker)
    movement.start()
    
    # Voice manager - persistent session with pause/resume
    voice_manager = VoiceManager(camera_worker.behavior)
    voice_manager.start_session()  # Start session once, keep alive forever
    
    def on_engagement(engaged: bool):
        if engaged:
            print("[Main] 🎯 ENGAGED — resuming voice!")
            try:
                voice_manager.engage()
            except Exception as e:
                print(f"[Main] Voice error: {e}")
        else:
            print("[Main] Disengaged — pausing voice")
            try:
                voice_manager.disengage()
            except Exception as e:
                print(f"[Main] Voice pause error: {e}")
    
    camera_worker.set_engagement_callback(on_engagement)
    
    print("[Main] Ready — http://0.0.0.0:8080")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
