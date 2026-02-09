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
import struct
import signal
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Callable, Tuple, List
from collections import deque

import numpy as np
import cv2
import pyaudio
from ultralytics import YOLO
from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation, AudioInterface

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
    """Audio interface using Reachy Mini's mic and speaker"""
    
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
                print(f"[Voice] Using Reachy audio device [{i}]: {info['name']}")
                break
    
    def start(self, input_callback, sample_rate=16000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.input_callback = input_callback
        self.running = True
        
        print(f"[Voice] Starting audio at {sample_rate}Hz...")
        
        self.input_stream = self.p.open(
            format=pyaudio.paInt16,
            channels=2,
            rate=sample_rate,
            input=True,
            input_device_index=self.reachy_device,
            frames_per_buffer=512
        )
        
        self.output_stream = self.p.open(
            format=pyaudio.paInt16,
            channels=2,
            rate=sample_rate,
            output=True,
            output_device_index=self.reachy_device,
            frames_per_buffer=512
        )
        
        self.input_thread = threading.Thread(target=self._read_input, daemon=True)
        self.input_thread.start()
        print("[Voice] Audio started!")
    
    def _read_input(self):
        while self.running:
            try:
                data = self.input_stream.read(512, exception_on_overflow=False)
                samples = struct.unpack(f'{len(data)//2}h', data)
                mono = samples[::2]
                mono_data = struct.pack(f'{len(mono)}h', *mono)
                if self.input_callback:
                    self.input_callback(mono_data)
            except Exception as e:
                if self.running:
                    print(f"[Voice] Input error: {e}")
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
                samples = struct.unpack(f'{len(audio_data)//2}h', audio_data)
                boosted = []
                for s in samples:
                    boosted_sample = int(s * VOLUME_BOOST)
                    boosted_sample = max(-32768, min(32767, boosted_sample))
                    boosted.append(boosted_sample)
                stereo = []
                for s in boosted:
                    stereo.extend([s, s])
                stereo_data = struct.pack(f'{len(stereo)}h', *stereo)
                self.output_stream.write(stereo_data)
            except Exception as e:
                print(f"[Voice] Output error: {e}")
    
    def interrupt(self):
        pass


class VoiceManager:
    """Manages voice conversations with 11Labs"""
    
    def __init__(self, behavior_engine: 'BehaviorEngine'):
        self.behavior = behavior_engine
        self.client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        self.conversation: Optional[Conversation] = None
        self.audio_interface: Optional[ReachyAudioInterface] = None
        self.is_speaking = False
        self.is_listening = False
        self.in_conversation = False
        self._lock = threading.Lock()
    
    def start_conversation(self, voice_name: str = DEFAULT_VOICE):
        """Start a voice conversation"""
        with self._lock:
            if self.in_conversation:
                print("[Voice] Already in conversation")
                return
            
            agent_id = VOICE_AGENTS.get(voice_name)
            if not agent_id:
                print(f"[Voice] Unknown voice: {voice_name}")
                return
            
            print(f"[Voice] Starting conversation with {voice_name}...")
            self.in_conversation = True
        
        try:
            self.audio_interface = ReachyAudioInterface()
            self.conversation = Conversation(
                self.client,
                agent_id,
                requires_auth=True,
                audio_interface=self.audio_interface,
                callback_agent_response=self._on_agent_response,
                callback_user_transcript=self._on_user_transcript,
            )
            self.conversation.start_session()
            print("[Voice] Conversation started!")
        except Exception as e:
            print(f"[Voice] Error starting conversation: {e}")
            self.in_conversation = False
    
    def end_conversation(self):
        """End the current conversation"""
        with self._lock:
            if not self.in_conversation:
                return
            self.in_conversation = False
            self.is_speaking = False
            self.is_listening = False
        
        if self.conversation:
            try:
                self.conversation.end_session()
            except:
                pass
        print("[Voice] Conversation ended")
    
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
            
            # Random glances
            target_yaw = 0.0
            target_pitch = 0.0
            if now > p.next_glance and not p.is_drowsy:
                # Do a glance
                target_yaw = random.uniform(-GLANCE_AMOUNT, GLANCE_AMOUNT)
                target_pitch = random.uniform(-GLANCE_AMOUNT * 0.5, GLANCE_AMOUNT * 0.5)
                self._schedule_next_glance()
            
            # Smooth movement
            p.current_yaw = self._smooth_toward(p.current_yaw, target_yaw, 0.05)
            p.current_pitch = self._smooth_toward(p.current_pitch, target_pitch, 0.05)
            
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
        
        # Camera intrinsics (Sony IMX708, 120° FOV at 320x180)
        self._fx = 160 / math.tan(math.radians(60))
        self._fy = self._fx
        self._cx = 160
        self._cy = 90
        
        # Callbacks
        self._engagement_callback: Optional[Callable[[bool], None]] = None
        self._was_engaged = False
    
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
                # Capture frame
                frame = self.reachy.media.get_frame()
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
                
                # Check engagement callback
                is_engaged = self.behavior.personality.state == State.ENGAGED
                if is_engaged != self._was_engaged:
                    self._was_engaged = is_engaged
                    if self._engagement_callback:
                        try:
                            self._engagement_callback(is_engaged)
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
        
        # Body state — lazy, organic movement
        self.body_yaw = 0.0
        self.body_drift = 0.0  # Slow accumulated drift toward attention
    
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
                
                # Head moves directly — no compensation
                pose = self._create(
                    x=off[0], y=off[1], z=off[2],
                    roll=off[3], pitch=off[4], yaw=off[5],
                    degrees=False, mm=False
                )
                
                combined = self._compose(self._neutral, pose, reorthonormalize=True)
                
                # Body tracking disabled for now
                self.reachy.set_target(head=combined, antennas=antennas)
                
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
    
    reachy = ReachyMini(media_backend="default")
    print("[Main] Reachy connected")
    
    camera_worker = CameraWorker(reachy)
    camera_worker.start()
    
    movement = MovementManager(reachy, camera_worker)
    movement.start()
    
    # Voice manager (disabled for now - pyaudio crash)
    # voice_manager = VoiceManager(camera_worker.behavior)
    
    # Hook engagement callback
    def on_engagement(engaged: bool):
        if engaged:
            print("[Main] 🎯 ENGAGED! (voice disabled for now)")
        else:
            print("[Main] Engagement ended")
    
    camera_worker.set_engagement_callback(on_engagement)
    
    print("[Main] Ready — http://0.0.0.0:8080")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
