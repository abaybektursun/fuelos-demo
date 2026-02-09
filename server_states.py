#!/usr/bin/env python3
"""
FuelOS Demo - Proper State Machine
===================================
Key insight: Stillness is the default. Movement is an EVENT.

States:
- IDLE: No face, at neutral, occasional wander
- NOTICING: Just saw face, brief pause before moving
- MOVING: Actively interpolating to target (fast, committed)
- HOLDING: At target, STILL, ignoring minor face jitter
- LOST: Face disappeared, brief search before returning to idle

The robot should spend most of its time HOLDING, not constantly moving.
"""

import threading
import time
import base64
import asyncio
import math
import random
import cv2
import numpy as np
from typing import Tuple, Optional, List
from enum import Enum
from scipy.spatial.transform import Rotation as R

# =============================================================================
# STATES
# =============================================================================

class State(Enum):
    IDLE = "idle"           # No face, neutral position
    NOTICING = "noticing"   # Just saw face, brief pause
    MOVING = "moving"       # Interpolating to target
    HOLDING = "holding"     # At target, STILL
    LOST = "lost"           # Face gone, searching


# =============================================================================
# CONFIG
# =============================================================================

class Config:
    # Movement thresholds
    major_threshold: float = 0.10      # Face must move this much to trigger new move (radians, ~5.7°)
    minor_threshold: float = 0.04      # Below this, completely ignore
    
    # Timing
    notice_duration: float = 0.08      # Pause before first move (seconds)
    move_duration: float = 0.25        # How long a movement takes
    lost_timeout: float = 1.2          # How long before returning to idle
    
    # Detection smoothing (just for noise reduction, not for motion)
    detection_alpha: float = 0.4       # Higher = more responsive to detection
    
    # Scale (1.0 = fully center on face)
    motion_scale: float = 1.0
    max_pitch: float = 25.0
    max_yaw: float = 40.0
    max_roll: float = 12.0
    
    # Life
    breathing_amount: float = 0.003
    breathing_speed: float = 0.2
    
    # Wander
    wander_interval: float = 3.0
    wander_amount: float = 0.05
    
    _lock = threading.Lock()
    
    @classmethod
    def get_all(cls) -> dict:
        with cls._lock:
            return {k: getattr(cls, k) for k in dir(cls) 
                    if not k.startswith('_') and isinstance(getattr(cls, k), (int, float, bool))}
    
    @classmethod
    def set(cls, key: str, value) -> None:
        with cls._lock:
            if hasattr(cls, key):
                setattr(cls, key, value)


# =============================================================================
# STATE MACHINE CONTROLLER
# =============================================================================

class StateMachine:
    def __init__(self):
        self.state = State.IDLE
        self.state_entered = time.perf_counter()
        
        # Detection (smoothed for noise reduction only)
        self.det_smoothed = [0.0] * 6
        self.det_initialized = False
        
        # Positions
        self.current_pos = [0.0] * 6     # Where we ARE
        self.committed_pos = [0.0] * 6   # Where we committed to go
        self.move_start_pos = [0.0] * 6  # Where movement started
        self.move_start_time = 0.0
        
        # Face tracking
        self.last_face_pos = [0.0] * 6   # Last known face position
        self.last_face_time = 0.0
        self.has_face = False
        
        # Life
        self.breath_phase = random.random() * math.pi * 2
        self.last_wander_time = time.perf_counter()
        
        self._lock = threading.Lock()
    
    def _enter_state(self, new_state: State) -> None:
        """Transition to a new state"""
        self.state = new_state
        self.state_entered = time.perf_counter()
    
    def _state_time(self) -> float:
        """Time spent in current state"""
        return time.perf_counter() - self.state_entered
    
    def _distance(self, a: List[float], b: List[float]) -> float:
        """Distance between two poses (rotation-weighted)"""
        d = 0.0
        for i in range(3):
            d += (a[i] - b[i]) ** 2 * 0.1  # Translation (less weight)
        for i in range(3, 6):
            d += (a[i] - b[i]) ** 2  # Rotation (more weight)
        return math.sqrt(d)
    
    def _apply_limits(self, raw: List[float]) -> List[float]:
        """Apply scale and limits"""
        scale = Config.motion_scale
        max_p = math.radians(Config.max_pitch)
        max_y = math.radians(Config.max_yaw)
        max_r = math.radians(Config.max_roll)
        
        return [
            np.clip(raw[0] * scale, -0.05, 0.05),
            np.clip(raw[1] * scale, -0.05, 0.05),
            np.clip(raw[2] * scale, -0.05, 0.05),
            np.clip(raw[3] * scale, -max_r, max_r),
            np.clip(raw[4] * scale, -max_p, max_p),
            np.clip(raw[5] * scale, -max_y, max_y)
        ]
    
    def _start_move(self, target: List[float]) -> None:
        """Start a movement to target"""
        self.move_start_pos = list(self.current_pos)
        self.committed_pos = list(target)
        self.move_start_time = time.perf_counter()
        self._enter_state(State.MOVING)
    
    def update_face(self, raw: List[float]) -> None:
        """Called when face detected"""
        with self._lock:
            now = time.perf_counter()
            
            # Smooth detection for noise reduction
            alpha = Config.detection_alpha
            if not self.det_initialized:
                self.det_smoothed = list(raw)
                self.det_initialized = True
            else:
                for i in range(6):
                    self.det_smoothed[i] = alpha * raw[i] + (1 - alpha) * self.det_smoothed[i]
            
            face_pos = self._apply_limits(self.det_smoothed)
            self.last_face_pos = face_pos
            self.last_face_time = now
            self.has_face = True
            
            # State transitions based on current state
            if self.state == State.IDLE:
                # Just saw a face! Start noticing
                self._enter_state(State.NOTICING)
                
            elif self.state == State.NOTICING:
                # Still noticing, wait for notice_duration
                if self._state_time() >= Config.notice_duration:
                    self._start_move(face_pos)
                    
            elif self.state == State.MOVING:
                # Already moving - check if face moved A LOT
                dist = self._distance(face_pos, self.committed_pos)
                if dist > Config.major_threshold * 1.5:
                    # Interrupt current move with new target
                    self._start_move(face_pos)
                    
            elif self.state == State.HOLDING:
                # HOLDING is key - only move if face moved significantly
                dist = self._distance(face_pos, self.committed_pos)
                if dist > Config.major_threshold:
                    self._start_move(face_pos)
                # Otherwise: DO NOTHING. Stay still. This is the key!
                
            elif self.state == State.LOST:
                # Found face again!
                self._start_move(face_pos)
    
    def update_no_face(self) -> None:
        """Called when no face detected"""
        with self._lock:
            now = time.perf_counter()
            self.has_face = False
            
            if self.state in (State.HOLDING, State.MOVING, State.NOTICING):
                self._enter_state(State.LOST)
                
            elif self.state == State.LOST:
                if self._state_time() >= Config.lost_timeout:
                    # Return to neutral
                    self._start_move([0.0] * 6)
                    # After this move completes, will go to IDLE
    
    def update(self) -> Tuple[float, ...]:
        """Called at control rate - returns current position"""
        with self._lock:
            now = time.perf_counter()
            
            # Handle state-specific logic
            if self.state == State.MOVING:
                # Interpolate toward committed position
                elapsed = now - self.move_start_time
                duration = Config.move_duration
                
                if elapsed >= duration:
                    # Movement complete
                    self.current_pos = list(self.committed_pos)
                    if self.has_face:
                        self._enter_state(State.HOLDING)
                    else:
                        self._enter_state(State.IDLE)
                else:
                    # Interpolate with smooth easing
                    t = elapsed / duration
                    # Sine ease-out for natural deceleration
                    ease = math.sin(t * math.pi / 2)
                    
                    for i in range(6):
                        self.current_pos[i] = (
                            self.move_start_pos[i] + 
                            (self.committed_pos[i] - self.move_start_pos[i]) * ease
                        )
            
            elif self.state == State.IDLE:
                # Wander occasionally
                if now - self.last_wander_time > Config.wander_interval:
                    self.last_wander_time = now
                    wander_target = [
                        0, 0, 0, 0,
                        random.uniform(-Config.wander_amount, Config.wander_amount),
                        random.uniform(-Config.wander_amount, Config.wander_amount)
                    ]
                    self._start_move(wander_target)
            
            # Apply breathing (subtle, always on)
            self.breath_phase += (1.0 / 30.0) * Config.breathing_speed * 2 * math.pi
            breath = math.sin(self.breath_phase) * Config.breathing_amount
            
            output = list(self.current_pos)
            output[2] += breath
            output[4] += breath * 0.3
            
            return tuple(output)
    
    def get_state(self) -> str:
        return self.state.value


# =============================================================================
# CAMERA WORKER
# =============================================================================

class CameraWorker:
    def __init__(self, reachy):
        self.reachy = reachy
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._machine = StateMachine()
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        
    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[Camera] Started")
    
    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()
    
    def get_output(self) -> Tuple[float, ...]:
        return self._machine.update()
    
    def get_state(self) -> str:
        return self._machine.get_state()
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            return self._latest_frame
    
    def _loop(self) -> None:
        from ultralytics import YOLO
        from huggingface_hub import hf_hub_download
        
        model_path = hf_hub_download(
            repo_id="AdamCodd/YOLOv11n-face-detection",
            filename="model.pt"
        )
        model = YOLO(model_path)
        print("[Camera] YOLO loaded")
        
        period = 1.0 / 20.0
        
        while not self._stop.is_set():
            start = time.perf_counter()
            
            try:
                frame = self.reachy.media.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                h, w = frame.shape[:2]
                small = cv2.resize(frame, (320, 180))
                results = model(small, verbose=False)
                boxes = results[0].boxes
                
                if len(boxes) > 0:
                    confs = boxes.conf.cpu().numpy()
                    xyxy = boxes.xyxy.cpu().numpy()
                    
                    best = confs.argmax()
                    box = xyxy[best]
                    conf = float(confs[best])
                    
                    if conf > 0.5:
                        sx, sy = w / 320, h / 180
                        u = int((box[0] + box[2]) / 2 * sx)
                        v = int((box[1] + box[3]) / 2 * sy)
                        
                        target_pose = self.reachy.look_at_image(
                            u, v, duration=0.0, perform_movement=False
                        )
                        
                        trans = target_pose[:3, 3]
                        rot = R.from_matrix(target_pose[:3, :3]).as_euler("xyz")
                        
                        raw = [trans[0], trans[1], trans[2], rot[0], rot[1], rot[2]]
                        self._machine.update_face(raw)
                        
                        state = self._machine.get_state()
                        colors = {
                            "idle": (128, 128, 128),
                            "noticing": (0, 165, 255),
                            "moving": (0, 255, 255),
                            "holding": (0, 255, 0),
                            "lost": (0, 0, 255)
                        }
                        cv2.circle(frame, (u, v), 15, colors.get(state, (255,255,255)), 3)
                    else:
                        self._machine.update_no_face()
                else:
                    self._machine.update_no_face()
                
                state = self._machine.get_state()
                cv2.putText(frame, state.upper(), (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                with self._frame_lock:
                    self._latest_frame = frame
                    
            except Exception as e:
                print(f"[Camera] {e}")
            
            elapsed = time.perf_counter() - start
            if elapsed < period:
                time.sleep(period - elapsed)


# =============================================================================
# MOVEMENT MANAGER
# =============================================================================

class MovementManager:
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
    
    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[Movement] Started (30Hz)")
    
    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()
    
    def _loop(self) -> None:
        period = 1.0 / 30.0
        
        while not self._stop.is_set():
            start = time.perf_counter()
            
            try:
                off = self.camera_worker.get_output()
                
                pose = self._create(
                    x=off[0], y=off[1], z=off[2],
                    roll=off[3], pitch=off[4], yaw=off[5],
                    degrees=False, mm=False
                )
                
                combined = self._compose(self._neutral, pose, reorthonormalize=True)
                self.reachy.set_target(head=combined)
                
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

app = FastAPI()
camera_worker: Optional[CameraWorker] = None

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>FuelOS - States</title>
    <style>
        body { font-family: system-ui; background: #111; color: #fff; margin: 0; padding: 20px; }
        h1 { color: #0c6; }
        .container { display: flex; gap: 20px; flex-wrap: wrap; }
        .video { flex: 1; min-width: 400px; }
        .controls { flex: 1; min-width: 300px; max-width: 400px; }
        img { width: 100%; border-radius: 8px; }
        .group { background: #222; border-radius: 8px; padding: 15px; margin-bottom: 15px; }
        .group h3 { margin: 0 0 10px 0; color: #0c6; font-size: 14px; }
        .row { display: flex; align-items: center; margin: 8px 0; }
        .row label { flex: 1; font-size: 12px; color: #888; }
        .row input { flex: 2; }
        .row .val { width: 60px; text-align: right; font-size: 12px; color: #0c6; }
        .state { font-size: 28px; color: #0c6; margin: 10px 0; font-weight: bold; }
        .note { background: #1a1a1a; border-left: 3px solid #0c6; padding: 10px; margin: 10px 0; font-size: 11px; color: #666; }
        .states-legend { display: flex; gap: 10px; flex-wrap: wrap; margin: 10px 0; }
        .state-item { padding: 5px 10px; border-radius: 4px; font-size: 11px; }
    </style>
</head>
<body>
    <h1>🎯 FuelOS - State Machine</h1>
    <div class="note">
        <strong>Philosophy:</strong> HOLDING is the default. Movement is an EVENT, not continuous tracking.<br>
        Robot should be mostly STILL, only moving when face shifts significantly.
    </div>
    <div class="states-legend">
        <span class="state-item" style="background:#808080">IDLE</span>
        <span class="state-item" style="background:#FFA500">NOTICING</span>
        <span class="state-item" style="background:#FFFF00;color:#000">MOVING</span>
        <span class="state-item" style="background:#00FF00;color:#000">HOLDING</span>
        <span class="state-item" style="background:#FF0000">LOST</span>
    </div>
    <div class="container">
        <div class="video">
            <img id="v">
            <div class="state">State: <span id="state">-</span></div>
        </div>
        <div class="controls">
            <div class="group">
                <h3>🎯 Thresholds</h3>
                <div class="row">
                    <label>Major Threshold (°)</label>
                    <input type="range" id="major_threshold" min="0.05" max="0.2" step="0.01" value="0.10">
                    <span class="val" id="major_threshold_val">5.7°</span>
                </div>
            </div>
            <div class="group">
                <h3>⏱️ Timing</h3>
                <div class="row">
                    <label>Notice Duration (s)</label>
                    <input type="range" id="notice_duration" min="0.02" max="0.2" step="0.02" value="0.08">
                    <span class="val" id="notice_duration_val">0.08</span>
                </div>
                <div class="row">
                    <label>Move Duration (s)</label>
                    <input type="range" id="move_duration" min="0.1" max="0.5" step="0.02" value="0.25">
                    <span class="val" id="move_duration_val">0.25</span>
                </div>
            </div>
            <div class="group">
                <h3>📏 Scale</h3>
                <div class="row">
                    <label>Motion Scale</label>
                    <input type="range" id="motion_scale" min="0.3" max="0.9" step="0.05" value="0.55">
                    <span class="val" id="motion_scale_val">0.55</span>
                </div>
            </div>
        </div>
    </div>
    <script>
        const v = document.getElementById('v');
        const stateEl = document.getElementById('state');
        
        function connect() {
            const ws = new WebSocket('ws://' + location.host + '/ws');
            ws.onmessage = e => {
                const d = JSON.parse(e.data);
                v.src = 'data:image/jpeg;base64,' + d.f;
                if (d.s) stateEl.textContent = d.s.toUpperCase();
            };
            ws.onclose = () => setTimeout(connect, 1000);
        }
        connect();
        
        const sliders = ['major_threshold', 'notice_duration', 'move_duration', 'motion_scale'];
        
        sliders.forEach(id => {
            const s = document.getElementById(id);
            const val = document.getElementById(id + '_val');
            s.addEventListener('input', () => {
                let v = parseFloat(s.value);
                if (id === 'major_threshold') {
                    val.textContent = (v * 180 / 3.14159).toFixed(1) + '°';
                } else {
                    val.textContent = v.toFixed(2);
                }
                fetch('/config', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({key: id, value: v})
                });
            });
        });
        
        fetch('/config').then(r=>r.json()).then(cfg => {
            sliders.forEach(id => {
                if (cfg[id] !== undefined) {
                    const s = document.getElementById(id);
                    const val = document.getElementById(id + '_val');
                    s.value = cfg[id];
                    let v = cfg[id];
                    if (id === 'major_threshold') {
                        val.textContent = (v * 180 / 3.14159).toFixed(1) + '°';
                    } else {
                        val.textContent = parseFloat(v).toFixed(2);
                    }
                }
            });
        });
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML

@app.get("/config")
async def get_config():
    return Config.get_all()

@app.post("/config")
async def set_config(data: dict):
    Config.set(data['key'], data['value'])
    return {"ok": True}

@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            if camera_worker:
                frame = camera_worker.get_latest_frame()
                if frame is not None:
                    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    await websocket.send_json({
                        'f': base64.b64encode(buf).decode(),
                        's': camera_worker.get_state()
                    })
            await asyncio.sleep(0.05)
    except:
        pass


@app.on_event("startup")
async def startup():
    global camera_worker
    
    print("=" * 60)
    print("FuelOS - STATE MACHINE")
    print("=" * 60)
    print("States: IDLE → NOTICING → MOVING → HOLDING → LOST")
    print("Key: Robot should be mostly HOLDING (still)")
    print("=" * 60)
    
    from reachy_mini import ReachyMini
    
    reachy = ReachyMini(media_backend="default")
    print("[Main] Reachy connected")
    
    camera_worker = CameraWorker(reachy)
    camera_worker.start()
    
    movement = MovementManager(reachy, camera_worker)
    movement.start()
    
    print("[Main] Ready")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
