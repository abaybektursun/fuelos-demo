#!/usr/bin/env python3
"""
Face Detection & Recognition Server for FuelOS Demo Booth
=========================================================
Real-time face detection with bounding boxes and person IDs.
Streams video via web interface on local network.

Usage:
    python face_server.py [--port 8080] [--host 0.0.0.0]
"""

import asyncio
import argparse
import base64
import cv2
import numpy as np
import time
import json
from pathlib import Path
from typing import Optional

# InsightFace for SOTA face detection + recognition
import insightface
from insightface.app import FaceAnalysis

# FastAPI for web server
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

# Reachy Mini camera
try:
    from reachy_mini import ReachyMini
    REACHY_AVAILABLE = True
except ImportError:
    REACHY_AVAILABLE = False
    print("Warning: reachy_mini not available, using webcam fallback")


class FaceTracker:
    """Manages face detection, recognition, and tracking."""
    
    def __init__(self, det_size=(640, 480)):
        print("Initializing InsightFace...")
        self.app = FaceAnalysis(
            name='buffalo_l',  # Best accuracy model
            providers=['CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=det_size)
        
        # Face database: {person_id: embedding}
        self.face_db = {}
        self.next_person_id = 1
        
        # Tracking state
        self.last_seen = {}
        self.recognition_threshold = 0.5
        
        print(f"InsightFace ready. Detection size: {det_size}")
    
    def _cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _find_or_create_person(self, embedding) -> tuple[int, bool]:
        best_match = None
        best_score = -1
        
        for pid, db_embedding in self.face_db.items():
            score = self._cosine_similarity(embedding, db_embedding)
            if score > best_score:
                best_score = score
                best_match = pid
        
        if best_match is not None and best_score > self.recognition_threshold:
            alpha = 0.1
            self.face_db[best_match] = (1 - alpha) * self.face_db[best_match] + alpha * embedding
            return best_match, False
        else:
            pid = self.next_person_id
            self.next_person_id += 1
            self.face_db[pid] = embedding
            return pid, True
    
    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """
        Process a frame for face detection and recognition.
        Frame should be BGR format (OpenCV standard).
        """
        # InsightFace expects BGR - frame is already BGR
        faces = self.app.get(frame)
        detections = []
        now = time.time()
        
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            embedding = face.embedding
            person_id, is_new = self._find_or_create_person(embedding)
            self.last_seen[person_id] = now
            
            # Distance estimation
            face_height = y2 - y1
            estimated_distance = (200 * frame.shape[0]) / (face_height * 2)
            estimated_distance_m = estimated_distance / 1000
            
            det_info = {
                'person_id': person_id,
                'is_new': is_new,
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(face.det_score),
                'distance_m': round(estimated_distance_m, 2),
                'age': int(face.age) if hasattr(face, 'age') and face.age else None,
                'gender': 'M' if hasattr(face, 'gender') and face.gender == 1 else 'F' if hasattr(face, 'gender') else None,
            }
            detections.append(det_info)
            
            # Draw bounding box (BGR colors)
            color = (0, 255, 0) if not is_new else (0, 255, 255)  # Green/Yellow in BGR
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"Person {person_id}"
            if det_info['distance_m']:
                label += f" ({det_info['distance_m']:.1f}m)"
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Clean up old faces
        timeout = 30
        self.face_db = {pid: emb for pid, emb in self.face_db.items() 
                       if now - self.last_seen.get(pid, 0) < timeout}
        
        return frame, detections


class VideoSource:
    """Video source - Reachy camera or webcam fallback."""
    
    def __init__(self, use_reachy=True):
        self.use_reachy = use_reachy and REACHY_AVAILABLE
        self.reachy = None
        self.cap = None
        
        if self.use_reachy:
            try:
                print("Connecting to Reachy Mini...")
                self.reachy = ReachyMini(media_backend="default")
                print("Reachy Mini connected!")
            except Exception as e:
                print(f"Reachy connection failed: {e}, falling back to webcam")
                self.use_reachy = False
        
        if not self.use_reachy:
            print("Using webcam...")
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get a BGR frame from the video source."""
        if self.use_reachy and self.reachy:
            try:
                frame = self.reachy.media.get_frame()
                # Reachy with OpenCV backend returns BGR
                # No conversion needed
                return frame
            except Exception as e:
                print(f"Reachy frame error: {e}")
                return None
        elif self.cap:
            ret, frame = self.cap.read()
            # Webcam returns BGR
            return frame if ret else None
        return None
    
    def release(self):
        if self.cap:
            self.cap.release()


# Global state
face_tracker: Optional[FaceTracker] = None
video_source: Optional[VideoSource] = None
latest_frame: Optional[np.ndarray] = None
latest_detections: list[dict] = []

# FastAPI app
app = FastAPI(title="FuelOS Face Detection Server")

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>FuelOS Face Detection</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            margin: 0;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #00ff88; margin-bottom: 20px; }
        .video-container { display: flex; gap: 20px; flex-wrap: wrap; }
        .video-box {
            background: #16213e;
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }
        #video-feed { border-radius: 8px; max-width: 100%; }
        .info-panel {
            background: #16213e;
            border-radius: 12px;
            padding: 20px;
            min-width: 300px;
        }
        .face-card {
            background: #0f3460;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 10px;
            border-left: 4px solid #00ff88;
        }
        .face-card.new { border-left-color: #ffcc00; }
        .person-id { font-size: 1.2em; font-weight: bold; color: #00ff88; }
        .face-card.new .person-id { color: #ffcc00; }
        .stat { color: #888; font-size: 0.9em; }
        .status {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            margin-left: 10px;
        }
        .status.connected { background: #00ff88; color: #000; }
        .status.disconnected { background: #ff4444; color: #fff; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 FuelOS Face Detection 
            <span id="connection-status" class="status disconnected">Disconnected</span>
        </h1>
        <div class="video-container">
            <div class="video-box">
                <img id="video-feed" width="640" height="480" alt="Video Feed">
                <div id="fps" class="stat">FPS: --</div>
            </div>
            <div class="info-panel">
                <h3>👥 Detected Faces</h3>
                <div id="faces-list"><p class="stat">No faces detected</p></div>
                <hr style="border-color: #333; margin: 20px 0;">
                <h3>📊 Stats</h3>
                <p class="stat">Total persons seen: <span id="total-persons">0</span></p>
                <p class="stat">Currently visible: <span id="current-count">0</span></p>
            </div>
        </div>
    </div>
    <script>
        const img = document.getElementById('video-feed');
        const facesList = document.getElementById('faces-list');
        const totalPersons = document.getElementById('total-persons');
        const currentCount = document.getElementById('current-count');
        const connectionStatus = document.getElementById('connection-status');
        const fpsDisplay = document.getElementById('fps');
        
        let ws, frameCount = 0, lastFpsTime = Date.now(), maxPersonId = 0;
        
        function connect() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);
            ws.onopen = () => {
                connectionStatus.textContent = 'Connected';
                connectionStatus.className = 'status connected';
            };
            ws.onclose = () => {
                connectionStatus.textContent = 'Disconnected';
                connectionStatus.className = 'status disconnected';
                setTimeout(connect, 2000);
            };
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                img.src = 'data:image/jpeg;base64,' + data.frame;
                
                frameCount++;
                const now = Date.now();
                if (now - lastFpsTime >= 1000) {
                    fpsDisplay.textContent = `FPS: ${frameCount}`;
                    frameCount = 0;
                    lastFpsTime = now;
                }
                
                const faces = data.detections;
                currentCount.textContent = faces.length;
                
                if (faces.length === 0) {
                    facesList.innerHTML = '<p class="stat">No faces detected</p>';
                } else {
                    let html = '';
                    faces.forEach(face => {
                        if (face.person_id > maxPersonId) maxPersonId = face.person_id;
                        html += `
                            <div class="face-card ${face.is_new ? 'new' : ''}">
                                <span class="person-id">Person ${face.person_id}</span>
                                ${face.is_new ? '<span class="stat">(NEW)</span>' : ''}
                                <br>
                                <span class="stat">Distance: ${face.distance_m}m</span>
                                <span class="stat">Confidence: ${(face.confidence * 100).toFixed(0)}%</span>
                                ${face.age ? `<br><span class="stat">Age: ~${face.age}</span>` : ''}
                                ${face.gender ? `<span class="stat">Gender: ${face.gender}</span>` : ''}
                            </div>
                        `;
                    });
                    facesList.innerHTML = html;
                }
                totalPersons.textContent = maxPersonId;
            };
        }
        connect();
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            if latest_frame is not None:
                # Frame is BGR, imencode expects BGR - no conversion needed
                _, buffer = cv2.imencode('.jpg', latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                await websocket.send_json({
                    'frame': frame_b64,
                    'detections': latest_detections,
                    'timestamp': time.time()
                })
            await asyncio.sleep(0.033)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")

@app.get("/api/faces")
async def get_faces():
    return {'detections': latest_detections, 'timestamp': time.time()}

async def video_loop():
    global latest_frame, latest_detections
    print("Starting video loop...")
    while True:
        frame = video_source.get_frame()
        if frame is not None:
            # Resize for faster processing if needed
            if frame.shape[0] > 720:
                scale = 720 / frame.shape[0]
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
            
            annotated_frame, detections = face_tracker.process_frame(frame)
            latest_frame = annotated_frame
            latest_detections = detections
        await asyncio.sleep(0.01)

@app.on_event("startup")
async def startup():
    global face_tracker, video_source
    print("=" * 50)
    print("FuelOS Face Detection Server")
    print("=" * 50)
    face_tracker = FaceTracker()
    video_source = VideoSource(use_reachy=True)
    asyncio.create_task(video_loop())
    print("\nServer ready!")
    print(f"Open http://<your-ip>:8080 in your browser")
    print("=" * 50)

@app.on_event("shutdown")
async def shutdown():
    if video_source:
        video_source.release()

def main():
    parser = argparse.ArgumentParser(description='FuelOS Face Detection Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to bind to')
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
