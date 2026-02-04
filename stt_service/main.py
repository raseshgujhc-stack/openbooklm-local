import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from api.routes import router as api_router
import logging
from datetime import datetime
import json

# Set up data directory
DATA_DIR = os.getenv('DATA_DIR', '/home/ubuntu/openbooklm-local/data')
STT_DATA_DIR = Path(DATA_DIR)

# Create subdirectories if they don't exist
SUBDIRS = ['stt-models', 'stt-uploads', 'stt-transcripts', 'stt-logs']
for subdir in SUBDIRS:
    dir_path = STT_DATA_DIR / subdir
    dir_path.mkdir(parents=True, exist_ok=True)
    print(f"Ensured directory exists: {dir_path}")

# Update paths for the application
MODEL_DIR = STT_DATA_DIR / "stt-models"
UPLOAD_DIR = STT_DATA_DIR / "stt-uploads"
TRANSCRIPT_DIR = STT_DATA_DIR / "stt-transcripts"
LOG_DIR = STT_DATA_DIR / "stt-logs"

# Configure logging
LOG_FILE = LOG_DIR / f"stt-service-{datetime.now().strftime('%Y%m%d')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Judicial STT Service",
    description="Speech-to-Text for Indian Judiciary with Legal Formatting",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],  # Added for CORS
)

# CORS headers middleware
@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# WebSocket connections manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.transcriber = None
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New WebSocket connection. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Remaining: {len(self.active_connections)}")
    
    async def send_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                self.disconnect(connection)

manager = ConnectionManager()

# Initialize transcriber lazily
def get_transcriber():
    if manager.transcriber is None:
        from core_models.transcription import JudicialTranscriber
        manager.transcriber = JudicialTranscriber()
        logger.info("Transcriber initialized")
    return manager.transcriber

# Handle OPTIONS requests for CORS preflight
@app.options("/{rest_of_path:path}")
async def options_handler():
    return JSONResponse(
        content={"status": "ok"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "true",
        }
    )

# WebSocket endpoint for live transcription
@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    # Add CORS headers for WebSocket connection
    websocket.headers.update({
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Credentials": "true"
    })
    
    await manager.connect(websocket)
    transcriber = get_transcriber()
    
    try:
        # Send connection confirmation
        await manager.send_message({
            "type": "status",
            "message": "Connected to STT service",
            "timestamp": datetime.now().isoformat()
        }, websocket)
        
        while True:
            # Receive data from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "audio_chunk":
                # Process audio chunk
                audio_data = message["data"]
                language = message.get("language", "en")
                
                # Transcribe audio
                try:
                    transcript = transcriber.transcribe_chunk(
                        audio_data, 
                        language=language
                    )
                    
                    # Send transcription back
                    await manager.send_message({
                        "type": "transcription",
                        "text": transcript["text"],
                        "formatted": transcript.get("formatted", transcript["text"]),
                        "timestamp": datetime.now().isoformat(),
                        "language": language
                    }, websocket)
                    
                except Exception as e:
                    logger.error(f"Transcription error: {e}")
                    await manager.send_message({
                        "type": "error",
                        "message": f"Transcription failed: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    }, websocket)
            
            elif message["type"] == "command":
                if message["command"] == "ping":
                    await manager.send_message({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }, websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# HTTP endpoints
@app.get("/")
async def root():
    return {
        "service": "Judicial STT Service",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "transcribe": "/api/v1/transcribe",
            "formats": "/api/v1/formats",
            "websocket": "/ws/transcribe",
            "test_client": "/test"  # Added test client endpoint
        }
    }

@app.get("/health")
async def health_check():
    import psutil
    import shutil
    
    # Check disk space
    total, used, free = shutil.disk_usage("/")
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "websocket_connections": len(manager.active_connections),
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_used_gb": psutil.virtual_memory().used / (1024**3),
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "disk_free_gb": free / (1024**3),
        }
    }

# Test client endpoint for microphone
@app.get("/test")
async def test_client():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Microphone Test - Judicial STT Service</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #2c3e50;
                text-align: center;
                margin-bottom: 30px;
            }
            .button-group {
                display: flex;
                gap: 10px;
                justify-content: center;
                margin-bottom: 20px;
            }
            button {
                padding: 12px 24px;
                font-size: 16px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                transition: background-color 0.3s;
            }
            #startBtn {
                background-color: #27ae60;
                color: white;
            }
            #startBtn:hover:not(:disabled) {
                background-color: #219653;
            }
            #stopBtn {
                background-color: #e74c3c;
                color: white;
            }
            #stopBtn:hover:not(:disabled) {
                background-color: #c0392b;
            }
            button:disabled {
                background-color: #95a5a6;
                cursor: not-allowed;
            }
            .status-box {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
                border-left: 4px solid #3498db;
            }
            .transcript-box {
                background-color: #fff;
                border: 2px solid #3498db;
                border-radius: 5px;
                padding: 20px;
                margin-top: 20px;
                min-height: 100px;
                white-space: pre-wrap;
                word-wrap: break-word;
            }
            .label {
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 5px;
                display: block;
            }
            .connection-info {
                background-color: #e8f4fc;
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
                font-size: 14px;
            }
            .instructions {
                background-color: #fff3cd;
                border: 1px solid #ffeaa7;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎤 Judicial STT Service - Microphone Test</h1>
            
            <div class="instructions">
                <strong>Instructions:</strong>
                <ol>
                    <li>Click "Start Recording" and allow microphone access</li>
                    <li>Speak clearly into your microphone</li>
                    <li>View real-time transcription below</li>
                    <li>Click "Stop Recording" when done</li>
                </ol>
            </div>
            
            <div class="connection-info">
                <strong>Connection Status:</strong> 
                <span id="connectionStatus">Not connected</span>
            </div>
            
            <div class="button-group">
                <button id="startBtn">🎤 Start Recording</button>
                <button id="stopBtn" disabled>⏹️ Stop Recording</button>
            </div>
            
            <div class="status-box">
                <span class="label">Status:</span>
                <span id="status">Ready to start recording</span>
            </div>
            
            <div>
                <span class="label">Live Transcription:</span>
                <div class="transcript-box" id="transcript">
                    Transcription will appear here...
                </div>
            </div>
            
            <div style="margin-top: 30px; font-size: 14px; color: #7f8c8d; text-align: center;">
                <p>Judicial STT Service v1.0.0 | For Indian Judiciary with Legal Formatting</p>
                <p id="serverInfo">Server: Connecting...</p>
            </div>
        </div>

        <script>
            let mediaRecorder;
            let audioChunks = [];
            let websocket;
            let isRecording = false;
            
            // Get server URL
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.hostname}:${window.location.port || 8003}/ws/transcribe`;
            const apiUrl = `${window.location.protocol}//${window.location.hostname}:${window.location.port || 8003}`;
            
            // Update server info
            document.getElementById('serverInfo').textContent = `Server: ${window.location.hostname}:${window.location.port || 8003}`;
            
            document.getElementById('startBtn').onclick = async () => {
                try {
                    // Request microphone access with optimal settings for STT
                    const stream = await navigator.mediaDevices.getUserMedia({ 
                        audio: {
                            channelCount: 1,        // Mono audio
                            sampleRate: 16000,      // 16kHz sample rate
                            echoCancellation: true,
                            noiseSuppression: true,
                            autoGainControl: true
                        }
                    });
                    
                    // Create MediaRecorder
                    mediaRecorder = new MediaRecorder(stream, {
                        mimeType: 'audio/webm;codecs=opus',
                        audioBitsPerSecond: 16000
                    });
                    
                    // Handle audio data
                    mediaRecorder.ondataavailable = async (event) => {
                        if (event.data.size > 0) {
                            // Convert blob to base64
                            const reader = new FileReader();
                            reader.onload = () => {
                                const arrayBuffer = reader.result;
                                const bytes = new Uint8Array(arrayBuffer);
                                
                                // Convert to base64
                                let binary = '';
                                for (let i = 0; i < bytes.byteLength; i++) {
                                    binary += String.fromCharCode(bytes[i]);
                                }
                                const base64Audio = btoa(binary);
                                
                                // Send to WebSocket
                                if (websocket && websocket.readyState === WebSocket.OPEN) {
                                    websocket.send(JSON.stringify({
                                        type: "audio_chunk",
                                        data: base64Audio,
                                        language: "en"
                                    }));
                                }
                            };
                            reader.readAsArrayBuffer(event.data);
                        }
                    };
                    
                    // Handle recording stop
                    mediaRecorder.onstop = () => {
                        stream.getTracks().forEach(track => track.stop());
                        isRecording = false;
                    };
                    
                    // Connect WebSocket
                    websocket = new WebSocket(wsUrl);
                    
                    websocket.onopen = () => {
                        document.getElementById('connectionStatus').textContent = 'Connected';
                        document.getElementById('connectionStatus').style.color = 'green';
                        
                        // Start recording
                        mediaRecorder.start(1000); // Send chunks every second
                        isRecording = true;
                        
                        document.getElementById('status').textContent = "Recording... Speak now";
                        document.getElementById('status').style.color = "green";
                        document.getElementById('startBtn').disabled = true;
                        document.getElementById('stopBtn').disabled = false;
                        
                        // Clear previous transcript
                        document.getElementById('transcript').innerHTML = "Listening...<br>";
                    };
                    
                    websocket.onmessage = (event) => {
                        try {
                            const data = JSON.parse(event.data);
                            console.log("Received:", data);
                            
                            if (data.type === "transcription") {
                                const transcriptDiv = document.getElementById('transcript');
                                const newText = data.text.trim();
                                if (newText) {
                                    // Add timestamp
                                    const now = new Date();
                                    const timeStr = now.toLocaleTimeString();
                                    
                                    // Append new transcription
                                    transcriptDiv.innerHTML += 
                                        `<strong>[${timeStr}]</strong> ${newText}<br>`;
                                    
                                    // Auto-scroll to bottom
                                    transcriptDiv.scrollTop = transcriptDiv.scrollHeight;
                                }
                            } else if (data.type === "status") {
                                document.getElementById('status').textContent = data.message;
                            } else if (data.type === "error") {
                                document.getElementById('status').textContent = "Error: " + data.message;
                                document.getElementById('status').style.color = "red";
                            }
                        } catch (error) {
                            console.error("Error processing message:", error);
                        }
                    };
                    
                    websocket.onerror = (error) => {
                        console.error("WebSocket error:", error);
                        document.getElementById('connectionStatus').textContent = 'Connection error';
                        document.getElementById('connectionStatus').style.color = 'red';
                        document.getElementById('status').textContent = "WebSocket connection failed";
                        document.getElementById('status').style.color = "red";
                    };
                    
                    websocket.onclose = () => {
                        document.getElementById('connectionStatus').textContent = 'Disconnected';
                        document.getElementById('connectionStatus').style.color = 'gray';
                        if (isRecording) {
                            document.getElementById('status').textContent = "Connection lost";
                            document.getElementById('status').style.color = "orange";
                        }
                    };
                    
                } catch (error) {
                    console.error("Error accessing microphone:", error);
                    document.getElementById('status').textContent = "Error: " + error.message;
                    document.getElementById('status').style.color = "red";
                    
                    if (error.name === "NotAllowedError") {
                        document.getElementById('status').textContent = "Microphone access denied. Please allow microphone permissions.";
                    } else if (error.name === "NotFoundError") {
                        document.getElementById('status').textContent = "No microphone found. Please connect a microphone.";
                    }
                }
            };
            
            document.getElementById('stopBtn').onclick = () => {
                if (mediaRecorder && isRecording) {
                    mediaRecorder.stop();
                    isRecording = false;
                }
                if (websocket) {
                    websocket.close();
                }
                
                document.getElementById('status').textContent = "Recording stopped";
                document.getElementById('status').style.color = "blue";
                document.getElementById('startBtn').disabled = false;
                document.getElementById('stopBtn').disabled = true;
                document.getElementById('connectionStatus').textContent = 'Disconnected';
                document.getElementById('connectionStatus').style.color = 'gray';
            };
            
            // Test connection on load
            window.addEventListener('load', async () => {
                try {
                    const response = await fetch(`${apiUrl}/health`);
                    if (response.ok) {
                        document.getElementById('serverInfo').style.color = 'green';
                    } else {
                        document.getElementById('serverInfo').style.color = 'red';
                        document.getElementById('serverInfo').textContent += ' (Health check failed)';
                    }
                } catch (error) {
                    console.log("Health check failed (might be normal if CORS not set up yet)");
                }
            });
            
            // Handle page unload
            window.addEventListener('beforeunload', () => {
                if (websocket) {
                    websocket.close();
                }
                if (mediaRecorder && isRecording) {
                    mediaRecorder.stop();
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

app.include_router(api_router, prefix="/api/v1")

# Mount static files
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/transcripts", StaticFiles(directory=TRANSCRIPT_DIR), name="transcripts")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8003,
        workers=1,  # Single worker for WebSocket compatibility
        log_level="info",
        headers=[
            ("Access-Control-Allow-Origin", "*"),
            ("Access-Control-Allow-Methods", "*"),
            ("Access-Control-Allow-Headers", "*"),
            ("Access-Control-Allow-Credentials", "true")
        ]
    )
