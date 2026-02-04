from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import os
import uuid
from datetime import datetime
from pathlib import Path

from core_models.transcription import JudicialTranscriber
from utils.audio_utils import save_upload_file, validate_audio_file

router = APIRouter()
transcriber = JudicialTranscriber()

@router.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: str = "en"
):
    """
    Transcribe uploaded audio file
    """
    # Validate file
    validation = validate_audio_file(audio)
    if not validation["valid"]:
        raise HTTPException(status_code=400, detail=validation["error"])
    
    # Save file temporarily
    temp_id = str(uuid.uuid4())
    file_path = save_upload_file(audio, temp_id)
    
    try:
        # Transcribe
        result = transcriber.transcribe_file(file_path, language)
        
        # Generate response
        response = {
            "success": True,
            "transcript": result["text"],
            "formatted": result["formatted"],
            "language": result["language"],
            "confidence": result.get("confidence", 0),
            "duration": result.get("duration", 0),
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        # Cleanup temp file
        if os.path.exists(file_path):
            os.remove(file_path)

@router.get("/formats")
async def get_supported_formats():
    """Get supported formats and languages"""
    return {
        "supported_formats": [
            {"id": "high_court", "name": "High Court Judgment"},
            {"id": "supreme_court", "name": "Supreme Court Judgment"},
            {"id": "district_court", "name": "District Court Order"},
        ],
        "supported_languages": [
            {"code": "en", "name": "English"},
            {"code": "hi", "name": "Hindi"}
        ],
        "max_file_size_mb": 500,
        "supported_audio_formats": [".mp3", ".wav", ".m4a", ".ogg", ".flac"]
    }
