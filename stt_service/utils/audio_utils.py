"""
Audio utility functions
"""
import os
import aiofiles
from typing import Tuple, Optional
from fastapi import UploadFile, HTTPException
from loguru import logger

# Supported audio formats
SUPPORTED_AUDIO_FORMATS = {
    '.wav': 'audio/wav',
    '.mp3': 'audio/mpeg',
    '.m4a': 'audio/mp4',
    '.ogg': 'audio/ogg',
    '.flac': 'audio/flac',
    '.webm': 'audio/webm',
}

async def save_upload_file(upload_file: UploadFile, upload_dir: str) -> str:
    """
    Save uploaded file to disk
    
    Args:
        upload_file: FastAPI UploadFile object
        upload_dir: Directory to save the file
        
    Returns:
        Path to saved file
    """
    # Create upload directory if it doesn't exist
    os.makedirs(upload_dir, exist_ok=True)
    
    # Generate safe filename
    original_filename = upload_file.filename or "audio_file"
    safe_filename = "".join(c for c in original_filename if c.isalnum() or c in (' ', '.', '_', '-')).rstrip()
    
    # Add extension if missing
    if not '.' in safe_filename:
        safe_filename += '.wav'
    
    # Create file path
    file_path = os.path.join(upload_dir, safe_filename)
    
    # Save file
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await upload_file.read()
        await out_file.write(content)
    
    logger.info(f"File saved: {file_path} ({len(content)} bytes)")
    return file_path

def validate_audio_file(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate if file is a supported audio format
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not os.path.exists(file_path):
        return False, f"File not found: {file_path}"
    
    if os.path.getsize(file_path) == 0:
        return False, "File is empty"
    
    # Check file extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    if ext not in SUPPORTED_AUDIO_FORMATS:
        return False, f"Unsupported file extension: {ext}. Supported: {list(SUPPORTED_AUDIO_FORMATS.keys())}"
    
    # Basic file size check (500MB limit)
    max_size_mb = 500
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    
    if file_size_mb > max_size_mb:
        return False, f"File too large: {file_size_mb:.2f}MB. Max: {max_size_mb}MB"
    
    return True, None

def get_audio_duration(file_path: str) -> float:
    """
    Get duration of audio file in seconds
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Duration in seconds
    """
    try:
        import librosa
        duration = librosa.get_duration(path=file_path)
        return duration
    except:
        try:
            import wave
            with wave.open(file_path, 'rb') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                return frames / float(rate)
        except:
            # Rough estimate
            file_size = os.path.getsize(file_path)
            return file_size / (1024 * 1024) * 60  # 1MB ≈ 1 minute

def cleanup_temp_files(file_paths: list):
    """
    Clean up temporary files
    
    Args:
        file_paths: List of file paths to delete
    """
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Cleaned up temp file: {file_path}")
        except Exception as e:
            logger.warning(f"Could not delete temp file {file_path}: {e}")
