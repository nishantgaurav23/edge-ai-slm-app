import os
import requests
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Handles downloading and verifying GGUF models.
    """
    
    # Example generic path - in real app, use app_storage_path via plyer
    MODEL_DIR = "models" 
    
    def __init__(self, model_dir=None):
        self.model_dir = model_dir if model_dir else self.MODEL_DIR
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def download_model(self, url: str, filename: str, progress_callback=None):
        """
        Downloads a model from a URL to the local storage.
        
        Args:
            url (str): Direct download link.
            filename (str): Name to save the file as.
            progress_callback (func): Function accepting (current_bytes, total_bytes).
        """
        file_path = os.path.join(self.model_dir, filename)
        
        if os.path.exists(file_path):
            logger.info(f"Model {filename} already exists at {file_path}")
            return file_path

        logger.info(f"Starting download for {filename}...")
        
        try:
            with requests.get(url, stream=True) as response:
                response.raise_for_status()
                total_length = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if progress_callback:
                                progress_callback(downloaded, total_length)
            
            logger.info(f"Download complete: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            # Cleanup partial download
            if os.path.exists(file_path):
                os.remove(file_path)
            raise e

    def get_model_path(self, filename: str):
        path = os.path.join(self.model_dir, filename)
        return path if os.path.exists(path) else None
