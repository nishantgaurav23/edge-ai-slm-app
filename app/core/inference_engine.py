import logging
import gc
import os
try:
    from llama_cpp import Llama
except ImportError:
    Llama = None # Handle case where deps aren't installed yet for linting/loading

logger = logging.getLogger(__name__)

class InferenceEngine:
    """
    Singleton-like class to manage the Llama instance.
    Handles Lazy Loading and Unloading to manage memory pressure.
    """
    _instance = None
    _model = None
    _current_model_path = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def is_loaded(self):
        return self._model is not None

    def load_model(self, model_path: str, n_ctx: int = 2048, n_gpu_layers: int = 0):
        """
        Loads the Llama model. Unloads existing model if different.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # If already loaded with same path, do nothing
        if self._model and self._current_model_path == model_path:
            logger.info("Model already loaded.")
            return

        # Unload previous
        self.unload_model()

        logger.info(f"Loading model from {model_path}...")
        try:
            if Llama is None:
                raise ImportError("llama_cpp library not found. Please install requirements.")

            # n_gpu_layers = 0 means CPU only detailed optimization logic will come in DeviceService
            self._model = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=True
            )
            self._current_model_path = model_path
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

    def unload_model(self):
        """
        Forcefully removes the model from memory.
        """
        if self._model:
            logger.info("Unloading model...")
            del self._model
            self._model = None
            self._current_model_path = None
            gc.collect() # Force garbage collection
            logger.info("Model unloaded.")

    def generate_response(self, prompt: str, max_tokens: int = 128, stream: bool = True,
                         temperature: float = 0.7, top_p: float = 0.9,
                         repeat_penalty: float = 1.1, stop: list = None):
        """
        Generates text based on prompt.

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            temperature: Sampling temperature (0.0-2.0). Lower = more focused
            top_p: Nucleus sampling probability
            repeat_penalty: Penalty for repeating tokens (>1.0 reduces repetition)
            stop: List of stop sequences
        """
        if not self._model:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        # Default stop sequences for TinyLlama
        if stop is None:
            stop = ["</s>", "<|user|>", "<|system|>"]

        return self._model(
            prompt,
            max_tokens=max_tokens,
            stream=stream,
            echo=False,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stop=stop
        )
