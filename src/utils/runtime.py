# src/utils/runtime.py  (TF2 / Keras 3 compatible)
import os
import logging
import inspect
import tensorflow as tf

_logger = logging.getLogger(__name__)

def funcname():
    """Return the caller's function name (used for logging)."""
    frame = inspect.currentframe()
    if frame is None or frame.f_back is None:
        return "<unknown>"
    return frame.f_back.f_code.co_name

def gpu_selection(memory_growth: bool = True) -> None:
    """
    TF2-safe GPU setup:
      - Logs visible GPUs
      - Optionally enables memory growth
    This is a no-op if no GPUs are present.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        _logger.info("No GPUs detected by TensorFlow.")
        return

    _logger.info("GPUs detected: %s", ", ".join(g.name for g in gpus))
    if memory_growth:
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception as e:
                _logger.warning("Could not set memory growth for %s: %s", g, e)

    # If you want to pin to a specific GPU via env, honor CUDA_VISIBLE_DEVICES
    visible_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_env:
        _logger.info("CUDA_VISIBLE_DEVICES=%s (honored by TF at import time).", visible_env)
    else:
        _logger.info("Using all visible GPUs with memory growth=%s.", memory_growth)
