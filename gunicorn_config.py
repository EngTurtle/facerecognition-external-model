import os

bind = f"[::]:{os.getenv('PORT', '5000')}"
accesslog = "-"
access_log_format = "%(h)s %(l)s %(u)s %(t)s '%(r)s' %(s)s %(b)s in %(M)sms"

workers = int(os.getenv("GUNICORN_WORKERS", 1))
threads = int(os.getenv("GUNICORN_THREADS", 1))

# Timeout for processing requests (in seconds)
timeout = int(os.getenv("REQ_TIMEOUT", 300))

# Worker class
worker_class = os.getenv("WORKER_CLASS", "sync")

# Preload app for better memory efficiency when using multiple workers
preload_app = bool(os.getenv("PRELOAD_APP", "true").lower() == "true")

# Maximum requests a worker will process before restarting (helps prevent memory leaks)
max_requests = int(os.getenv("MAX_REQUESTS", 1000))
max_requests_jitter = int(os.getenv("MAX_REQUESTS_JITTER", 50))


def on_starting(server):
    """Called just before the master process is initialized"""
    # Check and create temp directory
    temp_dir = "images"
    if not os.path.exists(temp_dir):
        server.log.info(f"Creating directory for temporary files: {temp_dir}")
        os.makedirs(temp_dir, exist_ok=True)
    
    # Clean old files if they exist
    server.log.info("Cleaning up old temporary files")
    for filename in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            server.log.error(f"Error deleting {file_path}: {e}")


def post_worker_init(worker):
    """Called after a worker has been forked"""
    worker.log.info(f"Worker {worker.pid} initialized")
    
    # Optionally preload models here if PRELOAD_MODELS is set
    if os.getenv("PRELOAD_MODELS", "false").lower() == "true":
        worker.log.info(f"Preloading models in worker {worker.pid}")
        try:
            from facerecognition_insightface import load_insightface_models
            load_insightface_models()
            worker.log.info(f"Models preloaded successfully in worker {worker.pid}")
        except Exception as e:
            worker.log.error(f"Failed to preload models in worker {worker.pid}: {e}")


def worker_exit(server, worker):
    """Called when a worker is exited"""
    server.log.info(f"Worker {worker.pid} exited")
