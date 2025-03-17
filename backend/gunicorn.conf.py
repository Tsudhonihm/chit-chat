import os

workers = int(os.getenv('WEB_CONCURRENCY', 1))
threads = int(os.getenv('PYTHON_MAX_THREADS', 1))
timeout = 120
bind = f"0.0.0.0:{os.getenv('PORT', '3000')}"
worker_class = 'sync'
max_requests = 1000
max_requests_jitter = 50