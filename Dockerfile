FROM python:3.10-slim

WORKDIR /app

# System dependencies for OpenCV, MediaPipe, video processing, and X11 display forwarding
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1-mesa-glx \
    ffmpeg \
    libx11-6 \
    libxcb1 \
    libxkbcommon-x11-0 \
    x11-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install PyTorch CPU (portable, no CUDA needed — swap index URL for GPU builds)
RUN pip install --no-cache-dir \
    torch==2.9.1 torchvision==0.24.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (torch already satisfied above)
RUN grep -vE "^(torch|torchvision)==" requirements.txt > /tmp/reqs.txt && \
    pip install --no-cache-dir -r /tmp/reqs.txt

# Copy project files
COPY . .

CMD ["python", "DEMO.py"]
