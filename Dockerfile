FROM python:3.12-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (required by deepfilternet)
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install uv
RUN pip install uv

WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY . .

# Install PyTorch CPU first (locked, before everything else)
# DeepFilterNet needs torch to already be present when it installs
RUN uv pip install --system torch torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Install deepfilternet
RUN uv pip install --system deepfilternet

# Install rest of project dependencies
RUN uv pip install --system .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "server"]