
FROM python:3.9-slim AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git gcc g++ cmake make libffi-dev python3-dev \
      libglib2.0-0 libsm6 libxrender1 libxext6 libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

RUN pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1

WORKDIR /detectron2
RUN git clone https://github.com/facebookresearch/detectron2.git . && \
    python -m pip wheel . -w /wheels

FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt 

COPY --from=builder /wheels/detectron2-*.whl /tmp/
RUN pip install --no-cache-dir /tmp/detectron2-*.whl

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
