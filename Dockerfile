FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    fontconfig \
    fonts-dejavu-core \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /usr/share/fonts/truetype/custom && \
    cp fonts/*.ttf /usr/share/fonts/truetype/custom/ 2>/dev/null || true && \
    fc-cache -fv

RUN mkdir -p outputs Audio_Voice

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}