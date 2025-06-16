FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    libomp-dev \
    git \
  && curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh \
      | bash \
  && apt-get update && apt-get install -y git-lfs \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

# RUN git config --global url."https://github.com/".insteadOf "git@github.com:" \
#   && git lfs install \
#   && git lfs pull --include="data/models/best_weights.pth,data/indexes/*"

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]