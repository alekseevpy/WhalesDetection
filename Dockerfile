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
#  && git lfs install \
#  && git lfs pull --include="data/models/best_weights.pth,data/indexes/*"

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN echo '#!/bin/bash\n\
set -e\n\
echo "=== Запуск тестов перед стартом сервиса ==="\n\
pytest tests --maxfail=1 --disable-warnings -q\n\
echo "=== Тесты успешно пройдены, запускаем API ==="\n\
exec uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload' > /entrypoint.sh

RUN chmod +x /entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]