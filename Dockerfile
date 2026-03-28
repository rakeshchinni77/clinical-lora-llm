FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PIP_NO_CACHE_DIR=1

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN python -m pip install --upgrade pip && \
	python -m pip install -r /workspace/requirements.txt

COPY . /workspace

# Run with bash explicitly so host mount permissions do not block script execution.
ENTRYPOINT ["/bin/bash", "-lc", "./src/main.sh"]
