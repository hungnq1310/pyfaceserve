### Pip stage ###

FROM python:3.10-slim as compiler
ENV PYTHONUNBUFFERED 1

ADD . .

RUN python -m venv /venv

ENV PATH="/venv/bin:$PATH"

RUN pip install nvidia-cuda-runtime-cu11 && \
    pip install fastapi[standard] uvicorn python-multipart attrdict && \
    pip install pillow numpy==1.26 grpcio python-dotenv scikit-image

FROM python:3.10-slim as runner

COPY --from=compiler /venv /venv
COPY faceserve /faceserve

ENV PATH="/venv/bin:$PATH"

CMD fastapi run main.py --port 6999