# Python Face Serving
Face recognition serving APIs based on FastAPI and Onnxruntime.
## License
[AGPL 3.0](/LICENSE)<br>
Copyright &copy; 2024 [Hieu Pham](https://github.com/hieupth). All rights reserved.

## Prequistance
```
!wget https://huggingface.co/datasets/hero-nq1310/stuffhub/resolve/main/pyfaceserve-models.zip
!unzip pyfaceserve-model.zip
```

## One step with docker compose
```bash
docker-compose up
```

## Customizaition in `doker-compose.yaml`

In `triton` service, you can run with cpu/gpus by commenting/uncommenting these following lines:
```
deploy:
    resources:
        reservations:
          devices:
          - driver: nvidia
            device_ids: ['1']
            capabilities: [gpu]
```

Change environment variables in `pyfaceserve` service when need to change **threshold**, **model_name**, **API**:
- TRITON_URL=localhost:6000
- DETECTION_NAME=yolov7-hf-v1
- SPOOFING_NAME=spoofer
- RECOGNITION_NAME=ghostfacenet
- DETECTION_THRESH=0.7
- SPOOFING_THRESH=0.4
- RECOGNITION_THRESH=0.4
- QDRANT_URL=localhost:6333
- IMG_DIR=face_images

## How to scale with different real case
Create new docker image with new value of `DB_NAME` and `volumes` in `docker-compose.yaml` file.
```
    image: heronq02/pyfaceserve:v1.0.0
    environment:
      ...
      - IMG_DIR=<new_path>
      - DB_NAME=<new-name-collection>
    volumes:
      - ${PWD}/<new_path>:/<new_path>
    ...
    command: fastapi run main.py --port <new_port>
```
`DB_NAME` will set database hard-code with different port API. 
