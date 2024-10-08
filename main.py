from fastapi import FastAPI
from faceserve.api import v2
from faceserve.api.v2 import FACES_IMG_DIR
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/imgs", StaticFiles(directory=FACES_IMG_DIR), name="imgs")
app.include_router(v2.router)

@app.get("/")
async def root():
    return {"message": "Hello Bigger Applications!"}