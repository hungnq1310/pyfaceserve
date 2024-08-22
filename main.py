from fastapi import FastAPI
from api import v1
from api.v1 import FACES_IMG_DIR
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/imgs", StaticFiles(directory=FACES_IMG_DIR), name="imgs")
app.include_router(v1.router)

@app.get("/")
async def root():
    return {"message": "Hello Bigger Applications!"}