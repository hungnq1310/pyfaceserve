from fastapi import FastAPI
from faceserve.api import v2
from faceserve.api.v2 import FACES_IMG_DIR
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.mount("/imgs", StaticFiles(directory=FACES_IMG_DIR), name="imgs")
app.include_router(v2.router)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello Bigger Applications!"}