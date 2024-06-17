from fastapi import FastAPI
from api import v1
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/imgs", StaticFiles(directory="imgs"), name="imgs")
app.include_router(v1.router)

@app.get("/")
async def root():
    return {"message": "Hello Bigger Applications!"}