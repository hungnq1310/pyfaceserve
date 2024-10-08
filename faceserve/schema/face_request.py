from pydantic import BaseModel, Field

class FaceRequest(BaseModel):
    base64images: list[str] = Field(
        default_factory=list,
    )