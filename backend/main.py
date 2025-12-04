from typing import Union

from fastapi import FastAPI

app = FastAPI()


@app.get("/api/v1/")
async def read_root():
    return {"Hello": "World"}


@app.get("/api/v1/items/{item_id}")
async def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}