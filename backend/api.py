import math
import os
from fastapi import FastAPI
from typing import TypedDict, Optional
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
import certifi

class Trasite(TypedDict):
    id_room: int
    legit_count: int
    fake_count: int
    latitude: float
    longitude: float
    neighbourhood: str
    name: str
    room_type: str
    last_review: str
    reviews_per_month: float
    host_id: int
    host_name: str
    price: int
    fake_reviews: list

app = FastAPI(json_encoders={float: lambda x: None if math.isnan(x) else x})

load_dotenv()
CONNECTION_STRING: Optional[str] = os.environ.get("CONNECTION_STRING")

if not CONNECTION_STRING:
    exit("ERROR: CONNECTION_STRING environment variable not found.")

def get_collection() -> Collection[Trasite]:
    client = MongoClient(
        CONNECTION_STRING,
        tls=True,
        tlsCAFile=certifi.where()
    )
    db = client['trasite_db']
    collection: Collection[Trasite] = db.get_collection('trasite')
    return collection


@app.get("/legit_counts/")
def get_legit_counts_in_range(min_value: int, max_value: int):
    collection = get_collection()

    cursor = collection.find(
        {"legit_count": {"$gte": min_value, "$lte": max_value}},
        {
            "_id": 0,
            "name": 1,
            "room_type": 1,
            "last_review": 1,
            "reviews_per_month": 1,
            "host_id": 1,
            "host_name": 1,
            "price": 1
        }
    )

    results = [clean_nan(doc) for doc in cursor]

    return {"count": len(results), "results": results}

@app.get("/hosts/")
def get_legit_fake_count(host_id: int):
    collection = get_collection()

    result = collection.find_one(
        {"host_id": host_id},
        {
            "_id": 0,
            "id_room": 1,
            "legit_count": 1,
            "fake_count": 1,
            "name": 1,
            "room_type": 1,
            "last_review": 1,
            "reviews_per_month": 1,
            "host_id": 1,
            "host_name": 1,
            "price": 1,
            "fake_reviews": 1
        }
    )

    return result

@app.get("/distance/")
def get_near_structure(
    lat: float,
    lon: float,
    delta: float = 0.01
):
    collection = get_collection()

    cursor = collection.find(
        {
            "latitude": {
                "$gte": lat - delta,
                "$lte": lat + delta
            },
            "longitude": {
                "$gte": lon - delta,
                "$lte": lon + delta
            }
        },
        {
            "_id": 0,
            "id_room": 1,
            "legit_count": 1,
            "fake_count": 1,
            "latitude": 1,
            "longitude": 1,
            "neighbourhood": 1,
            "name": 1,
            "room_type": 1,
            "last_review": 1,
            "reviews_per_month": 1,
            "host_id": 1,
            "host_name": 1,
            "price": 1,
            "fake_reviews": 1
        }
    )

    results = [clean_nan(doc) for doc in cursor]

    return {
        "count": len(results),
        "results": results
    }



def clean_nan(value):
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        else:
            return value
    elif isinstance(value, dict):
        return {k: clean_nan(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [clean_nan(v) for v in value]
    else:
        return value