import os
from pymongo import MongoClient

try:
    CONNECTION_STRING = os.getenv("CONNECTION_STRING")
except ImportError:
    exit("CONNECTION_STRING not found in environment variables")

def get_database():
   client = MongoClient(CONNECTION_STRING)
   return client['user_shopping_list']



def insert_geo_mongo_db(geo_data: dict) -> None:
    dbname = get_database()
    collection_name = dbname["trasite_data"]
    collection_name.insert_one(geo_data)


def create_geo_data(ip: str,  latitude: float, longitude: float, neighbourhood: str) -> dict:
    return {
        "id_room": ip,
        "latitude": latitude,
        "longitude": longitude,
        "neighbourhood": neighbourhood,
    }
