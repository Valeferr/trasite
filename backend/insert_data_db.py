import os
import pandas as pd
from pymongo import MongoClient

try:
    CONNECTION_STRING = os.getenv("CONNECTION_STRING")
except ImportError:
    exit("CONNECTION_STRING not found in environment variables")

def get_database():
   client = MongoClient(CONNECTION_STRING)
   return client['trasite_db']


def insert_aggregated_legit_mongo_db(aggregated_legit_data: pd.DataFrame) -> None:
    for _, row in aggregated_legit_data.iterrows():
        dbname = get_database()
        collection_name = dbname["trasite"]
        data = {
            "id_room": row['id_room'],
            "legit_count": int(row['legit_count']),
            "fake_count": int(row['fake_count']),
            "latitude": row.get('latitude', None),
            "longitude": row.get('longitude', None),
            "neighbourhood": row.get('neighbourhood', None),
        }
        collection_name.insert_one(data)
    


def legit_data_from_csv(file_path: str) -> pd.DataFrame:
    frame = pd.read_csv(file_path)
    frame = frame.dropna(subset=["id_room"])
    return frame.groupby('id_room')['legit'].agg(
        legit_count='sum',
        fake_count=lambda x: x.size - x.sum()
    ).reset_index()

def load_geo_data_from_csv(file_path: str) -> pd.DataFrame:
    geo_frame = pd.read_csv(file_path)
    geo_frame = geo_frame.dropna(subset=["id_room", "latitude", "longitude", "neighbourhood"])
    return geo_frame[['id_room', 'latitude', 'longitude', 'neighbourhood']]

def merge_geo_and_legit_data(geo_data: pd.DataFrame, legit_data: pd.DataFrame) -> pd.DataFrame:
    merged_data = pd.merge(geo_data, legit_data, on='id_room', how='inner', validate="many_to_many")
    return merged_data


def main():
    legit_data = legit_data_from_csv("./backend/data/reviews_en_clean.csv")
    geo_data = load_geo_data_from_csv("./backend/data/geo_data.csv")
    aggregated_legit_data = merge_geo_and_legit_data(geo_data, legit_data)
    insert_aggregated_legit_mongo_db(aggregated_legit_data)

if __name__ == "__main__":
    main()
