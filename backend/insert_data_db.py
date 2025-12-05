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


def insert_aggregated_mongo_db(aggregated_legit_data: pd.DataFrame) -> None:
    data = []
    dbname = get_database()
    print("Inserting aggregated legit data into MongoDB...")
    print(f"Total records to insert: {len(aggregated_legit_data)}")
    print(aggregated_legit_data.head())
    print(dbname)
    collection_name = dbname["trasite"]
    for _, row in aggregated_legit_data.iterrows():

        data.append({
            "id_room": row['id_room'],
            "legit_count": int(row['legit_count']),
            "fake_count": int(row['fake_count']),
            "latitude": row.get('latitude', None),
            "longitude": row.get('longitude', None),
            "neighbourhood": row.get('neighbourhood', None),
            "name": row.get('name', None),
            "room_type": row.get('room_type', None),
            "last_review": row.get('last_review', None),
            "reviews_per_month": row.get('reviews_per_month', None),
            "host_id": row.get('host_id', None),
            "host_name": row.get('host_name', None),
            "price": row.get('price', None)
        })
    collection_name.insert_many(data)
    


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
    return geo_frame[['id_room', 'latitude', 'longitude', 'neighbourhood', 'name', 'room_type', 'last_review', 'reviews_per_month', 'host_id', 'host_name', 'price']]

def merge_geo_and_legit_data(geo_data: pd.DataFrame, legit_data: pd.DataFrame) -> pd.DataFrame:
    print("Merging geographic data with legitimacy data...")
    merged_data = pd.merge(geo_data, legit_data, on='id_room', how='inner', validate="many_to_many")
    print(f"Merged data has {len(merged_data)} records.")
    return merged_data


def main():
    legit_data = legit_data_from_csv("./backend/data/reviews_en_clean_classified.csv")
    geo_data = load_geo_data_from_csv("./backend/data/listings.csv")
    aggregated_legit_data = merge_geo_and_legit_data(geo_data, legit_data)
    insert_aggregated_mongo_db(aggregated_legit_data)

if __name__ == "__main__":
    main()
