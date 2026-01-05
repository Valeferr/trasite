import os
import pandas as pd
from pymongo import MongoClient
from typing import Optional, List, Dict, Any

CONNECTION_STRING = os.getenv("CONNECTION_STRING")
if not CONNECTION_STRING:
    raise EnvironmentError("CONNECTION_STRING not found in environment variables")

DB_NAME = "trasite_db"
COLLECTION_NAME = "trasite"


def get_database():
    client = MongoClient(CONNECTION_STRING)
    return client[DB_NAME]


def load_reviews(file_path: str) -> pd.DataFrame:
    reviews = pd.read_csv(file_path)
    reviews = reviews.dropna(subset=["id_room"])
    reviews["id_room"] = reviews["id_room"].astype(int, errors="ignore")
    if reviews["legit"].dtype != bool:
        reviews["legit"] = reviews["legit"].map(
            lambda v: v if isinstance(v, bool) else str(v).strip().lower() in ("true", "1", "yes")
        )
    return reviews


def aggregate_legit_counts(reviews: pd.DataFrame) -> pd.DataFrame:
    aggregated = (
        reviews.groupby("id_room")["legit"]
        .agg(
            legit_count="sum",
            fake_count=lambda x: int(x.size - x.sum())
        )
        .reset_index()
    )
    aggregated["legit_count"] = aggregated["legit_count"].astype(int)
    aggregated["fake_count"] = aggregated["fake_count"].astype(int)
    return aggregated


def get_fake_reviews_by_room(reviews: pd.DataFrame, id_room: int) -> pd.DataFrame:
    return reviews[(reviews["id_room"] == id_room) & (reviews["legit"] == False)]


def load_geo_data_from_csv(file_path: str) -> pd.DataFrame:
    geo_frame = pd.read_csv(file_path)
    geo_frame = geo_frame.dropna(subset=["id_room", "latitude", "longitude", "neighbourhood"])
    geo_frame["id_room"] = geo_frame["id_room"].astype(int, errors="ignore")
    cols = [
        "id_room", "latitude", "longitude", "neighbourhood",
        "name", "room_type", "last_review", "reviews_per_month",
        "host_id", "host_name", "price"
    ]
    return geo_frame[[c for c in cols if c in geo_frame.columns]]


def merge_geo_and_counts(geo_data: pd.DataFrame, counts_data: pd.DataFrame) -> pd.DataFrame:
    return pd.merge(geo_data, counts_data, on="id_room", how="inner", validate="many_to_many")


def _nan_to_none(v: Any) -> Any:
    return None if pd.isna(v) else v


def _fake_reviews_to_docs(df: pd.DataFrame, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    needed = ["review", "confidence_score"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in reviews dataframe: {missing}")

    out = df[needed].copy()
    if limit is not None:
        out = out.head(limit)

    docs = out.to_dict(orient="records")
    for d in docs:
        for k, v in d.items():
            d[k] = _nan_to_none(v)
    return docs


def insert_aggregated_mongo_db(
    aggregated_data: pd.DataFrame,
    reviews: pd.DataFrame,
    fake_reviews_limit: Optional[int] = None
) -> None:
    db = get_database()
    collection = db[COLLECTION_NAME]

    data = []
    for _, row in aggregated_data.iterrows():
        id_room = int(row["id_room"])
        fake_df = get_fake_reviews_by_room(reviews, id_room)

        data.append({
            "id_room": id_room,
            "legit_count": int(row.get("legit_count", 0)),
            "fake_count": int(row.get("fake_count", 0)),
            "latitude": _nan_to_none(row.get("latitude")),
            "longitude": _nan_to_none(row.get("longitude")),
            "neighbourhood": _nan_to_none(row.get("neighbourhood")),
            "name": _nan_to_none(row.get("name")),
            "room_type": _nan_to_none(row.get("room_type")),
            "last_review": _nan_to_none(row.get("last_review")),
            "reviews_per_month": _nan_to_none(row.get("reviews_per_month")),
            "host_id": _nan_to_none(row.get("host_id")),
            "host_name": _nan_to_none(row.get("host_name")),
            "price": _nan_to_none(row.get("price")),
            "fake_reviews": _fake_reviews_to_docs(fake_df, fake_reviews_limit)
        })

    if data:
        collection.insert_many(data)


def main():
    reviews = load_reviews("./backend/data/reviews_en_clean_classified.csv")
    counts = aggregate_legit_counts(reviews)
    geo_data = load_geo_data_from_csv("./backend/data/listings.csv")
    aggregated = merge_geo_and_counts(geo_data, counts)
    insert_aggregated_mongo_db(aggregated, reviews)


if __name__ == "__main__":
    main()
