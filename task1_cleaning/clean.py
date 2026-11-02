"""Task 1 - Data Cleaning & Preprocessing of the Netflix dataset."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "Dataset.csv"
OUT = ROOT / "data" / "netflix_clean.csv"

RATING_GROUP = {
    "TV-Y": "Kids", "TV-Y7": "Kids", "TV-Y7-FV": "Kids", "TV-G": "Kids", "G": "Kids",
    "TV-PG": "Family", "PG": "Family",
    "TV-14": "Teens", "PG-13": "Teens",
    "TV-MA": "Adults", "R": "Adults", "NC-17": "Adults",
    "NR": "Unrated", "UR": "Unrated",
}


def main():
    # Step 1: import
    df = pd.read_csv(RAW)
    print(f"Raw shape: {df.shape}")

    # Step 2: identify missing and duplicate records
    # The raw file has no NaNs; missing values are encoded as the string "Not Given".
    df = df.replace({"Not Given": np.nan})
    print("\nMissing values (after decoding 'Not Given'):")
    print(df.isna().sum()[df.isna().sum() > 0])

    for col in df.columns.drop("release_year"):
        df[col] = df[col].str.strip()

    exact_dups = df.duplicated().sum()
    content_dups = df.duplicated(subset=["title", "type", "release_year"]).sum()
    print(f"\nExact duplicate rows: {exact_dups}")
    print(f"Duplicate (title, type, release_year): {content_dups}")
    df = df.drop_duplicates().drop_duplicates(subset=["title", "type", "release_year"])

    # Step 3: handle nulls and inconsistent formats
    df["director"] = df["director"].fillna("Unknown")
    df["country"] = df["country"].fillna("Unknown")
    df["date_added"] = pd.to_datetime(df["date_added"], format="%m/%d/%Y", errors="coerce")
    df = df.dropna(subset=["date_added"])

    # Step 4: transform categorical and date-related columns
    df["year_added"] = df["date_added"].dt.year
    df["month_added"] = df["date_added"].dt.month
    df["month_name_added"] = df["date_added"].dt.month_name()
    df["day_of_week_added"] = df["date_added"].dt.day_name()
    df["years_to_netflix"] = (df["year_added"] - df["release_year"]).clip(lower=0)

    dur = df["duration"].str.extract(r"(\d+)\s*(\w+)")
    df["duration_value"] = dur[0].astype(int)
    df["duration_unit"] = dur[1].str.lower().replace({"season": "seasons"})
    df["duration_min"] = np.where(df["type"] == "Movie", df["duration_value"], np.nan)
    df["seasons"] = np.where(df["type"] == "TV Show", df["duration_value"], np.nan)

    df["rating_group"] = df["rating"].map(RATING_GROUP).fillna("Unrated")
    df["primary_genre"] = df["listed_in"].str.split(",").str[0].str.strip()
    df["n_genres"] = df["listed_in"].str.split(",").str.len()
    df["is_movie"] = (df["type"] == "Movie").astype(int)

    for col in ["type", "rating", "rating_group", "country", "primary_genre"]:
        df[col] = df[col].astype("category")

    # Step 5: save clean dataset
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nClean shape: {df.shape}  ->  {OUT}")
    print(df.head(3).T)


if __name__ == "__main__":
    main()
