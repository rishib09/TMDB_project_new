"""TMDB 1970-2026 US Cinema Ingestion Pipeline.
Discovers and hydrates movie metadata, cast, director, keywords, and poster paths.
Expanded for comprehensive coverage (~10,000 movies).
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv

# Ensure stdout handles UTF-8 on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.database import MovieDatabase

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_READ_ACCESS_TOKEN = os.getenv("TMDB_READ_ACCESS_TOKEN")
BASE_URL = "https://api.themoviedb.org/3"


def get_headers() -> Dict[str, str]:
    if TMDB_READ_ACCESS_TOKEN:
        return {
            "Authorization": f"Bearer {TMDB_READ_ACCESS_TOKEN}",
            "Accept": "application/json",
        }
    return {"Accept": "application/json"}


def get_params(extra_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    params = {}
    if not TMDB_READ_ACCESS_TOKEN and TMDB_API_KEY:
        params["api_key"] = TMDB_API_KEY
    if extra_params:
        params.update(extra_params)
    return params


async def fetch_discovery_page(
    client: httpx.AsyncClient,
    year: int,
    page: int,
    semaphore: asyncio.Semaphore,
) -> List[Dict[str, Any]]:
    """Fetch a single page of US movies for a specific release year."""
    url = f"{BASE_URL}/discover/movie"
    params = get_params({
        "primary_release_year": year,
        "with_origin_country": "US",
        "sort_by": "popularity.desc",
        "include_adult": "false",
        "page": page,
    })

    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.get(url, headers=get_headers(), params=params, timeout=15.0)
                if response.status_code == 200:
                    data = response.json()
                    return data.get("results", [])
                elif response.status_code == 429:
                    await asyncio.sleep(2.0 * (attempt + 1))
            except (httpx.RequestError, httpx.TimeoutException):
                await asyncio.sleep(1.0 * (attempt + 1))
    return []


async def hydrate_movie_details(
    client: httpx.AsyncClient,
    movie_id: int,
    year: int,
    semaphore: asyncio.Semaphore,
) -> Optional[Dict[str, Any]]:
    """Fetch complete movie details including credits and keywords."""
    url = f"{BASE_URL}/movie/{movie_id}"
    params = get_params({"append_to_response": "credits,keywords"})

    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.get(url, headers=get_headers(), params=params, timeout=15.0)
                if response.status_code == 200:
                    raw = response.json()

                    # 1. Extract Director
                    crew = raw.get("credits", {}).get("crew", [])
                    director = next((m["name"] for m in crew if m.get("job") == "Director"), "")

                    # 2. Extract Top Cast (Top 10)
                    cast_raw = raw.get("credits", {}).get("cast", [])[:10]
                    cast_members = [
                        {
                            "name": c.get("name", ""),
                            "character": c.get("character", ""),
                            "order": c.get("order", 0),
                            "profile_path": c.get("profile_path"),
                        }
                        for c in cast_raw
                    ]

                    # 3. Extract Keywords
                    keywords_raw = raw.get("keywords", {}).get("keywords", [])
                    keywords = [k.get("name", "") for k in keywords_raw if k.get("name")]

                    # 4. Extract Genres
                    genres = [g.get("name", "") for g in raw.get("genres", []) if g.get("name")]

                    # Release Year Fallback
                    rel_date = raw.get("release_date", "")
                    rel_year = int(rel_date.split("-")[0]) if rel_date and "-" in rel_date else year

                    return {
                        "id": raw["id"],
                        "title": raw.get("title", ""),
                        "original_title": raw.get("original_title", ""),
                        "tagline": raw.get("tagline", ""),
                        "overview": raw.get("overview", ""),
                        "release_date": rel_date,
                        "release_year": rel_year,
                        "runtime": raw.get("runtime") or 0,
                        "vote_average": float(raw.get("vote_average") or 0.0),
                        "vote_count": int(raw.get("vote_count") or 0),
                        "popularity": float(raw.get("popularity") or 0.0),
                        "director": director,
                        "budget": int(raw.get("budget") or 0),
                        "revenue": int(raw.get("revenue") or 0),
                        "imdb_id": raw.get("imdb_id") or "",
                        "poster_path": raw.get("poster_path") or "",
                        "backdrop_path": raw.get("backdrop_path") or "",
                        "genres": genres,
                        "cast": cast_members,
                        "keywords": keywords,
                    }
                elif response.status_code == 429:
                    await asyncio.sleep(2.0 * (attempt + 1))
            except (httpx.RequestError, httpx.TimeoutException):
                await asyncio.sleep(1.0 * (attempt + 1))
    return None


async def ingest_tmdb(
    start_year: int = 1970,
    end_year: int = 2026,
    pages_per_year: int = 8,
    db_path: str = "data/tmdb_movies.db",
    json_path: str = "data/tmdb_movies.json",
) -> None:
    """Main ingestion coordinator fetching ~10,000 1970-2026 US cinema movies."""
    print(f"[INGEST] Starting TMDB Ingestion: Years {start_year} to {end_year} ({pages_per_year} pages/year = ~{ (end_year - start_year + 1) * pages_per_year * 20 } movies)...", flush=True)

    if not TMDB_API_KEY and not TMDB_READ_ACCESS_TOKEN:
        print("[ERROR] Neither TMDB_API_KEY nor TMDB_READ_ACCESS_TOKEN found in environment!", flush=True)
        return

    db = MovieDatabase(db_path)
    semaphore = asyncio.Semaphore(15)
    all_movies: List[Dict[str, Any]] = []

    async with httpx.AsyncClient() as client:
        for year in range(start_year, end_year + 1):
            discovery_tasks = [
                fetch_discovery_page(client, year, page, semaphore)
                for page in range(1, pages_per_year + 1)
            ]
            discovery_results = await asyncio.gather(*discovery_tasks)
            movie_ids = [m["id"] for page_res in discovery_results for m in page_res if "id" in m]

            # Hydrate discovered movies
            hydration_tasks = [
                hydrate_movie_details(client, mid, year, semaphore)
                for mid in movie_ids
            ]
            hydrated = await asyncio.gather(*hydration_tasks)
            valid_movies = [m for m in hydrated if m is not None]

            if valid_movies:
                db.upsert_movies_bulk(valid_movies)
                all_movies.extend(valid_movies)
                print(f"  [OK] Year {year}: Ingested {len(valid_movies)} movies (Total so far: {len(all_movies)})", flush=True)

    # Save offline JSON bundle
    json_file = Path(json_path)
    json_file.parent.mkdir(parents=True, exist_ok=True)
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(all_movies, f, indent=2, ensure_ascii=False)

    print("\n[COMPLETE] Ingestion Complete!", flush=True)
    print(f"  - Total Movies Saved: {db.count()}", flush=True)
    print(f"  - SQLite Database: {db_path}", flush=True)
    print(f"  - JSON Offline Bundle: {json_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest TMDB movies for 1970-2026")
    parser.add_argument("--start-year", type=int, default=1970)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--pages-per-year", type=int, default=8)
    parser.add_argument("--db-path", type=str, default="data/tmdb_movies.db")
    parser.add_argument("--json-path", type=str, default="data/tmdb_movies.json")
    args = parser.parse_args()

    asyncio.run(ingest_tmdb(
        start_year=args.start_year,
        end_year=args.end_year,
        pages_per_year=args.pages_per_year,
        db_path=args.db_path,
        json_path=args.json_path,
    ))
