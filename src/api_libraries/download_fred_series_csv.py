import requests
import pandas as pd
import json
import os

FRED_API_KEY = "dbe0b0309bf7116c11f595d3e1dffd53"

# Load your massive FRED catalog JSON
with open("fred_catalog.json", "r") as f:
    fred_catalog = json.load(f)

fred_series_db = fred_catalog["series"]

def search_series_by_title(keyword):
    """Return list of matching series IDs and titles"""
    results = []
    for sid, data in fred_series_db.items():
        if keyword.lower() in data["title"].lower():
            results.append((sid, data["title"]))
    return results

def download_fred_series_csv(series_id, start_date="1950-01-01", end_date=None, output_file=None):
    print(f"📥 Downloading series: {series_id}")

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start_date
    }
    if end_date:
        params["observation_end"] = end_date

    r = requests.get(url, params=params)
    if r.status_code != 200:
        print(f"❌ Failed to download {series_id} — Status code {r.status_code}")
        return

    data = r.json().get("observations", [])
    if not data:
        print(f"⚠️ No data found for series {series_id}")
        return

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df[series_id] = pd.to_numeric(df['value'], errors='coerce')
    df = df[[series_id]]

    if not output_file:
        title = fred_series_db.get(series_id, {}).get("title", series_id)
        safe_title = title.replace(" ", "_").replace("/", "_")
        output_file = f"{safe_title}.csv"

    df.to_csv(output_file)
    print(f"✅ Saved to {output_file}")

def get_series_csv(query):
    """
    Query can be a series ID or a title keyword
    """
    if query in fred_series_db:
        # Direct ID
        download_fred_series_csv(query)
    else:
        matches = search_series_by_title(query)
        if not matches:
            print("❌ No matches found.")
            return
        print(f"🔍 Found {len(matches)} match(es):")
        for i, (sid, title) in enumerate(matches[:5]):
            print(f"  [{i}] {sid}: {title}")

        # Auto-pick first match
        best_id = matches[0][0]
        download_fred_series_csv(best_id)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        get_series_csv(query)
    else:
        print("❌ No query provided. Usage: python download_fred_series_csv.py GDP")

# Example usage:
# get_series_csv("NROU")
# get_series_csv("Natural Rate of Unemployment")
