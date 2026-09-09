import json
import os
import sys
import duckdb
import pandas as pd
import requests

STATE_FILE = "data/state/team_ratings.json"
PARQUET_FILE = "data/processed/teams.parquet"
DUCKDB_FILE = "data/processed/cfb_data.duckdb"

FBS_DEFAULT_CATALOG = [
    {"school": "Georgia", "conference": "SEC", "rating": 1860.0, "pace": 26.2, "archetype": "standard", "ret_prod": 0.78, "portal_score": 88.5, "qb_snaps": 850},
    {"school": "Alabama", "conference": "SEC", "rating": 1835.0, "pace": 24.8, "archetype": "standard", "ret_prod": 0.69, "portal_score": 92.0, "qb_snaps": 620},
    {"school": "Ohio State", "conference": "Big Ten", "rating": 1850.0, "pace": 25.1, "archetype": "standard", "ret_prod": 0.74, "portal_score": 90.5, "qb_snaps": 920},
    {"school": "Texas", "conference": "SEC", "rating": 1825.0, "pace": 25.4, "archetype": "standard", "ret_prod": 0.72, "portal_score": 89.0, "qb_snaps": 780},
    {"school": "Oregon", "conference": "Big Ten", "rating": 1815.0, "pace": 23.9, "archetype": "standard", "ret_prod": 0.71, "portal_score": 87.0, "qb_snaps": 800},
    {"school": "Penn State", "conference": "Big Ten", "rating": 1780.0, "pace": 26.5, "archetype": "standard", "ret_prod": 0.76, "portal_score": 81.0, "qb_snaps": 700},
    {"school": "Notre Dame", "conference": "Independent", "rating": 1775.0, "pace": 25.8, "archetype": "standard", "ret_prod": 0.73, "portal_score": 84.0, "qb_snaps": 650},
    {"school": "Tennessee", "conference": "SEC", "rating": 1765.0, "pace": 20.2, "archetype": "air_raid", "ret_prod": 0.67, "portal_score": 86.0, "qb_snaps": 580},
    {"school": "LSU", "conference": "SEC", "rating": 1760.0, "pace": 24.5, "archetype": "standard", "ret_prod": 0.64, "portal_score": 89.5, "qb_snaps": 610},
    {"school": "Ole Miss", "conference": "SEC", "rating": 1750.0, "pace": 20.5, "archetype": "air_raid", "ret_prod": 0.61, "portal_score": 93.0, "qb_snaps": 720},
    {"school": "Clemson", "conference": "ACC", "rating": 1740.0, "pace": 25.0, "archetype": "standard", "ret_prod": 0.81, "portal_score": 60.0, "qb_snaps": 840},
    {"school": "Michigan", "conference": "Big Ten", "rating": 1745.0, "pace": 27.5, "archetype": "standard", "ret_prod": 0.62, "portal_score": 82.0, "qb_snaps": 450},
    {"school": "Miami", "conference": "ACC", "rating": 1730.0, "pace": 24.1, "archetype": "portal_heavy", "ret_prod": 0.63, "portal_score": 91.0, "qb_snaps": 890},
    {"school": "Texas A&M", "conference": "SEC", "rating": 1720.0, "pace": 25.2, "archetype": "standard", "ret_prod": 0.68, "portal_score": 85.0, "qb_snaps": 530},
    {"school": "USC", "conference": "Big Ten", "rating": 1710.0, "pace": 23.5, "archetype": "air_raid", "ret_prod": 0.64, "portal_score": 88.0, "qb_snaps": 550},
    {"school": "Oklahoma", "conference": "SEC", "rating": 1715.0, "pace": 22.8, "archetype": "standard", "ret_prod": 0.66, "portal_score": 86.5, "qb_snaps": 500},
    {"school": "Florida State", "conference": "ACC", "rating": 1690.0, "pace": 24.7, "archetype": "portal_heavy", "ret_prod": 0.58, "portal_score": 90.0, "qb_snaps": 600},
    {"school": "Utah", "conference": "Big 12", "rating": 1705.0, "pace": 26.8, "archetype": "standard", "ret_prod": 0.79, "portal_score": 75.0, "qb_snaps": 820},
    {"school": "Kansas State", "conference": "Big 12", "rating": 1680.0, "pace": 26.1, "archetype": "standard", "ret_prod": 0.75, "portal_score": 72.0, "qb_snaps": 640},
    {"school": "Oklahoma State", "conference": "Big 12", "rating": 1660.0, "pace": 24.3, "archetype": "standard", "ret_prod": 0.77, "portal_score": 74.0, "qb_snaps": 750},
    {"school": "Army", "conference": "AAC", "rating": 1495.0, "pace": 31.8, "archetype": "triple_option", "ret_prod": 0.82, "portal_score": 50.0, "qb_snaps": 550},
    {"school": "Navy", "conference": "AAC", "rating": 1470.0, "pace": 31.2, "archetype": "triple_option", "ret_prod": 0.79, "portal_score": 50.0, "qb_snaps": 480},
    {"school": "Air Force", "conference": "Mountain West", "rating": 1485.0, "pace": 31.5, "archetype": "triple_option", "ret_prod": 0.76, "portal_score": 50.0, "qb_snaps": 420},
    {"school": "Texas Tech", "conference": "Big 12", "rating": 1590.0, "pace": 20.8, "archetype": "air_raid", "ret_prod": 0.65, "portal_score": 82.0, "qb_snaps": 610},
    {"school": "Boise State", "conference": "Mountain West", "rating": 1640.0, "pace": 24.9, "archetype": "standard", "ret_prod": 0.75, "portal_score": 78.0, "qb_snaps": 670},
    {"school": "Memphis", "conference": "AAC", "rating": 1610.0, "pace": 23.8, "archetype": "standard", "ret_prod": 0.73, "portal_score": 76.0, "qb_snaps": 710},
    {"school": "Liberty", "conference": "CUSA", "rating": 1595.0, "pace": 25.5, "archetype": "standard", "ret_prod": 0.72, "portal_score": 75.0, "qb_snaps": 630},
    {"school": "App State", "conference": "Sun Belt", "rating": 1580.0, "pace": 24.2, "archetype": "standard", "ret_prod": 0.70, "portal_score": 73.0, "qb_snaps": 590},
    {"school": "Tulane", "conference": "AAC", "rating": 1590.0, "pace": 26.0, "archetype": "standard", "ret_prod": 0.68, "portal_score": 77.0, "qb_snaps": 540}
]

def fetch_cfbd_teams(api_key: str):
    """Attempts to pull live FBS teams from CFBD API."""
    url = "https://api.collegefootballdata.com/teams/fbs"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                print(f"[CFBD API] Successfully pulled {len(data)} FBS teams from live endpoint.")
                parsed = []
                for t in data:
                    name = t.get("school", "Unknown")
                    conf = t.get("conference", "Independent")
                    # Assign baseline ratings from catalog or sensible defaults
                    base_match = next((item for item in FBS_DEFAULT_CATALOG if item["school"].lower() == name.lower()), None)
                    if base_match:
                        parsed.append(base_match)
                    else:
                        parsed.append({
                            "school": name,
                            "conference": conf,
                            "rating": 1520.0,
                            "pace": 25.5,
                            "archetype": "standard",
                            "ret_prod": 0.70,
                            "portal_score": 75.0,
                            "qb_snaps": 500
                        })
                return parsed
        print(f"[CFBD API] Response code {resp.status_code}. Using built-in FBS dataset.")
    except Exception as e:
        print(f"[CFBD API] Live fetch error: {e}. Falling back to default FBS dataset.")
    return FBS_DEFAULT_CATALOG

def update_duckdb_and_parquet(team_records: list):
    """Writes team metrics into Apache Parquet and an in-process DuckDB database."""
    os.makedirs("data/processed", exist_ok=True)
    df = pd.DataFrame(team_records)

    # 1. Write Apache Parquet
    df.to_parquet(PARQUET_FILE, index=False)
    print(f"[PARQUET] Successfully serialized {len(df)} teams to {PARQUET_FILE}")

    # 2. Write DuckDB Table
    con = duckdb.connect(DUCKDB_FILE)
    con.execute("CREATE OR REPLACE TABLE fbs_teams AS SELECT * FROM df")
    count = con.execute("SELECT count(*) FROM fbs_teams").fetchone()[0]
    con.close()
    print(f"[DUCKDB] Table 'fbs_teams' verified with {count} rows in {DUCKDB_FILE}")

def sync_state_file(team_records: list):
    """Synchronizes ingested teams into the persistent team_ratings.json state."""
    os.makedirs("data/state", exist_ok=True)
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
    else:
        state = {
            "version": "2.0",
            "base_k_factor": 32.0,
            "mab_settings": {"exploration_constant": 1.414, "brier_ceiling": 0.2200, "decay_rate": 0.90},
            "archetypes": {
                "triple_option": {"k_factor": 54.0, "recent_brier_scores": [0.245, 0.238, 0.251], "kelly_multiplier": 0.40, "pulls": 14},
                "air_raid": {"k_factor": 32.0, "recent_brier_scores": [0.182, 0.179, 0.185], "kelly_multiplier": 1.00, "pulls": 22},
                "standard": {"k_factor": 32.0, "recent_brier_scores": [0.191, 0.188, 0.194], "kelly_multiplier": 1.00, "pulls": 65},
                "portal_heavy": {"k_factor": 44.0, "recent_brier_scores": [0.231, 0.224, 0.229], "kelly_multiplier": 0.60, "pulls": 18}
            },
            "teams": {}
        }

    current_teams = state.get("teams", {})
    for item in team_records:
        school = item["school"]
        if school not in current_teams:
            current_teams[school] = {
                "rating": item["rating"],
                "pace": item["pace"],
                "archetype": item["archetype"],
                "returning_production": item["ret_prod"],
                "transfer_inflow_rating": item["portal_score"],
                "qb_experience_snaps": item["qb_snaps"],
                "qb_backup_dropoff": 6.0,
                "rest_days": 7,
                "travel_miles": 0,
                "lookahead_flag": False,
                "letdown_flag": False
            }
        else:
            # Update dynamic metrics while preserving existing ratings
            current_teams[school]["returning_production"] = item["ret_prod"]
            current_teams[school]["transfer_inflow_rating"] = item["portal_score"]
            current_teams[school]["qb_experience_snaps"] = item["qb_snaps"]

    state["teams"] = current_teams
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    print(f"[STATE SYNC] Synced {len(current_teams)} teams into {STATE_FILE}")

if __name__ == "__main__":
    cfbd_key = os.getenv("CFBD_API_KEY", "PLACEHOLDER")
    teams_data = fetch_cfbd_teams(cfbd_key)
    update_duckdb_and_parquet(teams_data)
    sync_state_file(teams_data)
    print("[DATA PIPELINE COMPLETE] All storage engines synchronized successfully.")
