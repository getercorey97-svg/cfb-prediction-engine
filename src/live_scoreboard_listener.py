import json
import os
import requests
import subprocess
from datetime import datetime

PROCESSED_GAMES_FILE = "data/state/processed_games.json"
CFBD_API_KEY = os.getenv("CFBD_API_KEY", "")

def load_processed_games() -> list:
    """Loads the ledger of already evaluated games to prevent double-counting."""
    if os.path.exists(PROCESSED_GAMES_FILE):
        with open(PROCESSED_GAMES_FILE, "r") as f:
            return json.load(f)
    return []

def save_processed_games(processed_list: list):
    """Commits the updated ledger to disk."""
    os.makedirs(os.path.dirname(PROCESSED_GAMES_FILE), exist_ok=True)
    with open(PROCESSED_GAMES_FILE, "w") as f:
        json.dump(processed_list, f, indent=2)

def fetch_live_scoreboard():
    """Queries the CFBD API for current live and completed games."""
    if not CFBD_API_KEY or CFBD_API_KEY == "DISABLED":
        print("[API ERROR] CFBD_API_KEY is missing or disabled. Cannot poll live games.")
        return []

    headers = {"Authorization": f"Bearer {CFBD_API_KEY}", "Accept": "application/json"}
    url = "https://api.collegefootballdata.com/scoreboard"
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"[API ERROR] CFBD returned status {response.status_code}")
            return []
    except Exception as e:
        print(f"[NETWORK ERROR] Failed to reach CFBD Scoreboard: {e}")
        return []

def process_completed_games():
    """Identifies newly completed games and executes the Kalman update."""
    scoreboard = fetch_live_scoreboard()
    processed_games = load_processed_games()
    
    newly_processed = 0

    for game in scoreboard:
        game_id = str(game.get("id"))
        status = game.get("status", "")
        
        # Only process if the game is completely finished and hasn't been logged yet
        if status == "completed" and game_id not in processed_games:
            home_team = game.get("homeTeam", {}).get("name", "")
            away_team = game.get("awayTeam", {}).get("name", "")
            home_score = game.get("homeTeam", {}).get("points", 0)
            away_score = game.get("awayTeam", {}).get("points", 0)
            is_neutral = game.get("neutralSite", False)

            print(f"\n[NEW FINAL DETECTED] {away_team} ({away_score}) @ {home_team} ({home_score})")
            
            # Pipe the result directly into the Bayesian Kalman engine
            cmd = [
                "python", "src/self_correction.py",
                "--single-game",
                "--home", home_team,
                "--away", away_team,
                "--home-score", str(home_score),
                "--away-score", str(away_score)
            ]
            
            if is_neutral:
                cmd.append("--neutral")

            try:
                subprocess.run(cmd, check=True)
                processed_games.append(game_id)
                newly_processed += 1
            except subprocess.CalledProcessError as e:
                print(f"[EXECUTION ERROR] Kalman update failed for {game_id}: {e}")

    if newly_processed > 0:
        save_processed_games(processed_games)
        print(f"\n[POLLING COMPLETE] Successfully processed {newly_processed} new game(s).")
    else:
        print("\n[POLLING COMPLETE] No new completed games detected.")

if __name__ == "__main__":
    print(f"[{datetime.utcnow().isoformat()}] Initiating Live Scoreboard Polling Sequence...")
    process_completed_games()
