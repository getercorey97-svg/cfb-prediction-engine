import os
import sys
import datetime
import requests
from simulation_engine import run_monte_carlo_engine

CFBD_API_KEY = os.getenv("CFBD_API_KEY", "")

def get_today_games(target_date: str = None) -> list:
    """Fetches all FBS games scheduled for the target date from the CFBD API."""
    if not target_date:
        target_date = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    
    headers = {"Authorization": f"Bearer {CFBD_API_KEY}", "Accept": "application/json"}
    year = target_date.split("-")[0]
    
    url = f"https://api.collegefootballdata.com/games?year={year}&seasonType=regular"
    try:
        res = requests.get(url, headers=headers, timeout=15)
        if res.status_code != 200:
            print(f"[API ERROR] Failed to fetch schedule: HTTP {res.status_code}")
            return []
        
        all_games = res.json()
        # Filter games matching the specific calendar date
        daily_games = [
            g for g in all_games 
            if g.get("start_date", "").startswith(target_date)
        ]
        return daily_games
    except Exception as e:
        print(f"[NETWORK ERROR] {e}")
        return []

def run_daily_slate(target_date: str = None, summary_file: str = None):
    today_str = target_date or datetime.datetime.utcnow().strftime("%Y-%m-%d")
    games = get_today_games(today_str)
    
    print(f"\n==========================================")
    print(f"CFB SLATE RUNNER: {today_str}")
    print(f"Found {len(games)} game(s) scheduled for today.")
    print(f"==========================================\n")
    
    if not games:
        msg = f"## CFB Daily Slate: {today_str}\n*No games scheduled for this date.*"
        if summary_file:
            with open(summary_file, "a") as f:
                f.write(msg)
        print(msg)
        return

    slate_summary = [
        f"## CFB Daily Slate Projections ({today_str})",
        f"| Matchup | Proj Spread | Win Prob (Home) | Total | Kelly Rec |",
        f"| :--- | :--- | :--- | :--- | :--- |"
    ]

    for game in games:
        home = game.get("home_team")
        away = game.get("away_team")
        neutral = game.get("neutral_site", False)
        
        print(f"[SIMULATING] {away} @ {home}...")
        try:
            # Feeds the matchup into your existing engine
            results = run_monte_carlo_engine(
                home_team=home,
                away_team=away,
                n_sims=50000,
                neutral=neutral
            )
            
            spread_str = f"{home} {results['predicted_spread']:+.1f}"
            win_str = f"{results['home_win_prob'] * 100:.1f}%"
            total_str = f"{results['predicted_total']:.1f}"
            kelly_str = f"{results['quarter_kelly_pct']:.2f}%"
            
            slate_summary.append(
                f"| **{away}** @ **{home}** | {spread_str} | {win_str} | {total_str} | {kelly_str} |"
            )
        except Exception as e:
            print(f"[ERROR] Failed simulation for {away} @ {home}: {e}")
            slate_summary.append(f"| {away} @ {home} | Error | Error | - | - |")

    full_output = "\n".join(slate_summary) + "\n"
    
    if summary_file:
        with open(summary_file, "a") as f:
            f.write(full_output)
            
    print("\n[SLATE COMPLETE] All scheduled games simulated successfully.")

if __name__ == "__main__":
    date_arg = sys.argv[1] if len(sys.argv) > 1 else None
    summary_path = os.getenv("GITHUB_STEP_SUMMARY", "")
    run_daily_slate(target_date=date_arg, summary_file=summary_path)
