import os
import sys
import datetime
import requests
from simulation_engine import run_monte_carlo_engine

CFBD_API_KEY = os.getenv("CFBD_API_KEY", "")

def get_today_games(target_date: str = None) -> list:
    if not target_date:
        target_date = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-4))).strftime("%Y-%m-%d")
    
    headers = {"Authorization": f"Bearer {CFBD_API_KEY}", "Accept": "application/json"}
    year = target_date.split("-")[0]
    
    url = f"https://api.collegefootballdata.com/games?year={year}&seasonType=regular"
    try:
        res = requests.get(url, headers=headers, timeout=15)
        if res.status_code != 200:
            print(f"[API ERROR] Failed to fetch schedule: HTTP {res.status_code}")
            return []
        
        all_games = res.json()
        daily_games = []
        
        for g in all_games:
            utc_date_str = g.get("start_date")
            if not utc_date_str:
                continue
                
            try:
                utc_dt = datetime.datetime.fromisoformat(utc_date_str.replace('Z', '+00:00'))
                eastern_dt = utc_dt.astimezone(datetime.timezone(datetime.timedelta(hours=-4)))
                game_local_date = eastern_dt.strftime("%Y-%m-%d")
                
                if game_local_date == target_date:
                    daily_games.append(g)
            except Exception:
                pass
                
        return daily_games
    except Exception as e:
        print(f"[NETWORK ERROR] {e}")
        return []

def get_game_weather(game_id: int, year: str) -> tuple:
    if not CFBD_API_KEY:
        return 5.0, 72.0
    headers = {"Authorization": f"Bearer {CFBD_API_KEY}", "Accept": "application/json"}
    url = f"https://api.collegefootballdata.com/games/weather?year={year}"
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            weather_data = res.json()
            for w in weather_data:
                if w.get("id") == game_id:
                    return float(w.get("windSpeed", 5.0) or 5.0), float(w.get("temperature", 72.0) or 72.0)
    except Exception:
        pass
    return 5.0, 72.0

def run_daily_slate(target_date: str = None, summary_file: str = ""):
    today_str = target_date or datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-4))).strftime("%Y-%m-%d")
    year = today_str.split("-")[0]
    games = get_today_games(today_str)
    
    print(f"\n==========================================")
    print(f"🏈 SOTA CFB DAILY SLATE RUNNER: {today_str}")
    print(f"Found {len(games)} game(s) scheduled for today.")
    print(f"==========================================\n")
    
    if not games:
        msg = f"## 🏈 CFB Daily Slate: {today_str}\n*No games scheduled for this date.*"
        if summary_file:
            with open(summary_file, "a") as f:
                f.write(msg)
        print(msg)
        return

    slate_summary = [
        f"## 🏈 CFB Daily Slate Projections ({today_str})",
        "| Away Team | Home Team | Proj Spread | Win Prob (Home) | Total | Kelly Rec |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |"
    ]

    for game in games:
        home = game.get("home_team")
        away = game.get("away_team")
        neutral = game.get("neutral_site", False)
        game_id = game.get("id")
        
        wind, temp = get_game_weather(game_id, year)
        print(f"[SIMULATING] {away} @ {home} (Wind: {wind}mph, Temp: {temp}F)...")
        
        try:
            results = run_monte_carlo_engine(
                home_team=home, away_team=away, n_sims=50000, neutral=neutral,
                wind_mph=wind, temp_f=temp, elevation_ft=500.0,
                home_qb_active=True, away_qb_active=True
            )
            
            spread_str = f"{home} {results['predicted_spread']:+.1f}"
            win_str = f"{results['home_win_prob'] * 100:.1f}%"
            total_str = f"{results['predicted_total']:.1f}"
            kelly_str = f"{results['quarter_kelly_pct']:.2f}%"
            
            slate_summary.append(f"| **{away}** | **{home}** | {spread_str} | {win_str} | {total_str} | {kelly_str} |")
        except Exception as e:
            print(f"[ERROR] Failed simulation for {away} @ {home}: {e}")
            slate_summary.append(f"| {away} | {home} | Error | Error | - | - |")

    full_output = "\n".join(slate_summary) + "\n"
    if summary_file:
        with open(summary_file, "a") as f:
            f.write(full_output)
    print(full_output)
    print("\n[SLATE COMPLETE] All scheduled games simulated successfully.")

if __name__ == "__main__":
    date_arg = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].strip() else None
    summary_path = os.getenv("GITHUB_STEP_SUMMARY", "")
    run_daily_slate(target_date=date_arg, summary_file=summary_path)
