import argparse
import json
import math
import os
import sys
import duckdb
import numpy as np
STATE_FILE = "data/state/team_ratings.json"
DUCKDB_FILE = "data/processed/cfb_data.duckdb"
def load_system_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"base_k_factor": 32.0, "archetypes": {}, "teams": {}}
def fetch_team_from_duckdb(team_name: str) -> dict:
    """Queries local DuckDB columnar storage for team roster and efficiency metrics."""
    default_record = {
        "rating": 1520.0, "pace": 25.5, "archetype": "standard",
        "returning_production": 0.70, "transfer_inflow_rating": 75.0,
        "qb_experience_snaps": 500, "qb_backup_dropoff": 6.0,
        "rest_days": 7, "travel_miles": 0, "lookahead_flag": False, "letdown_flag": False
    }
    if not os.path.exists(DUCKDB_FILE):
        return default_record
    try:
        con = duckdb.connect(DUCKDB_FILE, read_only=True)
        query = "SELECT rating, pace, archetype, ret_prod, portal_score, qb_snaps FROM fbs_teams WHERE LOWER(school) = LOWER(?) LIMIT 1"
        row = con.execute(query, [team_name]).fetchone()
        con.close()
        if row:
            return {
                "rating": float(row[0]),
                "pace": float(row[1]),
                "archetype": str(row[2]),
                "returning_production": float(row[3]),
                "transfer_inflow_rating": float(row[4]),
                "qb_experience_snaps": int(row[5]),
                "qb_backup_dropoff": 6.0,
                "rest_days": 7,
                "travel_miles": 0,
                "lookahead_flag": False,
                "letdown_flag": False
            }
    except Exception as e:
        print(f"[DUCKDB NOTICE] Query fallback: {e}")
    return default_record
def apply_situational_modifiers(meta: dict, is_home: bool, wind_mph: float, temp_f: float, elevation_ft: float, qb_active: bool):
    rating = meta.get("rating", 1520.0)
    pace = meta.get("pace", 25.5)
    
    ret_prod = meta.get("returning_production", 0.70)
    portal = meta.get("transfer_inflow_rating", 75.0)
    rating += (ret_prod - 0.70) * 45.0 + ((portal - 75.0) * 0.35)
    
    if not qb_active:
        rating -= (meta.get("qb_backup_dropoff", 6.0) * 25.0)
    if meta.get("lookahead_flag", False):
        rating -= 35.0
    if meta.get("letdown_flag", False):
        rating -= 45.0
    
    rating += (meta.get("rest_days", 7) - 7) * 6.0
    if not is_home and meta.get("travel_miles", 0) > 1000:
        rating -= 20.0
    if not is_home and elevation_ft > 4000:
        rating -= 25.0
        
    pass_penalty = min(0.28, (wind_mph - 15.0) * 0.018) if wind_mph > 15.0 else 0.0
    if wind_mph > 15.0:
        pace += min(3.5, (wind_mph - 15.0) * 0.15)
    return rating, pace, pass_penalty
def calibrate_probability_isotonic(raw_prob: float) -> float:
    """Non-parametric piecewise calibration aligning raw outputs with historical win frequencies."""
    bounds = [0.0, 0.20, 0.35, 0.50, 0.65, 0.80, 1.0]
    calibrated = [0.0, 0.18, 0.34, 0.50, 0.66, 0.83, 1.0]
    return float(np.interp(raw_prob, bounds, calibrated))
def run_monte_carlo_engine(home_team: str, away_team: str, n_sims: int = 50000, 
                           neutral: bool = False, wind_mph: float = 5.0, 
                           temp_f: float = 72.0, elevation_ft: float = 500.0,
                           home_qb_active: bool = True, away_qb_active: bool = True):
    state = load_system_state()
    archetypes = state.get("archetypes", {})
    home_meta = fetch_team_from_duckdb(home_team)
    away_meta = fetch_team_from_duckdb(away_team)
    home_elo, home_pace, home_wind = apply_situational_modifiers(home_meta, True, wind_mph, temp_f, elevation_ft, home_qb_active)
    away_elo, away_pace, away_wind = apply_situational_modifiers(away_meta, False, wind_mph, temp_f, elevation_ft, away_qb_active)
    elo_delta = (home_elo - away_elo) + (0.0 if neutral else 65.0)
    blended_pace = (home_pace + away_pace) / 2.0
    possessions = max(8, min(17, int((3600 / blended_pace) / 6.2)))
    
    # Dynamic Momentum Clustering factor (mu_t)
    tempo_ratio = 25.0 / max(blended_pace, 18.0)
    mu_t = 1.0 + max(-0.10, min(0.15, (tempo_ratio - 1.0) * 0.25))
    
    base_td = (0.28 + (elo_delta / 2400.0)) * mu_t
    home_td_p = max(0.08, min(0.65, base_td - (home_wind * 0.5)))
    home_fg_p = max(0.05, min(0.28, 0.17 - (home_wind * 0.4)))
    home_probs = [max(0.10, 1.0 - (home_td_p + home_fg_p)), home_fg_p, home_td_p]
    
    away_td_p = max(0.08, min(0.65, (0.28 - (elo_delta / 2400.0)) * (2.0 - mu_t) - (away_wind * 0.5)))
    away_fg_p = max(0.05, min(0.28, 0.17 - (away_wind * 0.4)))
    away_probs = [max(0.10, 1.0 - (away_td_p + away_fg_p)), away_fg_p, away_td_p]
    
    points_matrix = np.array([0, 3, 7])
    home_drives = np.random.choice(points_matrix, size=(n_sims, possessions), p=home_probs)
    away_drives = np.random.choice(points_matrix, size=(n_sims, possessions), p=away_probs)
    
    home_scores = home_drives.sum(axis=1)
    away_scores = away_drives.sum(axis=1)
    margins = home_scores - away_scores
    totals = home_scores + away_scores
    
    raw_win_prob = float(np.mean(margins > 0))
    home_win_prob = calibrate_probability_isotonic(raw_win_prob)
    away_win_prob = 1.0 - home_win_prob
    
    # Prop distributions with blowout tapering
    rb_base = 100.0 + ((home_elo - 1500.0) * 0.05)
    home_rb_rush = np.random.gamma(shape=8.5, scale=(rb_base / 8.5), size=n_sims)
    qb_mu = math.log(max(100.0, (250.0 + (home_elo - 1500.0) * 0.11) * (1.0 - home_wind)))
    home_qb_pass = np.random.lognormal(mean=qb_mu, sigma=0.22, size=n_sims)
    home_qb_tds = np.random.poisson(lam=max(0.5, (home_td_p * possessions * 0.65)), size=n_sims)
    
    blowout_idx = np.abs(margins) >= 24
    home_qb_pass[blowout_idx] *= 0.78
    home_rb_rush[blowout_idx] *= 0.85
    
    arch_key = home_meta.get("archetype", "standard")
    kelly_mult = archetypes.get(arch_key, {}).get("kelly_multiplier", 1.0)
    raw_kelly = max(0.0, ((home_win_prob - 0.5238) / (1.0 - 0.5238))) / 4.0
    final_kelly = (raw_kelly * kelly_mult) * 100.0
    
    return {
        "home_team": home_team, "away_team": away_team, "n_sims": n_sims,
        "home_win_prob": home_win_prob, "away_win_prob": away_win_prob,
        "predicted_spread": -float(np.median(margins)), "predicted_total": float(np.median(totals)),
        "home_adjusted_elo": home_elo, "away_adjusted_elo": away_elo, "archetype": arch_key,
        "momentum_multiplier": mu_t, "quarter_kelly_pct": final_kelly, "mab_multiplier": kelly_mult,
        "player_distributions": {
            "qb_pass_p50": float(np.percentile(home_qb_pass, 50)),
            "qb_pass_p75": float(np.percentile(home_qb_pass, 75)),
            "qb_pass_p90": float(np.percentile(home_qb_pass, 90)),
            "rb_rush_p50": float(np.percentile(home_rb_rush, 50)),
            "rb_rush_p75": float(np.percentile(home_rb_rush, 75)),
            "rb_rush_p90": float(np.percentile(home_rb_rush, 90)),
            "qb_td_over_1_5": float(np.mean(home_qb_tds >= 2) * 100.0)
        }
    }
def format_mobile_markdown(res: dict) -> str:
    p = res["player_distributions"]
    
    json_payload = json.dumps({
        "matchup": f"{res['away_team']} @ {res['home_team']}",
        "home_win_prob": round(res['home_win_prob'] * 100, 2),
        "away_win_prob": round(res['away_win_prob'] * 100, 2),
        "spread": round(res['predicted_spread'], 1),
        "total": round(res['predicted_total'], 1),
        "quarter_kelly_pct": round(res['quarter_kelly_pct'], 2),
        "qb_passing_medians": {
            "p50": round(p['qb_pass_p50'], 1),
            "p75": round(p['qb_pass_p75'], 1),
            "p90": round(p['qb_pass_p90'], 1)
        },
        "rb_rushing_medians": {
            "p50": round(p['rb_rush_p50'], 1),
            "p75": round(p['rb_rush_p75'], 1),
            "p90": round(p['rb_rush_p90'], 1)
        },
        "qb_over_1_5_td_prob": round(p['qb_td_over_1_5'], 1)
    }, indent=2)
    
    return f"""## Matchup Projection: {res['away_team']} @ {res['home_team']}
*Engine Run Depth: {res['n_sims']:,} Vectorized Iterations (Isotonic Calibrated)*
### Outcome Projections

| Metric | Engine Projection | Status |
| :--- | :--- | :--- |
| **{res['home_team']} Win Prob** | **{res['home_win_prob']*100:.2f}%** | Adjusted Elo: {res['home_adjusted_elo']:.1f} |
| **{res['away_team']} Win Prob** | **{res['away_win_prob']*100:.2f}%** | Adjusted Elo: {res['away_adjusted_elo']:.1f} |
| **Calculated Spread** | **{res['home_team']} {res['predicted_spread']:+.1f}** | Median Simulation Margin |
| **Projected Total** | **{res['predicted_total']:.1f} pts** | Full-Game Score Median |
| **Momentum Scale** | **{res['momentum_multiplier']:.3f}x** | Pacing Cluster Applied |
| **Quarter-Kelly Bet Size** | **{res['quarter_kelly_pct']:.2f}%** | MAB Multiplier: {res['mab_multiplier']*100:.0f}% |

---
### Player Prop Distributions (Game-Script Adjusted)

| Projection Category | Median (50th) | 75th Percentile | 90th Ceiling |
| :--- | :--- | :--- | :--- |
| **{res['home_team']} QB Passing** | {p['qb_pass_p50']:.1f} yds | {p['qb_pass_p75']:.1f} yds | {p['qb_pass_p90']:.1f} yds |
| **{res['home_team']} RB Rushing** | {p['rb_rush_p50']:.1f} yds | {p['rb_rush_p75']:.1f} yds | {p['rb_rush_p90']:.1f} yds |
| **{res['home_team']} QB Over 1.5 TDs** | **{p['qb_td_over_1_5']:.1f}% Probability** | Poisson Sampling | - |

---
### Structured JSON Output (For Betting Scripts / Slips)
```json
{json_payload}
```"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", type=str, required=True)
    parser.add_argument("--away", type=str, required=True)
    parser.add_argument("--sims", type=int, default=50000)
    parser.add_argument("--neutral", type=lambda x: str(x).lower() == 'true', default=False)
    parser.add_argument("--wind", type=float, default=5.0)
    parser.add_argument("--temp", type=float, default=72.0)
    parser.add_argument("--elevation", type=float, default=500.0)
    parser.add_argument("--home-qb", type=lambda x: str(x).lower() == 'true', default=True)
    parser.add_argument("--away-qb", type=lambda x: str(x).lower() == 'true', default=True)
    parser.add_argument("--summary-file", type=str, default="")
    args = parser.parse_args()
    results = run_monte_carlo_engine(
        args.home, args.away, args.sims, args.neutral,
        args.wind, args.temp, args.elevation, args.home_qb, args.away_qb
    )
    md = format_mobile_markdown(results)
    if args.summary_file:
        with open(args.summary_file, "a") as f:
            f.write(md)
    print(md)
