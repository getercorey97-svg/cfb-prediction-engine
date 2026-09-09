import argparse
import json
import math
import os
import sys
import numpy as np
STATE_FILE = "data/state/team_ratings.json"
def load_system_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"base_k_factor": 32.0, "archetypes": {}, "teams": {}}
def apply_situational_and_roster_modifiers(team_name: str, meta: dict, is_home: bool, 
                                           wind_mph: float, temp_f: float, elevation_ft: float,
                                           qb_starter_active: bool) -> tuple:
    """
    Computes modified dynamic Elo rating and pace based on environmental drag,
    scheduling spots (letdown/lookahead/rest), and roster/quarterback continuity.
    """
    rating = meta.get("rating", 1500.0)
    pace = meta.get("pace", 25.5)
    # 1. Roster Continuity and QB Status
    ret_prod = meta.get("returning_production", 0.70)
    transfer_inflow = meta.get("transfer_inflow_rating", 75.0)
    stability_factor = (ret_prod - 0.70) * 40.0 + ((transfer_inflow - 75.0) * 0.3)
    rating += stability_factor
    if not qb_starter_active:
        dropoff_pts = meta.get("qb_backup_dropoff", 6.0)
        rating -= (dropoff_pts * 25.0)
    # 2. Situational Scheduling & Circumstance Index
    if meta.get("lookahead_flag", False):
        rating -= 35.0
    if meta.get("letdown_flag", False):
        rating -= 45.0
    rest_diff = meta.get("rest_days", 7) - 7
    rating += (rest_diff * 6.0)
    travel = meta.get("travel_miles", 0)
    if travel > 1000 and not is_home:
        rating -= 20.0
    # 3. Environmental & Weather Modifiers
    if elevation_ft > 4000 and not is_home:
        rating -= 25.0
    pass_efficiency_penalty = 0.0
    if wind_mph > 15.0:
        wind_excess = wind_mph - 15.0
        pass_efficiency_penalty = min(0.25, wind_excess * 0.015)
        pace += min(3.0, wind_excess * 0.15)
    heat_fatigue_mult = 1.0
    if temp_f > 85.0:
        heat_fatigue_mult = 1.0 + min(0.20, (temp_f - 85.0) * 0.01)
    return rating, pace, pass_efficiency_penalty, heat_fatigue_mult
def run_monte_carlo_engine(home_team: str, away_team: str, n_sims: int = 50000, 
                           neutral: bool = False, wind_mph: float = 5.0, 
                           temp_f: float = 72.0, elevation_ft: float = 500.0,
                           home_qb_active: bool = True, away_qb_active: bool = True):
    """
    Executes a 50,000-iteration vectorized Monte Carlo simulation with player prop distributions
    and dynamic game-script usage adjustments.
    """
    state = load_system_state()
    teams = state.get("teams", {})
    archetypes = state.get("archetypes", {})
    default_meta = {
        "rating": 1500.0, "pace": 25.5, "archetype": "standard",
        "returning_production": 0.70, "transfer_inflow_rating": 75.0,
        "qb_experience_snaps": 500, "qb_backup_dropoff": 6.0,
        "rest_days": 7, "travel_miles": 0, "lookahead_flag": False, "letdown_flag": False
    }
    home_meta = teams.get(home_team, default_meta)
    away_meta = teams.get(away_team, default_meta)
    home_elo, home_pace, home_wind_pen, home_heat = apply_situational_and_roster_modifiers(
        home_team, home_meta, True, wind_mph, temp_f, elevation_ft, home_qb_active
    )
    away_elo, away_pace, away_wind_pen, away_heat = apply_situational_and_roster_modifiers(
        away_team, away_meta, False, wind_mph, temp_f, elevation_ft, away_qb_active
    )
    elo_delta = home_elo - away_elo
    if not neutral:
        elo_delta += 65.0
    blended_pace = (home_pace + away_pace) / 2.0
    total_plays = int(3600 / blended_pace)
    possessions = max(8, min(17, int(total_plays / 6.2)))
    base_td_prob = 0.28 + (elo_delta / 2400.0)
    home_td_p = max(0.08, min(0.65, base_td_prob - (home_wind_pen * 0.5)))
    home_fg_p = max(0.05, min(0.28, 0.17 - (home_wind_pen * 0.4)))
    home_stop_p = max(0.10, 1.0 - (home_td_p + home_fg_p))
    home_probs = [home_stop_p, home_fg_p, home_td_p]
    away_td_p = max(0.08, min(0.65, (0.28 - (elo_delta / 2400.0)) - (away_wind_pen * 0.5)))
    away_fg_p = max(0.05, min(0.28, 0.17 - (away_wind_pen * 0.4)))
    away_stop_p = max(0.10, 1.0 - (away_td_p + away_fg_p))
    away_probs = [away_stop_p, away_fg_p, away_td_p]
    points_matrix = np.array([0, 3, 7])
    home_drives = np.random.choice(points_matrix, size=(n_sims, possessions), p=home_probs)
    away_drives = np.random.choice(points_matrix, size=(n_sims, possessions), p=away_probs)
    home_scores = home_drives.sum(axis=1)
    away_scores = away_drives.sum(axis=1)
    margins = home_scores - away_scores
    totals = home_scores + away_scores
    home_win_prob = float(np.mean(margins > 0))
    away_win_prob = float(np.mean(margins < 0))
    median_spread = float(np.median(margins))
    median_total = float(np.median(totals))
    rb_base = 100.0 + ((home_elo - 1500.0) * 0.05)
    home_rb_rush = np.random.gamma(shape=8.5, scale=(rb_base / 8.5), size=n_sims)
    qb_mu = math.log(max(100.0, (250.0 + (home_elo - 1500.0) * 0.11) * (1.0 - home_wind_pen)))
    home_qb_pass = np.random.lognormal(mean=qb_mu, sigma=0.22, size=n_sims)
    td_lambda = max(0.5, (home_td_p * possessions * 0.65))
    home_qb_tds = np.random.poisson(lam=td_lambda, size=n_sims)
    blowout_masks = np.abs(margins) >= 24
    home_qb_pass[blowout_masks] = home_qb_pass[blowout_masks] * 0.78
    home_rb_rush[blowout_masks] = home_rb_rush[blowout_masks] * 0.85
    market_line_prob = 0.5238
    arch_key = home_meta.get("archetype", "standard")
    kelly_mult = archetypes.get(arch_key, {}).get("kelly_multiplier", 1.0)
    raw_kelly = max(0.0, ((home_win_prob - market_line_prob) / (1.0 - market_line_prob))) / 4.0
    final_kelly = (raw_kelly * kelly_mult) * 100.0
    return {
        "home_team": home_team,
        "away_team": away_team,
        "n_sims": n_sims,
        "home_win_prob": home_win_prob,
        "away_win_prob": away_win_prob,
        "predicted_spread": -median_spread,
        "predicted_total": median_total,
        "home_adjusted_elo": home_elo,
        "away_adjusted_elo": away_elo,
        "environmental_drag": {
            "wind_mph": wind_mph,
            "pass_penalty_pct": home_wind_pen * 100.0,
            "elevation_ft": elevation_ft
        },
        "quarter_kelly_pct": final_kelly,
        "mab_multiplier": kelly_mult,
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
    e = res["environmental_drag"]
    return f"""## Matchup Projection: {res['away_team']} @ {res['home_team']}
*Engine Run Depth: {res['n_sims']:,} Vectorized Iterations*
### Game Outcome Metrics

| Metric | Engine Projection | Notes |
| :--- | :--- | :--- |
| **{res['home_team']} Win Prob** | **{res['home_win_prob'] * 100:.2f}%** | Elo: {res['home_adjusted_elo']:.1f} |
| **{res['away_team']} Win Prob** | **{res['away_win_prob'] * 100:.2f}%** | Elo: {res['away_adjusted_elo']:.1f} |
| **Calculated Spread** | **{res['home_team']} {res['predicted_spread']:+.1f}** | Median Difference |
| **Calculated Total** | **{res['predicted_total']:.1f} pts** | Full-Game Output |
| **Quarter-Kelly Size** | **{res['quarter_kelly_pct']:.2f}%** | MAB Throttle: {res['mab_multiplier']*100:.0f}% |

---
### Situational & Environmental Adjustments

| Condition | Parameter Value | Direct Impact |
| :--- | :--- | :--- |
| **Sustained Wind** | {e['wind_mph']:.1f} mph | Pass Efficiency Penalty: -{e['pass_penalty_pct']:.1f}% |
| **Stadium Elevation** | {e['elevation_ft']:.0f} ft | Fatigue Index Applied |

---
### Player Usage Distributions (Game-Script Adjusted)

| Market Projection | Median (50th) | 75th Percentile | 90th Ceiling |
| :--- | :--- | :--- | :--- |
| **{res['home_team']} QB Pass Yds** | {p['qb_pass_p50']:.1f} yds | {p['qb_pass_p75']:.1f} yds | {p['qb_pass_p90']:.1f} yds |
| **{res['home_team']} RB Rush Yds** | {p['rb_rush_p50']:.1f} yds | {p['rb_rush_p75']:.1f} yds | {p['rb_rush_p90']:.1f} yds |
| **{res['home_team']} QB Over 1.5 TDs** | **{p['qb_td_over_1_5']:.1f}% Probability** | Poisson Sampling | - |

"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", type=str, required=True)
    parser.add_argument("--away", type=str, required=True)
    parser.add_argument("--sims", type=int, default=50000)
    parser.add_argument("--neutral", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--wind", type=float, default=5.0)
    parser.add_argument("--temp", type=float, default=72.0)
    parser.add_argument("--elevation", type=float, default=500.0)
    parser.add_argument("--home-qb", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--away-qb", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--summary-file", type=str, default="")
    args = parser.parse_args()
    results = run_monte_carlo_engine(
        home_team=args.home, away_team=args.away, n_sims=args.sims, neutral=args.neutral,
        wind_mph=args.wind, temp_f=args.temp, elevation_ft=args.elevation,
        home_qb_active=args.home_qb, away_qb_active=args.away_qb
    )
    md_output = format_mobile_markdown(results)
    if args.summary_file:
        with open(args.summary_file, "a") as sf:
            sf.write(md_output)
    print(md_output)
