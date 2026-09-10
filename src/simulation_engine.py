import argparse
import json
import math
import os
import duckdb
import numpy as np
from sklearn.isotonic import IsotonicRegression
STATE_FILE = "data/state/team_ratings.json"
DUCKDB_FILE = "data/processed/cfb_data.duckdb"
def load_system_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"base_k_factor": 32.0, "archetypes": {}, "teams": {}}
def run_monte_carlo_engine(home_team: str, away_team: str, n_sims: int = 50000, 
                           home_qb_name: str = "QB1", home_rb_name: str = "RB1", 
                           rb_share: float = 0.55, wr_share: float = 0.25):
    
    state = load_system_state()
    
    # Baseline ratings (Fallback for demonstration)
    home_elo, home_pace = 1850.0, 26.2
    away_elo, away_pace = 1500.0, 28.5
    
    elo_delta = (home_elo - away_elo) + 65.0
    blended_pace = (home_pace + away_pace) / 2.0
    possessions = max(8, min(17, int((3600 / blended_pace) / 6.2)))
    # Dynamic Momentum Clustering (mu_t)
    tempo_ratio = 25.0 / max(blended_pace, 18.0)
    mu_t = 1.0 + max(-0.10, min(0.15, (tempo_ratio - 1.0) * 0.25))
    # Drive Probabilities
    base_td = (0.28 + (elo_delta / 2400.0)) * mu_t
    home_td_p = max(0.08, min(0.65, base_td))
    home_probs = [max(0.10, 1.0 - (home_td_p + 0.17)), 0.17, home_td_p]
    away_td_p = max(0.08, min(0.65, (0.28 - (elo_delta / 2400.0)) * (2.0 - mu_t)))
    away_probs = [max(0.10, 1.0 - (away_td_p + 0.17)), 0.17, away_td_p]
    points_matrix = np.array([0, 3, 7])
    home_drives = np.random.choice(points_matrix, size=(n_sims, possessions), p=home_probs)
    away_drives = np.random.choice(points_matrix, size=(n_sims, possessions), p=away_probs)
    home_scores = home_drives.sum(axis=1)
    away_scores = away_drives.sum(axis=1)
    margins = home_scores - away_scores
    totals = home_scores + away_scores
    # Isotonic Probability Calibration
    raw_home_win_prob = float(np.mean(margins > 0))
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit([0.0, 0.20, 0.50, 0.80, 1.0], [0.0, 0.18, 0.50, 0.83, 1.0])
    calibrated_home_win_prob = float(calibrator.predict([raw_home_win_prob])[0])
    # Player Prop Seeding (Gamma, Log-Normal, Poisson)
    blowout_idx = np.abs(margins) >= 24
    
    rb_base = (100.0 + ((home_elo - 1500.0) * 0.05)) * (rb_share / 0.50)
    home_rb_rush = np.random.gamma(shape=8.5, scale=(rb_base / 8.5), size=n_sims)
    home_rb_rush[blowout_idx] *= 0.85 
    
    qb_mu = math.log(max(100.0, (250.0 + (home_elo - 1500.0) * 0.11)))
    home_qb_pass = np.random.lognormal(mean=qb_mu, sigma=0.22, size=n_sims)
    home_qb_pass[blowout_idx] *= 0.78 
    
    home_qb_tds = np.random.poisson(lam=max(0.5, (home_td_p * possessions * 0.65)), size=n_sims)
    return {
        "home_team": home_team, "away_team": away_team, "n_sims": n_sims,
        "home_win_prob": calibrated_home_win_prob,
        "predicted_spread": -float(np.median(margins)),
        "player_distributions": {
            "qb_name": home_qb_name,
            "rb_name": home_rb_name,
            "qb_pass_p50": float(np.percentile(home_qb_pass, 50)),
            "qb_pass_p75": float(np.percentile(home_qb_pass, 75)),
            "rb_rush_p50": float(np.percentile(home_rb_rush, 50)),
            "qb_td_over_1_5": float(np.mean(home_qb_tds >= 2) * 100.0)
        }
    }
def format_mobile_markdown(res: dict) -> str:
    p = res["player_distributions"]
    return f"""## Matchup Projection: {res['away_team']} @ {res['home_team']}
*Engine Depth: {res['n_sims']:,} Vectorized Iterations (Isotonic Calibrated)*
### Player Prop Distributions (Game-Script Adjusted)

| Projection Category | Median (50th) | 75th Percentile |
| :--- | :--- | :--- |
| **{p['qb_name']} (QB) Passing** | {p['qb_pass_p50']:.1f} yds | {p['qb_pass_p75']:.1f} yds |
| **{p['rb_name']} (RB) Rushing** | {p['rb_rush_p50']:.1f} yds | Script Adjusted |
| **{p['qb_name']} Over 1.5 TDs** | **{p['qb_td_over_1_5']:.1f}% Probability** | Poisson Sampling |

"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", type=str, required=True)
    parser.add_argument("--away", type=str, required=True)
    parser.add_argument("--sims", type=int, default=50000)
    parser.add_argument("--home-qb", type=str, default="Starting QB")
    parser.add_argument("--home-rb", type=str, default="Starting RB")
    parser.add_argument("--rb-share", type=float, default=0.55)
    parser.add_argument("--summary-file", type=str, default="")
    args = parser.parse_args()
    results = run_monte_carlo_engine(args.home, args.away, args.sims, args.home_qb, args.home_rb, args.rb_share)
    md = format_mobile_markdown(results)
    
    if args.summary_file:
        with open(args.summary_file, "a") as f:
            f.write(md)
    print(md)
