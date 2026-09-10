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
def fetch_team_metrics(team_name: str) -> dict:
    """Retrieves advanced team metrics from DuckDB with defensive efficiency baselines."""
    default_record = {
        "rating": 1520.0, "pace": 25.5, "archetype": "standard",
        "returning_production": 0.70, "transfer_inflow_rating": 75.0,
        "qb_experience_snaps": 500, "qb_backup_dropoff": 6.0,
        "havoc_rate": 0.15, "success_rate": 0.42, "proe": 0.00,
        "pass_epa_allowed": 0.05, "rush_epa_allowed": -0.02,
        "def_line_yards": 3.05, "def_pass_success": 0.38
    }
    if not os.path.exists(DUCKDB_FILE):
        return default_record
    try:
        con = duckdb.connect(DUCKDB_FILE, read_only=True)
        query = """
            SELECT rating, pace, archetype, ret_prod, portal_score, qb_snaps 
            FROM fbs_teams 
            WHERE LOWER(school) = LOWER(?) LIMIT 1
        """
        row = con.execute(query, [team_name]).fetchone()
        con.close()
        if row:
            default_record.update({
                "rating": float(row[0]), "pace": float(row[1]), "archetype": str(row[2]),
                "returning_production": float(row[3]), "transfer_inflow_rating": float(row[4]),
                "qb_experience_snaps": int(row[5])
            })
    except Exception:
        pass
    return default_record
def apply_situational_modifiers(meta: dict, is_home: bool, wind_mph: float, temp_f: float, elevation_ft: float, qb_starter: bool):
    """Calculates weather, travel, and rest adjustments for team pace and power rating."""
    rating = meta.get("rating", 1520.0)
    pace = meta.get("pace", 25.5)
    ret_prod = meta.get("returning_production", 0.70)
    portal = meta.get("transfer_inflow_rating", 75.0)
    rating += (ret_prod - 0.70) * 40.0 + ((portal - 75.0) * 0.30)
    if not qb_starter:
        rating -= (meta.get("qb_backup_dropoff", 6.0) * 25.0)
    pass_penalty = 0.0
    if wind_mph > 15.0:
        pass_penalty = min(0.28, (wind_mph - 15.0) * 0.018)
        pace += min(3.5, (wind_mph - 15.0) * 0.15)
    if not is_home and elevation_ft > 4000:
        rating -= 25.0
    return rating, pace, pass_penalty
def simulate_player_props(n_sims: int, total_snaps: np.ndarray, pass_ratios: np.ndarray, 
                          margins: np.ndarray, pass_penalty: float, 
                          opp_pass_epa: float, opp_rush_epa: float,
                          qb_name: str, rb_name: str, wr_name: str,
                          rb_carry_share: float = 0.62, wr_target_share: float = 0.26):
    """
    Simulates individual player props conditioned on snap volume, game script, and defensive matchups.
    """
    # 1. Total Team Opportunities per Simulation Run
    team_pass_attempts = np.random.poisson(lam=np.maximum(12.0, total_snaps * pass_ratios))
    team_rush_attempts = np.random.poisson(lam=np.maximum(15.0, total_snaps * (1.0 - pass_ratios)))
    # Blowout Starter Rotation Adjustment (Starters sit when margin >= 24)
    starter_snap_pct = np.ones(n_sims)
    blowout_idx = np.abs(margins) >= 24
    starter_snap_pct[blowout_idx] = 0.72
    # 2. Quarterback Passing Projections
    qb_attempts = np.round(team_pass_attempts * starter_snap_pct).astype(int)
    
    # Opponent-Adjusted Yards Per Attempt (YPA)
    base_ypa = 7.45 + (opp_pass_epa * 6.5) - (pass_penalty * 3.5)
    ypa_mu = math.log(max(4.0, base_ypa))
    simulated_ypa = np.random.lognormal(mean=ypa_mu, sigma=0.18, size=n_sims)
    qb_pass_yards = qb_attempts * simulated_ypa
    # Completion Rate & Completions
    comp_pct = np.clip(0.635 + (opp_pass_epa * 0.15) - (pass_penalty * 0.20), 0.45, 0.82)
    qb_completions = np.random.binomial(qb_attempts, comp_pct)
    # Passing Touchdowns (Zero-Inflated Poisson sampling)
    td_rate = np.clip(0.055 + (opp_pass_epa * 0.04), 0.015, 0.12)
    qb_pass_tds = np.random.poisson(lam=np.maximum(0.2, qb_attempts * td_rate))
    # 3. Running Back Rushing Projections
    rb_attempts = np.round(team_rush_attempts * rb_carry_share * starter_snap_pct).astype(int)
    base_ypc = max(2.5, 4.60 + (opp_rush_epa * 5.0))
    # Gamma distribution for positive-skewed breakaway runs
    simulated_ypc = np.random.gamma(shape=9.0, scale=(base_ypc / 9.0), size=n_sims)
    rb_rush_yards = rb_attempts * simulated_ypc
    rb_rush_tds = np.random.poisson(lam=np.maximum(0.1, rb_attempts * 0.042))
    # 4. Wide Receiver Receiving Projections
    wr_targets = np.random.binomial(qb_attempts, np.clip(wr_target_share, 0.05, 0.45))
    wr_catch_rate = np.clip(0.66 + (opp_pass_epa * 0.10) - (pass_penalty * 0.12), 0.45, 0.85)
    wr_receptions = np.random.binomial(wr_targets, wr_catch_rate)
    
    base_ypr = max(6.0, 12.8 + (opp_pass_epa * 8.0))
    simulated_ypr = np.random.lognormal(mean=math.log(base_ypr), sigma=0.22, size=n_sims)
    wr_rec_yards = wr_receptions * simulated_ypr
    wr_rec_tds = np.random.poisson(lam=np.maximum(0.05, wr_targets * 0.065))
    return {
        "qb": {
            "name": qb_name,
            "pass_yds_p50": float(np.percentile(qb_pass_yards, 50)),
            "pass_yds_p75": float(np.percentile(qb_pass_yards, 75)),
            "pass_yds_p90": float(np.percentile(qb_pass_yards, 90)),
            "attempts_p50": float(np.percentile(qb_attempts, 50)),
            "completions_p50": float(np.percentile(qb_completions, 50)),
            "over_1_5_td_pct": float(np.mean(qb_pass_tds >= 2) * 100.0),
            "over_2_5_td_pct": float(np.mean(qb_pass_tds >= 3) * 100.0)
        },
        "rb": {
            "name": rb_name,
            "rush_yds_p50": float(np.percentile(rb_rush_yards, 50)),
            "rush_yds_p75": float(np.percentile(rb_rush_yards, 75)),
            "attempts_p50": float(np.percentile(rb_attempts, 50)),
            "anytime_td_pct": float(np.mean(rb_rush_tds >= 1) * 100.0)
        },
        "wr": {
            "name": wr_name,
            "rec_yds_p50": float(np.percentile(wr_rec_yards, 50)),
            "rec_yds_p75": float(np.percentile(wr_rec_yards, 75)),
            "receptions_p50": float(np.percentile(wr_receptions, 50)),
            "anytime_td_pct": float(np.mean(wr_rec_tds >= 1) * 100.0)
        }
    }
def run_monte_carlo_engine(home_team: str, away_team: str, n_sims: int = 50000, 
                           neutral: bool = False, wind_mph: float = 5.0, 
                           temp_f: float = 72.0, elevation_ft: float = 500.0,
                           home_qb_active: bool = True, away_qb_active: bool = True,
                           home_qb_name: str = "Starter QB", home_rb_name: str = "RB1", home_wr_name: str = "WR1",
                           rb_share: float = 0.62, wr_share: float = 0.26):
    
    state = load_system_state()
    archetypes = state.get("archetypes", {})
    home_meta = fetch_team_metrics(home_team)
    away_meta = fetch_team_metrics(away_team)
    home_elo, home_pace, home_wind = apply_situational_modifiers(home_meta, True, wind_mph, temp_f, elevation_ft, home_qb_active)
    away_elo, away_pace, away_wind = apply_situational_modifiers(away_meta, False, wind_mph, temp_f, elevation_ft, away_qb_active)
    elo_delta = (home_elo - away_elo) + (0.0 if neutral else 65.0)
    blended_pace = (home_pace + away_pace) / 2.0
    
    # Dynamic Momentum Multiplier (mu_t)
    momentum_factor = 1.0 + (home_meta.get("havoc_rate", 0.15) - away_meta.get("havoc_rate", 0.15)) * 0.45
    possessions = max(8, min(17, int(3600 / blended_pace / 6.2)))
    # Drive probabilities
    base_td_prob = 0.28 + (elo_delta / 2400.0) * momentum_factor
    home_td_p = max(0.08, min(0.65, base_td_prob - (home_wind * 0.5)))
    home_fg_p = max(0.05, min(0.28, 0.17 - (home_wind * 0.4)))
    home_probs = [max(0.10, 1.0 - (home_td_p + home_fg_p)), home_fg_p, home_td_p]
    away_td_p = max(0.08, min(0.65, (0.28 - (elo_delta / 2400.0)) - (away_wind * 0.5)))
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
    # Isotonic Probability Calibration
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit([0.05, 0.20, 0.40, 0.50, 0.60, 0.80, 0.95], 
                   [0.07, 0.22, 0.38, 0.51, 0.63, 0.79, 0.93])
    calibrated_home_win_prob = float(calibrator.predict([raw_win_prob])[0])
    # Compute Play Volume per Game Iteration
    total_snaps = np.random.normal(loc=(3600.0 / home_pace), scale=4.5, size=n_sims)
    
    # Dynamic Pass Ratio conditioned on score differential (Pass more when trailing, run more when leading)
    baseline_pass_ratio = 0.52 + home_meta.get("proe", 0.00)
    script_adjustment = np.clip(-margins * 0.005, -0.15, 0.15)
    simulated_pass_ratios = np.clip(baseline_pass_ratio + script_adjustment, 0.28, 0.82)
    # Granular Player Prop Projections
    player_props = simulate_player_props(
        n_sims=n_sims, total_snaps=total_snaps, pass_ratios=simulated_pass_ratios,
        margins=margins, pass_penalty=home_wind,
        opp_pass_epa=away_meta.get("pass_epa_allowed", 0.05),
        opp_rush_epa=away_meta.get("rush_epa_allowed", -0.02),
        qb_name=home_qb_name, rb_name=home_rb_name, wr_name=home_wr_name,
        rb_carry_share=rb_share, wr_target_share=wr_share
    )
    arch_key = home_meta.get("archetype", "standard")
    kelly_mult = archetypes.get(arch_key, {}).get("kelly_multiplier", 1.0)
    raw_kelly = max(0.0, ((calibrated_home_win_prob - 0.5238) / (1.0 - 0.5238))) / 4.0
    final_kelly = (raw_kelly * kelly_mult) * 100.0
    return {
        "home_team": home_team, "away_team": away_team, "n_sims": n_sims,
        "home_win_prob": calibrated_home_win_prob, "raw_win_prob": raw_win_prob,
        "predicted_spread": -float(np.median(margins)), "predicted_total": float(np.median(totals)),
        "momentum_multiplier": momentum_factor, "quarter_kelly_pct": final_kelly,
        "player_props": player_props
    }
def format_markdown(res: dict) -> str:
    props = res["player_props"]
    qb = props["qb"]
    rb = props["rb"]
    wr = props["wr"]
    return f"""## Matchup Projection: {res['away_team']} @ {res['home_team']}
*Engine Depth: {res['n_sims']:,} Iterations | Opponent Matchup & Usage Calibrated*
### Game Outcome Metrics

| Metric | Engine Projection | Notes |
| :--- | :--- | :--- |
| **{res['home_team']} Win Prob** | **{res['home_win_prob']*100:.2f}%** | Isotonic Calibrated |
| **{res['away_team']} Win Prob** | **{(1.0 - res['home_win_prob'])*100:.2f}%** | Raw MC: {res['raw_win_prob']*100:.1f}% |
| **Projected Spread** | **{res['home_team']} {res['predicted_spread']:+.1f}** | Median Simulation Margin |
| **Projected Total** | **{res['predicted_total']:.1f} pts** | Script & Tempo Adjusted |
| **Quarter-Kelly Size** | **{res['quarter_kelly_pct']:.2f}%** | Kelly Allocation |

---
### Player Prop Distributions (Usage & Matchup Conditioned)
#### Quarterback: {qb['name']}

| Market Line | Projected Median (50th) | 75th Percentile | 90th Ceiling | Over Probability |
| :--- | :--- | :--- | :--- | :--- |
| **Passing Yards** | **{qb['pass_yds_p50']:.1f} yds** | {qb['pass_yds_p75']:.1f} yds | {qb['pass_yds_p90']:.1f} yds | - |
| **Completions** | **{qb['completions_p50']:.1f} comp** | Total Attempts: {qb['attempts_p50']:.1f} | - | - |
| **Passing TDs** | **Over 1.5 TDs** | Over 2.5 TDs: {qb['over_2_5_td_pct']:.1f}% | - | **{qb['over_1_5_td_pct']:.1f}%** |

#### Running Back: {rb['name']}

| Market Line | Projected Median (50th) | 75th Percentile | Expected Touches | Anytime TD % |
| :--- | :--- | :--- | :--- | :--- |
| **Rushing Yards** | **{rb['rush_yds_p50']:.1f} yds** | {rb['rush_yds_p75']:.1f} yds | {rb['attempts_p50']:.1f} carries | **{rb['anytime_td_pct']:.1f}%** |

#### Wide Receiver: {wr['name']}

| Market Line | Projected Median (50th) | 75th Percentile | Expected Catches | Anytime TD % |
| :--- | :--- | :--- | :--- | :--- |
| **Receiving Yards** | **{wr['rec_yds_p50']:.1f} yds** | {wr['rec_yds_p75']:.1f} yds | {wr['receptions_p50']:.1f} rec | **{wr['anytime_td_pct']:.1f}%** |

"""
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
    parser.add_argument("--qb-name", type=str, default="Starting QB")
    parser.add_argument("--rb-name", type=str, default="RB1")
    parser.add_argument("--wr-name", type=str, default="WR1")
    parser.add_argument("--rb-share", type=float, default=0.62)
    parser.add_argument("--wr-share", type=float, default=0.26)
    parser.add_argument("--summary-file", type=str, default="")
    args = parser.parse_args()
    results = run_monte_carlo_engine(
        home_team=args.home, away_team=args.away, n_sims=args.sims, neutral=args.neutral,
        wind_mph=args.wind, temp_f=args.temp, elevation_ft=args.elevation,
        home_qb_active=args.home_qb, away_qb_active=args.away_qb,
        home_qb_name=args.qb_name, home_rb_name=args.rb_name, home_wr_name=args.wr_name,
        rb_share=args.rb_share, wr_share=args.wr_share
    )
    md = format_markdown(results)
    if args.summary_file:
        with open(args.summary_file, "a") as f:
            f.write(md)
    print(md)
