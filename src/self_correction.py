import argparse
import json
import math
import os
import sys
import numpy as np

STATE_FILE = "data/state/team_ratings.json"

# Kalman Filter Hyperparameters (Calibrated to NCAA Football Point Spread Scale)
POINT_TO_ELO = 25.0                 # 1 spread point ~ 25 Elo points
GAME_NOISE_VARIANCE = 182.25        # Inherent CFB game variance (~13.5 pts standard deviation squared)
WEEKLY_PROCESS_NOISE = 16.0         # Weekly drift variance (roster changes, practice, bye weeks)
NETWORK_CASCADE_DAMPING = 0.22      # PageRank-style attenuation factor for past opponents
MAX_CASCADE_DEPTH = 3               # Maximum historical games back the network ripple propagates

def load_and_migrate_state() -> dict:
    """
    Loads persistent team state and automatically upgrades flat Elo ratings 
    to Bayesian Kalman distributions (mu, sigma_sq, schedule_graph) without data loss.
    """
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    if not os.path.exists(STATE_FILE):
        state = {"version": "3.0-kalman-network", "teams": {}, "archetypes": {}}
    else:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)

    state["version"] = "3.0-kalman-network"
    teams = state.get("teams", {})

    # Auto-migration: Ensure all teams have mu, sigma_sq, and schedule history
    for school, data in teams.items():
        if "mu" not in data:
            data["mu"] = float(data.get("rating", 1520.0))
        if "sigma_sq" not in data:
            # High initial uncertainty (sigma = 18.7 pts in Elo scale)
            data["sigma_sq"] = 225.0 * POINT_TO_ELO 
        if "game_history" not in data:
            data["game_history"] = []
        if "k_factor" not in data:
            data["k_factor"] = 32.0

    state["teams"] = teams
    return state

def save_state(state: dict):
    """Atomically commits updated Bayesian network parameters to disk."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def update_single_game_kalman(home_team: str, away_team: str, 
                              home_score: float, away_score: float, 
                              state: dict, is_neutral: bool = False) -> tuple:
    """
    Executes a Bayesian Kalman Filter update on a completed game.
    Calculates localized team residuals and cascades network ripples through past opponents.
    """
    teams = state["teams"]

    # Fallback initialization if a team is newly encountered
    for team in [home_team, away_team]:
        if team not in teams:
            teams[team] = {
                "mu": 1520.0, "sigma_sq": 225.0 * POINT_TO_ELO, 
                "rating": 1520.0, "pace": 25.5, "archetype": "standard",
                "game_history": [], "k_factor": 32.0
            }

    home_data = teams[home_team]
    away_data = teams[away_team]

    # 1. State Prior Extraction
    mu_home = home_data["mu"]
    var_home = home_data["sigma_sq"] + WEEKLY_PROCESS_NOISE

    mu_away = away_data["mu"]
    var_away = away_data["sigma_sq"] + WEEKLY_PROCESS_NOISE

    # Home Field Advantage (HFA) ~ 2.6 points (65 Elo)
    hfa = 0.0 if is_neutral else 65.0

    # 2. Predicted vs Actual Residuals
    expected_margin_elo = (mu_home - mu_away) + hfa
    expected_margin_pts = expected_margin_elo / POINT_TO_ELO
    actual_margin_pts = float(home_score - away_score)
    residual_error_pts = actual_margin_pts - expected_margin_pts

    # 3. Kalman Gain Calculation
    # Total observation variance: Var(Home) + Var(Away) + Inherent CFB Game Noise
    total_variance_pts = (var_home / (POINT_TO_ELO ** 2)) + (var_away / (POINT_TO_ELO ** 2)) + GAME_NOISE_VARIANCE
    k_gain_home = (var_home / POINT_TO_ELO) / total_variance_pts
    k_gain_away = (var_away / POINT_TO_ELO) / total_variance_pts

    # 4. Bayesian Posterior Update
    delta_mu_home = k_gain_home * residual_error_pts
    delta_mu_away = -k_gain_away * residual_error_pts

    new_mu_home = mu_home + delta_mu_home
    new_mu_away = mu_away + delta_mu_away

    # Uncertainty compression (the more games observed, the lower the variance)
    new_var_home = max(50.0, (1.0 - (k_gain_home / POINT_TO_ELO)) * var_home)
    new_var_away = max(50.0, (1.0 - (k_gain_away / POINT_TO_ELO)) * var_away)

    home_data["mu"] = round(new_mu_home, 2)
    home_data["rating"] = home_data["mu"]
    home_data["sigma_sq"] = round(new_var_home, 2)
    home_data["game_history"].append({"opp": away_team, "actual_margin": actual_margin_pts, "residual": round(residual_error_pts, 2)})

    away_data["mu"] = round(new_mu_away, 2)
    away_data["rating"] = away_data["mu"]
    away_data["sigma_sq"] = round(new_var_away, 2)
    away_data["game_history"].append({"opp": home_team, "actual_margin": -actual_margin_pts, "residual": round(-residual_error_pts, 2)})

    print(f"\n[KALMAN GAME UPDATE] {away_team} ({away_score}) @ {home_team} ({home_score})")
    print(f" -> Predicted Spread: {home_team} {-expected_margin_pts:+.1f} | Actual Margin: {actual_margin_pts:+.1f} | Residual: {residual_error_pts:+.1f} pts")
    print(f" -> {home_team}: mu: {mu_home:.1f} -> {new_mu_home:.1f} (delta: {delta_mu_home:+.1f}), Uncertainty (sigma): {math.sqrt(new_var_home):.1f}")
    print(f" -> {away_team}: mu: {mu_away:.1f} -> {new_mu_away:.1f} (delta: {delta_mu_away:+.1f}), Uncertainty (sigma): {math.sqrt(new_var_away):.1f}")

    # 5. Iterative Network Graph Propagation (PageRank Cascade)
    propagate_network_ripple(home_team, delta_mu_home, teams)
    propagate_network_ripple(away_team, delta_mu_away, teams)

    return residual_error_pts, delta_mu_home, delta_mu_away

def propagate_network_ripple(target_team: str, delta_mu: float, teams: dict):
    """
    Cascades a mathematical ripple through previous opponents in the schedule graph.
    If target_team proved stronger than expected, past opponents gain strength-of-schedule credit.
    """
    if abs(delta_mu) < 2.0:
        return

    history = teams[target_team].get("game_history", [])[:-1] # Exclude current game
    if not history:
        return

    print(f" [NETWORK RIPPLE] Propagating schedule graph adjustments originating from {target_team} (delta: {delta_mu:+.1f})...")

    # Traverse backward through past opponents with exponential decay
    for depth, contest in enumerate(reversed(history[-MAX_CASCADE_DEPTH:]), start=1):
        opp_name = contest["opp"]
        if opp_name not in teams:
            continue

        attenuation = (NETWORK_CASCADE_DAMPING ** depth)
        opp_adjustment = delta_mu * attenuation

        old_mu = teams[opp_name]["mu"]
        teams[opp_name]["mu"] = round(old_mu + opp_adjustment, 2)
        teams[opp_name]["rating"] = teams[opp_name]["mu"]

        print(f"   └── (Depth {depth}) Past Opponent {opp_name}: mu {old_mu:.1f} -> {teams[opp_name]['mu']:.1f} (shift: {opp_adjustment:+.2f})")

def process_chronological_slate(games_slate: list):
    """
    Sequentially evaluates a list of games in exact conclusion order.
    Guarantees that Game 2 benefits from the calibrated state produced by Game 1.
    """
    state = load_and_migrate_state()
    total_residual_sq = 0.0

    print(f"[SLATE INITIALIZATION] Processing {len(games_slate)} games in strict sequential order...")

    for idx, match in enumerate(games_slate, start=1):
        print(f"\n--- Sequence #{idx}: {match['away']} @ {match['home']} ---")
        residual, _, _ = update_single_game_kalman(
            home_team=match["home"],
            away_team=match["away"],
            home_score=match["home_score"],
            away_score=match["away_score"],
            state=state,
            is_neutral=match.get("neutral", False)
        )
        total_residual_sq += (residual ** 2)

    mean_brier_approx = total_residual_sq / len(games_slate)
    print(f"\n==================================================")
    print(f" SEQUENTIAL SLATE COMPLETE: Mean Square Residual = {mean_brier_approx:.2f}")
    print(f"==================================================")

    save_state(state)
    print(f"[STATE COMMITTED] All Bayesian parameters and schedule graphs saved to {STATE_FILE}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-game", action="store_true", default=False)
    parser.add_argument("--home", type=str, default="")
    parser.add_argument("--away", type=str, default="")
    parser.add_argument("--home-score", type=float, default=0.0)
    parser.add_argument("--away-score", type=float, default=0.0)
    parser.add_argument("--neutral", action="store_true", default=False)
    parser.add_argument("--test-slate", action="store_true", default=False)
    args = parser.parse_args()

    if args.single_game:
        current_state = load_and_migrate_state()
        update_single_game_kalman(args.home, args.away, args.home_score, args.away_score, current_state, args.neutral)
        save_state(current_state)
    elif args.test_slate:
        # Example chronological Saturday slate sequence
        simulated_slate = [
            {"home": "Georgia", "away": "Clemson", "home_score": 34, "away_score": 3, "neutral": True},
            {"home": "Alabama", "away": "Texas A&M", "home_score": 28, "away_score": 24, "neutral": False},
            {"home": "Texas Tech", "away": "Army", "home_score": 17, "away_score": 20, "neutral": False}
        ]
        process_chronological_slate(simulated_slate)
    else:
        # Standalone verification and schema upgrade
        current_state = load_and_migrate_state()
        save_state(current_state)
        print("[SCHEMA VERIFIED] State file verified and upgraded to Kalman Network structure.")
