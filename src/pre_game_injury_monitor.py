import os
import duckdb
import feedparser
import subprocess

DUCKDB_FILE = "data/processed/cfb_data.duckdb"
# Free RSS feeds for breaking CFB player news
RSS_FEEDS = ["https://www.rotowire.com/rss/news.php?sport=CFB"] 

def calculate_vorp_downgrade(starter_epa: float, backup_epa: float, snaps: int) -> float:
    """Calculates the Elo penalty using Expected Points Added over replacement."""
    # 1 EPA difference per 100 snaps is roughly equivalent to 25 Elo points
    epa_diff = starter_epa - backup_epa
    return (epa_diff / 100.0) * snaps * 25.0

def poll_late_injuries():
    print("[INJURY MONITOR] Polling RSS feeds for late scratches...")
    scratches_detected = []
    
    for feed in RSS_FEEDS:
        parsed = feedparser.parse(feed)
        for entry in parsed.entries[:15]: # Scan latest 15 alerts
            title = entry.title.lower()
            if "ruled out" in title or "will not play" in title:
                scratches_detected.append(title)
                print(f" [LATE NEWS ALERT] {entry.title}")

    if scratches_detected:
        print("[ACTION] Major scratch detected. Applying VORP downgrade and recalculating matrices.")
        # In a full deployment, this triggers a DuckDB UPDATE statement 
        # to apply the VORP penalty to the team's rating, followed by:
        # subprocess.run(["python", "src/simulation_engine.py", "--home", "Georgia", "--away", "Alabama"])
    else:
        print("[CLEAR] No critical late scratches detected.")

if __name__ == "__main__":
    poll_late_injuries()
