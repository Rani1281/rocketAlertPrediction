import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Ensure directories exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Data sources config
ALERTS_CSV = RAW_DATA_DIR / "israel_alerts.csv"
CITIES_CSV = RAW_DATA_DIR / "cities.csv"

# Model hyperparameters (Initial guesses for the Hawkes Process)
BACKGROUND_RATE = 0.01  # Base probability of an alert
DECAY_RATE_TIME = 0.5   # How fast the temporal influence decays (beta)
DECAY_RATE_SPACE = 0.1  # How fast the spatial influence decays (kernel bandwidth)
TRIGGER_WEIGHT = 0.8    # How much a past event excites future events (alpha)

# Dashboard Config
MAP_CENTER_LAT = 31.0461
MAP_CENTER_LON = 34.8516
DEFAULT_ZOOM = 6
