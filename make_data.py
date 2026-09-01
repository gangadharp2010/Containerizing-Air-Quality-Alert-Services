"""Writes data/readings.csv, the station-day readings the model was fitted on.

    python make_data.py

Standard library only, and seeded, so the file is identical on every machine.
Each row is one monitoring station on one day. The target says whether the
next day's average PM2.5 went above 35 micrograms per cubic metre.
"""
import csv
import math
import random
from pathlib import Path

ROWS = 4000
rng = random.Random(20260925)

out = Path(__file__).resolve().parent / "data" / "readings.csv"
out.parent.mkdir(parents=True, exist_ok=True)

with out.open("w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["pm25_today", "wind_speed_ms", "humidity_pct", "temp_c",
                     "traffic_index", "exceeds_tomorrow"])
    for _ in range(ROWS):
        pm25 = round(min(250.0, rng.lognormvariate(3.0, 0.55)), 1)
        wind = round(max(0.0, rng.gauss(4.0, 2.2)), 1)
        humidity = round(min(100.0, max(15.0, rng.gauss(62, 16))), 0)
        temp = round(rng.gauss(14, 9), 1)
        traffic = round(min(10.0, max(0.0, rng.gauss(5, 2.2))), 1)
        z = (-2.2 + 0.075 * (pm25 - 22) - 0.55 * (wind - 4) + 0.03 * (humidity - 62)
             - 0.05 * (temp - 14) + 0.25 * (traffic - 5) + rng.gauss(0, 1.1))
        label = 1 if rng.random() < 1 / (1 + math.exp(-z)) else 0
        writer.writerow([pm25, wind, humidity, temp, traffic, label])

print("wrote %d rows to %s" % (ROWS, out))
