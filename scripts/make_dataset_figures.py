"""Create the dataset overview figures used in data/README.md and the main README.

Run from the project root:  python scripts/make_dataset_figures.py
Reads the two ORIGINAL files in data/ and writes PNGs to docs/images/.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Transformer

DATA, OUT = Path("data"), Path("docs/images")
OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, TEAL, INK = "#0284C7", "#F97316", "#0D9488", "#0F172A"
plt.rcParams.update({"font.size": 12, "axes.titleweight": "bold", "axes.grid": True, "grid.alpha": .3, "savefig.dpi": 150})

# ---------- Wholesale customers ----------
w = pd.read_csv(DATA / "Wholesale_customers_data.csv")
CAT = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]

fig, ax = plt.subplots(figsize=(9, 4.5))
w.groupby("Channel")[CAT].median().rename(index={1: "Horeca (1)", 2: "Retail (2)"}).T.plot.bar(ax=ax, color=[BLUE, ORANGE])
ax.set_ylabel("median annual spending (m.u.)"); ax.set_title("Wholesale: median spending per category and channel")
plt.xticks(rotation=20); fig.tight_layout(); fig.savefig(OUT / "dataset_wholesale_channel_medians.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 5.5))
for ch, col, nm in [(1, BLUE, "Horeca"), (2, ORANGE, "Retail")]:
    s = w[w.Channel == ch]; ax.scatter(s.Fresh, s.Grocery, s=14, color=col, label=nm, alpha=.7)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Fresh (m.u., log axis)"); ax.set_ylabel("Grocery (m.u., log axis)")
ax.set_title("Fresh vs Grocery for 440 customers"); ax.legend(); fig.tight_layout(); fig.savefig(OUT / "dataset_wholesale_fresh_grocery.png"); plt.close(fig)

# ---------- Ridgecrest catalogue ----------
ev = pd.read_csv(DATA / "ridgecrest_shelly2019_catalog_raw.txt", comment="#", sep=r"\s+", header=None,
                 names=["year", "month", "day", "hour", "minute", "second", "lat", "lon", "depth_km", "mag", "event_id"])
ev["t"] = pd.to_datetime(dict(year=ev.year, month=ev.month, day=ev.day, hour=ev.hour, minute=ev.minute)) + pd.to_timedelta(ev.second, unit="s")
ms = ev.loc[ev.mag.idxmax()]

fig, ax = plt.subplots(figsize=(6.5, 6.5))
sc = ax.scatter(ev.lon, ev.lat, s=1.5, c=ev.depth_km.clip(upper=20), cmap="viridis_r"); plt.colorbar(sc, label="depth (km, clipped at 20)")
ax.scatter(ms.lon, ms.lat, marker="*", s=250, color=ORANGE, edgecolor=INK, zorder=5, label="M7.1 mainshock")
ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.set_title(f"Ridgecrest catalogue map ({len(ev):,} events)"); ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "dataset_ridgecrest_map.png"); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
axs[0].hist(ev.mag, bins=50, color=BLUE); axs[0].set_yscale("log"); axs[0].set_xlabel("magnitude"); axs[0].set_ylabel("events (log)"); axs[0].set_title("Magnitude distribution")
day = ev.t.dt.floor("D").value_counts().sort_index(); axs[1].bar(day.index.strftime("%m-%d"), day.values, color=TEAL)
axs[1].set_yscale("log"); axs[1].set_ylabel("events per day (log)"); axs[1].set_title("Events per day (mainshock on 07-06)"); plt.setp(axs[1].get_xticklabels(), rotation=60, fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "dataset_ridgecrest_mag_time.png"); plt.close(fig)

# ---------- the 24-hour analysis set R in km ----------
R = ev[(ev.t > ms.t) & (ev.t <= ms.t + pd.Timedelta(hours=24))]
tr = Transformer.from_crs("EPSG:4326", f"+proj=aeqd +lat_0={ms.lat} +lon_0={ms.lon} +datum=WGS84 +units=km", always_xy=True)
x, y = tr.transform(R.lon.values, R.lat.values)
fig, ax = plt.subplots(figsize=(6.5, 6.2)); ax.set_aspect("equal")
ax.scatter(x, y, s=3, color=BLUE); ax.scatter(0, 0, marker="*", s=250, color=ORANGE, edgecolor=INK, zorder=5)
ax.set_xlabel("x east of mainshock (km)"); ax.set_ylabel("y north of mainshock (km)"); ax.set_title(f"Analysis set R: first 24 h after M7.1 (n={len(R):,})")
fig.tight_layout(); fig.savefig(OUT / "dataset_ridgecrest_R_km.png"); plt.close(fig)
print("figures written to", OUT)
