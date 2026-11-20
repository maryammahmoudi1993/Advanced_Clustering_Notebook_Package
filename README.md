# Advanced Clustering — A Hands-On Tutorial

> **Course note:** this code was provided for the **Data Mining** course taught by **Professor Nadimi**. Presenter: Maryam Mahmoudi.

Learn advanced clustering by *running and reading* real code. The notebook [Advanced_Clustering_Worked_Examples.ipynb](Advanced_Clustering_Worked_Examples.ipynb) works through two real datasets, and this README explains every part of it: the idea, the code line by line, and the figure it produces.

## What you will learn

| Topic | Algorithms | Dataset |
|---|---|---|
| Partitioning | K-means (Lloyd iterations by hand) | Wholesale customers |
| Hierarchical | AGNES (agglomerative), DIANA (divisive), 7 linkage rules, dendrogram cutting | Wholesale customers |
| Scalable / graph-based | BIRCH, CHAMELEON idea | Wholesale customers |
| Density-based | DBSCAN (by hand), OPTICS, HDBSCAN | Ridgecrest earthquakes |

## Contents

1. [Quick start: how to run the code](#1-quick-start-how-to-run-the-code)
2. [Project layout](#2-project-layout)
3. [The datasets](#3-the-datasets)
4. [Clustering in one page](#4-clustering-in-one-page)
5. [Code walkthrough, Part A: setup and helpers](#5-part-a-setup-and-helpers)
6. [Code walkthrough, Part B: Wholesale customers (hierarchical and K-means)](#6-part-b-wholesale-customers)
7. [Code walkthrough, Part C: Ridgecrest earthquakes (density-based)](#7-part-c-ridgecrest-earthquakes)
8. [Choosing an algorithm](#8-choosing-an-algorithm)
9. [Exercises](#9-exercises)
10. [Credits and licences](#10-credits-and-licences)

---

## 1. Quick start: how to run the code

You need Python 3.11+ (the results shipped here were produced with Python 3.12). No internet is needed at run time because both datasets are in `data/`.

**Windows (PowerShell)**

```powershell
git clone https://github.com/maryammahmoudi1993/Advanced_Clustering_Notebook_Package.git
cd Advanced_Clustering_Notebook_Package
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install jupyter
jupyter notebook Advanced_Clustering_Worked_Examples.ipynb
```

**macOS / Linux**

```bash
git clone https://github.com/maryammahmoudi1993/Advanced_Clustering_Notebook_Package.git
cd Advanced_Clustering_Notebook_Package
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt jupyter
jupyter notebook Advanced_Clustering_Worked_Examples.ipynb
```

Then choose **Kernel → Restart & Run All**. (In VS Code: open the `.ipynb`, pick the `.venv` kernel, click *Run All*.)

What happens when it runs:

* all plots are written to `figs/` (the images you see below),
* all numbers are written to `results/results.json`,
* the derived CSV files in `data/` are regenerated,
* about 20 assertions check that the hand-written algorithms give the same answer as scikit-learn/SciPy. The last line printed is `All assertions passed.`

Notes:

* Run the notebook **from the project folder**: all paths are relative (`data/`, `figs/`, `results/`).
* The notebook uses recent library versions (see `requirements.txt`). If `pip install` cannot find one, install the newest available versions; the algorithms do not change.
* Full run time is typically a few minutes; the slowest part is the DBSCAN timing comparison in section 2.4.

## 2. Project layout

```text
Advanced_Clustering_Notebook_Package/
├── Advanced_Clustering_Worked_Examples.ipynb   <- the code explained below
├── requirements.txt                            <- pinned package versions
├── data/                                       <- original datasets + derived CSVs  (see data/README.md)
├── datasets/                                   <- raw downloads under their original names
├── figs/                                       <- figures produced by the notebook
├── results/results.json                        <- every number used in the lecture slides
├── docs/images/                                <- dataset overview figures
└── scripts/make_dataset_figures.py             <- regenerates docs/images
```

## 3. The datasets

Full column dictionaries are in [data/README.md](data/README.md). In short:

**Wholesale customers (UCI)** - 440 customers of a wholesale distributor, with their annual spending on six product categories (`Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen`) plus `Channel` (Horeca / Retail) and `Region`. We cluster on `log10(Fresh)` and `log10(Grocery)`.

![Fresh vs Grocery](docs/images/dataset_wholesale_fresh_grocery.png)

**Ridgecrest 2019 earthquakes (USGS)** - 34,091 relocated earthquakes from the 2019 Ridgecrest, California sequence, including the M7.1 mainshock. We cluster the 3,505 epicentres from the 24 hours after the mainshock.

![Ridgecrest map](docs/images/dataset_ridgecrest_map.png)

---

## 4. Clustering in one page

**Clustering** groups objects so that objects in the same group are more similar to each other than to those in other groups. There is no "correct answer" column: it is *unsupervised* learning, so you judge a result by whether it is useful and stable.

Every clustering method must answer three questions:

1. **How do we measure closeness?** Here: Euclidean distance `d(x, y) = sqrt(Σ (xᵢ − yᵢ)²)`.
2. **What is a cluster?** A compact blob (K-means), a branch of a tree (hierarchical), or a dense region (DBSCAN).
3. **How many clusters?** Fixed in advance (K-means), chosen by cutting a tree (hierarchical), or discovered from density (DBSCAN).

| Family | Idea | Needs K? | Finds odd shapes? | Handles noise? |
|---|---|---|---|---|
| **K-means** | K centres; assign each point to its nearest centre; move centres to the mean; repeat | yes | no (round blobs) | no |
| **Hierarchical (AGNES)** | start with every point alone; repeatedly merge the two closest clusters | no (cut the tree) | depends on linkage | partly |
| **Hierarchical (DIANA)** | start with one cluster; repeatedly split | no | depends | partly |
| **BIRCH** | summarise points into tiny Clustering Features, then cluster the summaries | optional | no | partly |
| **DBSCAN** | a cluster = a dense region; sparse points are noise | no | yes | yes |
| **OPTICS / HDBSCAN** | DBSCAN without one fixed density threshold | no | yes | yes |

### The key ideas used in the notebook

* **Feature scaling matters.** Distances are dominated by the largest numbers, so we use `log10` for the skewed spending data.
* **K-means objective:** minimise `J = Σₖ Σ_{x∈Cₖ} ‖x − μₖ‖²` (total squared distance to the cluster centre). Each Lloyd iteration can only decrease `J`.
* **Linkage** = the rule that defines the distance *between two clusters* in hierarchical clustering:

| Linkage | Distance between clusters A and B | Typical behaviour |
|---|---|---|
| Single | smallest distance between a point of A and a point of B | long "chains" |
| Complete | largest such distance | compact clusters |
| Average | mean of all cross-pair distances | in between |
| Centroid | distance between the cluster means | can have *inversions* |
| Median (WPGMC) | distance between midpoint representatives | can have *inversions* |
| Medoid | distance between the most central real members | classroom rule |
| Ward | increase in within-cluster squared error when merged | balanced, K-means-like |

* **Inversion:** a merge that happens at a *lower* height than the previous merge, so the dendrogram "goes backwards". It can occur with Centroid and Median linkage.
* **DBSCAN** has two parameters: radius `ε` (eps) and `MinPts`. A point is **core** if at least `MinPts` points (itself included) lie within `ε`; **border** if it is not core but lies within `ε` of a core point; otherwise **noise**. A cluster is a set of core points connected through `ε`-neighbourhoods plus their border points.

![DBSCAN point types in a hand-traced window](figs/r2_H_status.png)

*Figure: blue circles are core points, orange triangles are border points, grey crosses are noise. The thin circles have radius ε = 1 km.*

---

## 5. Part A: setup and helpers

Notebook cell `[2]`. This cell prepares everything the rest of the notebook needs.

```python
import json, itertools, time, platform, importlib.metadata as md
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

* **Line 1** - standard-library tools: `json` (save results), `itertools` (all pairs of clusters), `time` (timing algorithms), `platform` and `importlib.metadata` (print library versions for reproducibility).
* **Line 2** - `Path` gives portable file paths (works on Windows and Linux).
* **Line 3** - `numpy` for arrays, `pandas` for tables.
* **Line 4** - `matplotlib.use("Agg")` selects a backend that draws straight to files, so plots are saved to `figs/` instead of popping up.

```python
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, cophenet
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment
from scipy.stats import skew
from sklearn.cluster import KMeans, DBSCAN, OPTICS, HDBSCAN, Birch
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score, silhouette_score
from pyproj import Transformer
```

| Import | What it does |
|---|---|
| `linkage` | builds the hierarchical tree (a *linkage matrix*) |
| `fcluster` | cuts the tree into flat clusters |
| `dendrogram` | draws the tree |
| `cophenet` | (imported; the notebook computes its own cophenetic distance) |
| `pdist`, `squareform` | all pairwise distances as a vector / as a square matrix |
| `linear_sum_assignment` | the Hungarian algorithm: matches cluster labels between two clusterings |
| `skew` | measures how lopsided a distribution is |
| `KMeans, DBSCAN, OPTICS, HDBSCAN, Birch` | the clustering algorithms |
| `NearestNeighbors` | fast "who is within ε of me?" searches |
| `adjusted_rand_score` | **ARI**: agreement of two clusterings (1 = identical, ≈0 = random) |
| `silhouette_score` | how well separated clusters are (−1…1, higher = better) |
| `Transformer` | converts latitude/longitude to kilometres |

```python
DATA, FIG, OUT = Path("data"), Path("figs"), Path("results")
for p in (FIG, OUT): p.mkdir(exist_ok=True)
VERS = {k: md.version(k) for k in [...]}
```

Defines the three folders and creates `figs/` and `results/` if missing. `VERS` records exact library versions and prints them.

```python
INK, BLUE, TEAL, ORANGE, GRID, ICE = "#0F172A", "#0284C7", "#0D9488", "#F97316", "#CBD5E1", "#F8FAFC"
CL = [BLUE, ORANGE, TEAL, "#7C3AED", "#CA8A04", "#BE123C", "#64748B", INK]
MK = ["o", "s", "^", "D", "v", "P", "X", "h"]
plt.rcParams.update({...})
```

A fixed colour palette `CL` and marker list `MK`. Cluster *k* always gets colour `CL[k]` **and** shape `MK[k]`, so figures stay readable in black-and-white. `rcParams` sets the look of every plot (background, fonts, 200 dpi).

```python
RES = {"versions": VERS}
def save(fig, name): ...
def jsonable(o): ...
def canon(lab): ...
def prov_csv(df, path, lines): ...
```

| Helper | Purpose |
|---|---|
| `RES` | a dictionary that collects every number; written to `results/results.json` at the end |
| `save(fig, name)` | saves a figure into `figs/` and closes it |
| `jsonable(o)` | converts numpy types into plain Python so they can be saved as JSON |
| `canon(lab)` | **canonical relabelling**: renames labels by order of first appearance, so `[2,2,0,1]` and `[0,0,1,2]` are recognised as the *same partition*. Cluster numbers are arbitrary, so this is needed to compare clusterings |
| `prov_csv(df, path, lines)` | writes a CSV whose first lines are `#` comments documenting the data source |

---

## 6. Part B: Wholesale customers

Teaching question: *can we find customer purchasing profiles, and which profiles survive when we change the linkage rule or the tree cut?*

### B1. Loading and checking the data (cell `[4]`)

```python
w = pd.read_csv(DATA / "Wholesale_customers_data.csv")
w.insert(0, "source_row_id", np.arange(1, len(w) + 1))
CAT = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
```

* `read_csv` loads the 440 × 8 table.
* `insert(0, "source_row_id", 1…440)` adds a stable ID column (row number in the original file) so any customer can be traced back later.
* `CAT` lists the six spending columns.

```python
assert w.shape == (440, 9) and w.isna().sum().sum() == 0
assert set(w.Channel) == {1, 2} and set(w.Region) == {1, 2, 3}
assert w.Channel.value_counts().to_dict() == {1: 298, 2: 142} and ...
```

`assert` stops the notebook with an error if the file is not what we expect (9 columns after the new ID, no missing values, known category counts). This protects against a wrong or edited dataset.

```python
print(w[CAT].mean().round(2).to_dict())
RES["wholesale_source"] = {...}
```

Prints the mean of each category and stores it in `RES`.

### B2. Skew and the log transform (cell `[6]`)

```python
w["lf"], w["lg"] = np.log10(w.Fresh), np.log10(w.Grocery)
```

Creates `lf = log10(Fresh)` and `lg = log10(Grocery)`. **Why log?** A few customers spend 10-100× more than the typical one. In raw units those customers would completely decide every distance. In `log10` units, *equal ratios become equal distances* (+1 means ×10).

```python
sk = {"raw_skew": {c: float(skew(w[c])) ...}, "log10_skew": {...}}
chan_med = w.groupby("Channel")[["Fresh", "Grocery"]].median()
```

`skew` quantifies lopsidedness: raw Fresh = **2.55**, raw Grocery = **3.58** (strongly right-skewed). After `log10` the values are −1.62 and −0.74: the long right tail is gone (a mild left tail remains). `groupby("Channel").median()` shows Horeca spends more on Fresh (median 9,582 vs 5,994) and Retail on Grocery (12,390 vs 2,684): the two channels move in opposite directions, which is why this plane is interesting.

The plotting loop draws a 2×2 grid of histograms (raw on top, log below):

![Skew before and after log10](figs/w1_skew.png)

*How to read it: the top histograms have a huge spike near zero and a long right tail; the bottom ones are much more symmetric. Clustering on the bottom version is meaningful.*

### B3. Choosing a reproducible sample (cells `[8]`, `[9]`)

Hand-tracing an algorithm on 440 points is impossible, so the notebook uses two nested, reproducible samples:

* **S40** (40 customers): `source_row_id % 11 == 2`
* **S8** (8 customers): every 5th row of S40, labelled `a`…`h`

So S8 ⊂ S40 ⊂ all 440 rows, and every hand calculation uses real customers.

```python
def medoid_linkage(X): ...
```

A *classroom* linkage rule (not in SciPy). Line by line:

1. `D = squareform(pdist(X))` - full distance matrix.
2. `members`, `med` - dictionaries: which points are in each cluster and which member is its **medoid** (the member with the smallest total distance to the others).
3. The `while` loop runs until one cluster is left. Inside, `itertools.combinations(sorted(members), 2)` tries every pair of clusters, `d = D[med[a], med[b]]` is the distance between their medoids, and the closest pair `best` wins.
4. The pair is merged, the new medoid computed, and a SciPy-style row `[a, b, distance, size]` is appended to `Z`.

```python
METHODS6 = ["single", "complete", "average", "centroid", "median", "ward"]
def all_Z(X): return {**{m: linkage(X, m) for m in METHODS6}, "medoid": medoid_linkage(X)}
def has_inversion(Z): return bool((np.diff(Z[:, 2]) < -1e-12).any())
```

`all_Z` builds the tree under all seven rules. `has_inversion` checks whether any merge height is lower than the one before it (`np.diff` of the height column < 0).

The next block loops over every residue `r = 0…10` and start phase `p = 0…4`, and keeps the *first* combination for which S8 has (i) no tied distances, (ii) an inversion for both Centroid and Median linkage, and (iii) at least 3 different 4-cluster results across methods. It lands on **residue 2, phase 0** (checked by an `assert`). This is a **curated teaching sample**, not a random one, and the scan code lets you audit that.

The last part of cell `[9]` builds `S40`, `S8`, saves them as CSV with `prov_csv`, and draws:

![S8 inside S40 inside all 440 customers](figs/w1_subsets.png)

### B4. K-means by hand (cells `[11]`, `[12]`)

**Concept.** Pick K centres, then repeat two steps until nothing changes:

1. **Assignment:** give each point to its nearest centre.
2. **Update:** move each centre to the mean of its points.

`J` (sum of squared distances to the own centre) never increases.

```python
def lloyd(X, init_idx, maxit=20):
    c = X[init_idx].copy(); hist = []
    for it in range(maxit):
        d = np.linalg.norm(X[:, None] - c[None], axis=2); lab = d.argmin(1); J = float((d.min(1) ** 2).sum())
        newc = np.array([X[lab == k].mean(0) for k in range(len(c))])
        hist.append({...})
        if np.allclose(newc, c): break
        c = newc
    return hist
```

| Line | Meaning |
|---|---|
| `c = X[init_idx].copy()` | the starting centres are the observations chosen by index |
| `d = np.linalg.norm(X[:, None] - c[None], axis=2)` | broadcasting trick: an (n × K) table of distances from every point to every centre |
| `lab = d.argmin(1)` | assignment step: nearest centre per point |
| `J = ...(d.min(1) ** 2).sum()` | the objective for the current assignment |
| `newc = ...X[lab == k].mean(0)` | update step: mean of each cluster |
| `np.allclose(newc, c)` | converged when centres stop moving |

`INIT = [0, 3, 7]` starts the centres on customers **a, d, h**. The loop takes 3 iterations and ends at `J = 0.4488`. Then

```python
km_sk = KMeans(3, init=XS8[INIT], n_init=1).fit(XS8)
assert canon(km_sk.labels_) == canon(H[-1]["labels"]) and abs(km_sk.inertia_ - H[-1]["J"]) < 1e-9
```

proves the hand-written loop gives **exactly** the same partition and `J` as scikit-learn (`canon` makes the comparison independent of label numbering).

**Practical effects on S40 (cell `[12]`).**

```python
for k in Ks: km = KMeans(k, n_init=10, random_state=0).fit(XS40); inertia.append(km.inertia_); sil.append(silhouette_score(...))
```

`n_init=10` runs 10 random starts and keeps the best; `random_state=0` makes it repeatable.

![K-means practical effects](figs/w1_kmeans_effects.png)

| Panel | What it teaches | Result |
|---|---|---|
| (a) Choosing K | `J` always falls as K grows (look for an "elbow"); silhouette measures separation | silhouette is highest at K = 4 (0.432) |
| (b) Scaling | K-means on *raw* m.u. vs on log10 | the two partitions barely agree: ARI = −0.01 |
| (c) Initialization | 20 single random starts | 9 different partitions, J from 7.03 to 7.51: bad starts give worse local optima |
| (d) Geometry | the decision regions | straight (Voronoi) boundaries: K-means can only make convex, roughly round clusters |

### B5. AGNES (average linkage) by hand (cell `[14]`)

**Concept.** AGNES = *agglomerative nesting*, "bottom-up":

1. Every point is its own cluster; compute the distance matrix.
2. Merge the two clusters with the **smallest** distance.
3. Update the matrix (the Lance-Williams formula) and repeat until one cluster is left.

(If you have a *similarity* instead of a distance, you merge the **largest** value; the pair chosen is the same.)

```python
D8 = squareform(pdist(XS8)); names = list(S8.pt)
```

The 8 × 8 matrix of Euclidean distances between customers `a`…`h` in log10 space.

```python
def agnes_by_hand(D, method="average"):
    n = len(D); act = [[i] for i in range(n)]; M = D.copy().astype(float); steps = []
    while len(act) > 1:
        best = None
        for i, j in itertools.combinations(range(len(act)), 2):
            if best is None or M[i, j] < best[0] - 1e-12: best = (M[i, j], i, j)
```

* `act` = list of active clusters (each a list of point indices).
* The double loop scans every pair and remembers the smallest `M[i, j]` (ties: first pair wins).

```python
        for k in range(len(act)):
            if k in (i, j): continue
            if method == "single": v = min(M[i, k], M[j, k])
            elif method == "complete": v = max(M[i, k], M[j, k])
            else: v = (ni * M[i, k] + nj * M[j, k]) / (ni + nj)
```

The **Lance-Williams update**: the distance from the new merged cluster to every other cluster `k` is computed from the old distances. Single takes the min, Complete the max, Average the size-weighted mean. Then the matrix is rebuilt without rows `i, j` and with one new row/column for the merged cluster.

```python
assert np.allclose([s["height"] for s in AG], Zavg8[:, 2]) and ...
```

Checks that the hand-coded merge heights equal SciPy's `linkage(XS8, "average")`.

**Result on S8 (average linkage):**

| Merge | Clusters joined | Height |
|---|---|---|
| 1 | d + e | 0.081 |
| 2 | a + {d,e} | 0.146 |
| 3 | c + g | 0.191 |
| 4 | b + {a,d,e} | 0.466 |
| 5 | {c,g} + {a,b,d,e} | 0.481 |
| 6 | f + {a,b,c,d,e,g} | 0.730 |
| 7 | h + the rest | 1.110 |

### B6. DIANA by hand (cell `[16]`)

**Concept.** DIANA = *divisive analysis*, "top-down" (Kaufman and Rousseeuw, 1990):

1. Start with all points in one cluster. Pick the cluster with the **largest diameter** (largest pairwise distance).
2. The **splinter** group starts with the member whose *average* distance to the others is largest.
3. Repeatedly move into the splinter group the point `i` with the largest positive value of `mean d(i, rest) − mean d(i, splinter)`. Stop when no value is positive.
4. The split height is the diameter of the cluster that was split.

```python
diam = [D[np.ix_(c, c)].max() if len(c) > 1 else -1 for c in clusters]; ci = int(np.argmax(diam))
avg = {i: float(D[i, [j for j in C if j != i]].mean()) for i in C}; s = max(C, key=lambda i: (avg[i], -i))
```

* `D[np.ix_(c, c)]` extracts the sub-matrix of one cluster; `.max()` is its diameter; `argmax` picks the widest cluster.
* `avg` = each member's mean distance to the others; `s` is the most "isolated" member, and `-i` breaks ties toward the smaller index.

```python
while len(Rm) > 1:
    tab = {i: (mean_dist_to_remaining, mean_dist_to_splinter) for i in Rm}
    diffs = {i: a - b for i, (a, b) in tab.items()}; best = max(Rm, key=lambda i: (diffs[i], -i))
    if diffs[best] <= 0: break
    S.append(best); Rm.remove(best)
```

This is the moving loop of step 3: compute the table, move the best candidate, stop when nobody prefers the splinter group.

Because DIANA's height is the *diameter* it differs from a generic "bisecting" method that would just run 2-means on a cluster.

### B7. Seven linkage rules on the same 8 customers (cells `[18]`, `[19]`)

All seven methods share the first three merges (d+e, a joins, c+g), leaving the state **{a,d,e}, {b}, {c,g}, {f}, {h}**. Then each rule answers one question: *which two clusters merge next?*

```python
def partition_k(Z, n, k): ...      # cut the tree by merge ORDER
def cophenetic(Z, n): ...          # height at which each pair of points first joins
def reps_wpgmc(Z, X): ...          # Median (WPGMC) representatives
```

* `partition_k` replays the first `n − k` merges to obtain `k` clusters. It is used instead of a height cut because with *inversions* a height threshold can be ambiguous.
* `cophenetic` fills a matrix with the merge height of every pair; its correlation with the real distances measures how faithfully the tree preserves the data.
* `reps_wpgmc`: in Median linkage the representative of a merged cluster is the **midpoint of its two parents' representatives** (not the coordinate-wise median, not the centroid).

`cand_value(method, ci, cj)` computes each rule's cost for a candidate pair:

| Method | Formula in code |
|---|---|
| Single | `cross.min()` |
| Complete | `cross.max()` |
| Average | `cross.mean()` |
| Centroid | distance between the two cluster means |
| Median | distance between WPGMC representatives |
| Medoid | `D8[medoid_of(A), medoid_of(B)]` |
| Ward | `nA·nB/(nA+nB) · ‖cA − cB‖²` (increase in error, ΔESS) |

**Which pair does each rule merge next?**

| Rule | Next merge |
|---|---|
| Single, Median | {a,d,e} + {c,g} |
| Complete, Average, Centroid, Medoid | {a,d,e} + b |
| Ward | b + {c,g} |

Same data, same state, three different decisions: the **linkage rule is a modelling choice**, not a detail.

**Inversions.** On S8 both Centroid and Median show one: after a merge at height 0.461 (Centroid) the next merge happens *lower*, at 0.427. A real dendrogram cannot be cut cleanly there, which is why `has_inversion` exists.

### B8. Dendrograms and comparisons on 40 customers (cell `[21]`)

```python
dist40 = pdist(XS40)            # 780 pairwise distances
Zs40 = all_Z(XS40)
```

**Reading a linkage-matrix row `[i, j, h, s]`:** clusters `i` and `j` merge at height `h` into a cluster of `s` points. IDs 0-39 are the original customers; ID `40 + r` is the cluster created by row `r`.

The table `M40` records, per method, whether an inversion exists and the **cophenetic correlation** (`np.corrcoef(cophenetic, dist40)`). The partitions at K = 4 are then compared:

```python
P = {m: partition_k(Z, nL, K) for m, Z in Zs40.items()}
def map_to(ref, lab): ... linear_sum_assignment(-C) ...
```

Cluster numbers are arbitrary, so `map_to` builds a contingency table against the Ward reference and applies the **Hungarian algorithm** to renumber clusters for maximum overlap. Then `adjusted_rand_score` (ARI) and the **Jaccard** overlap of each cluster measure how much two methods agree.

![Dendrograms: single, complete, average, centroid](figs/w1_dendrograms_a.png)

*How to read a dendrogram: the x-axis lists the customers (by `source_row_id`), the y-axis is the merge height. Two leaves joined low are very similar. Single linkage produces a "staircase" (chaining), Complete produces balanced compact branches, and the Centroid panel is flagged **"inversion present"**: a branch that sits lower than the one it joins.*

![Dendrograms: median, medoid, ward](figs/w1_dendrograms_b.png)

![K = 4 partitions: single, complete, average, centroid](figs/w1_partitions_k4_a.png)

![K = 4 partitions: median, medoid, ward](figs/w1_partitions_k4_b.png)

*Same data, same K = 4, different groupings: the choice of linkage changes the clusters.*

### B9. Cutting the dendrogram (cell `[23]`)

Complete linkage has no inversions and a clear meaning: the height is the **largest distance between any two customers inside the cluster**, so a cut at height `h` guarantees no pair inside a cluster is farther apart than `h`.

```python
lab = fcluster(Zav, hcut, "distance") - 1
```

`fcluster(..., "distance")` cuts at height `hcut` and returns a cluster number per customer (minus 1 to start from 0). The code then renumbers clusters by size and saves a dendrogram with the cut line plus a scatter plot for each of three cuts (heights 2.40, 1.80 and 1.40, chosen inside gaps between merge heights). A lower cut line gives more, smaller clusters.

![Cut at 2.40](figs/w1_cut_state1.png)
![Cut at 1.80](figs/w1_cut_state2.png)
![Cut at 1.40](figs/w1_cut_state3.png)

*The orange line is the cut. Every vertical branch it crosses becomes one cluster.*

### B10. Interpreting the clusters (cell `[25]`)

Clusters are fitted only on the two log features; the other four categories, `Channel` and `Region` are looked at *afterwards* to describe them. The reference partition is **Ward, K = 4** on S40.

```python
REF = PM["ward"]; S40["ward4"] = REF + 1
prof = S40.groupby("ward4")[CAT].median().round(0).astype(int)
ct_ch = pd.crosstab(S40.ward4, S40.Channel)
```

* `groupby("ward4")[CAT].median()` - the typical (median) spending of each cluster in every category: a **cluster profile**.
* `pd.crosstab` - counts of Horeca/Retail (and Region) inside each cluster.

Median annual spending (m.u.) of the four Ward clusters:

| Cluster | n | Fresh | Milk | Grocery | Frozen | Detergents_Paper | Delicassen | Reading |
|---|---|---|---|---|---|---|---|---|
| 1 | 11 | 5,181 | 9,810 | 13,567 | 1,465 | 4,973 | 1,776 | grocery / milk / detergent heavy |
| 2 | 13 | 12,759 | 7,330 | 6,964 | 1,646 | 2,730 | 2,503 | high Fresh, broad basket |
| 3 | 6 | 1,354 | 3,658 | 2,568 | 470 | 498 | 900 | small accounts |
| 4 | 10 | 11,150 | 1,780 | 1,708 | 1,217 | 344 | 570 | Fresh-focused |

![Profile heat map](figs/w1_profiles.png)

![Ward K=4 scatter](figs/w1_ward4_scatter.png)

*The names in the "Reading" column are our own descriptive labels, not something the algorithm knows.*

**A wider check.** The notebook then runs Ward with K = 4 on **all 440** customers and compares it with the S40 result on the same 40 rows: `adjusted_rand_score(REF, Pall[S40.source_row_id.values - 1])` = **0.55**. The two partitions agree moderately, a reminder that a 40-row sample is a teaching tool and conclusions must be re-checked on the full data.

![Ward K=4 on all 440 rows](figs/w1_wide440.png)

### B11. BIRCH and CHAMELEON (cell `[27]`)

**BIRCH** compresses data into **Clustering Features** `CF = (N, LS, SS)`: the point count, the linear sum and the sum of squares. Two properties make BIRCH scalable:

* CFs are **additive**: `CF(A ∪ B) = CF(A) + CF(B)`, so a point can be absorbed without storing it.
* From a CF you can recover the centroid `LS / N` and the radius `sqrt(SS/N − ‖LS/N‖²)`.

```python
def CF(P): P = np.atleast_2d(P); return {"N": len(P), "LS": P.sum(0), "SS": float((P ** 2).sum())}
cf_de, cf_a, cf_ade = CF(XS8[[3, 4]]), CF(XS8[[0]]), CF(XS8[[0, 3, 4]])
assert add["N"] == cf_ade["N"] and np.allclose(add["LS"], cf_ade["LS"]) ...
```

The code builds the CF of customers {d, e}, of {a}, and of {a, d, e}, then **asserts that CF(d,e) + CF(a) equals CF(a,d,e)**: the additivity property, verified on real numbers. The centroid of {a,d,e} is (3.755, 3.990) and its radius 0.075.

```python
br = Birch(threshold=0.25, branching_factor=4, n_clusters=None).fit(XS40)
```

`threshold` is the maximum radius of a leaf sub-cluster; `branching_factor` is the maximum number of children per tree node; `n_clusters=None` keeps the leaf sub-clusters as the output. The 40 customers are summarised by **10** leaf sub-clusters.

**CHAMELEON** (illustration only; scikit-learn has no CHAMELEON) builds a k-nearest-neighbour graph, splits it into many small pieces and merges the pair with high *relative interconnectivity* and *relative closeness*.

```python
nn = NearestNeighbors(n_neighbors=4).fit(XS40); dk, ik = nn.kneighbors(XS40)
edges = {(min(i, j), max(i, j)): d ...}
cross = [(e, d) for e, d in edges.items() if REF[e[0]] != REF[e[1]]]
```

`n_neighbors=4` includes the point itself, so each point gets its 3 nearest neighbours (a 3-NN graph, 76 unique edges). `cross` lists the edges that connect *different* Ward clusters. Only 9 of 76 edges cross clusters, and they are longer (mean 0.48) than the average edge (0.27): clusters are connected internally and weakly connected to each other, which is the intuition CHAMELEON uses.

![BIRCH sub-clusters and 3-NN graph](figs/w1_birch_chameleon.png)

*Left: squares are BIRCH leaf summaries of the 40 points. Right: the 3-NN graph; orange edges cross cluster borders.*

---

## 7. Part C: Ridgecrest earthquakes

Teaching question: *how can an analyst find dense concentrations of events and isolated events after a big earthquake?* A "cluster" here is a **concentration of recorded events** in a 2-D map. Calling it a fault needs independent geological evidence.

### C1. Reading the catalogue (cell `[29]`)

```python
RAW = DATA / "ridgecrest_shelly2019_catalog_raw.txt"
lines = RAW.read_text().splitlines(); hdr = [l for l in lines if l.startswith("#")]
src_line = [i + 1 for i, l in enumerate(lines) if l.strip() and not l.startswith("#")]
```

Reads the file as text, collects the `#` header and records the **line number** of each data row (`source_line`) so every event can be located in the original file.

```python
ev = pd.read_csv(RAW, comment="#", sep=r"\s+", header=None, names=[...])
ev["t_utc"] = pd.to_datetime(dict(year=ev.year, month=ev.month, day=ev.day, hour=ev.hour, minute=ev.minute)) + pd.to_timedelta(ev.second, unit="s")
```

* `comment="#"` skips header lines; `sep=r"\s+"` splits on any run of spaces; `header=None` + `names=` supply the 11 column names.
* The time columns are combined into one timestamp; seconds (with decimals) are added as a timedelta.

```python
assert len(ev) == 34091 and ev.event_id.is_unique and ev.t_utc.is_monotonic_increasing
ms = ev.loc[ev.mag.idxmax()]; assert ms.mag == 7.1
```

Sanity checks (34,091 events, unique IDs, sorted by time), then `idxmax` finds the **mainshock**: magnitude 7.1 on 2019-07-06 03:19:52.86 at (35.7718° N, −117.5942° E), depth 3.2 km. The file gives no time zone; the notebook treats the times as UTC because the M7.1 time matches the published 03:19:53 UTC.

![Events per day and magnitude distribution](docs/images/dataset_ridgecrest_mag_time.png)

*Left: many small events and few large ones (log scale). Right: activity peaks on the mainshock day (07-06) and then decays.*

### C2. From degrees to kilometres (cells `[31]`)

Degrees of longitude and latitude are not the same length on the ground, so a circle of radius ε in degrees would be an ellipse in reality. The notebook projects every epicentre onto a flat plane in km, centred on the mainshock:

```python
TR = Transformer.from_crs("EPSG:4326", f"+proj=aeqd +lat_0={ms.lat} +lon_0={ms.lon} +datum=WGS84 +units=km", always_xy=True)
ev["x_km"], ev["y_km"] = TR.transform(ev.lon.values, ev.lat.values)
```

* `EPSG:4326` = ordinary latitude/longitude on the WGS84 ellipsoid.
* `+proj=aeqd` = **azimuthal equidistant** projection: distances from the centre are exact and nearby distances are almost exact.
* `+units=km` makes `x_km` (east) and `y_km` (north) directly usable as Euclidean coordinates.
* `always_xy=True` fixes the argument order as (lon, lat).

To prove the projection is accurate the code compares 5,000 random pairs with the great-circle **haversine** distance:

```python
def haversine(la1, lo1, la2, lo2, Rk=6371.0088): ...
rel = np.abs(dp - dh) / np.maximum(dh, 1e-9)
```

Result: the largest relative error for pairs more than 5 km apart is **0.23 %**. Note that the analysis uses epicentres only, so **depth is ignored** (we measure what that costs in section C6).

### C3. The analysis set R and the k-distance graph (cell `[33]`)

```python
R = ev[(ev.t_utc > ms.t_utc) & (ev.t_utc <= ms.t_utc + pd.Timedelta(hours=24))].reset_index(drop=True)
XR = R[["x_km", "y_km"]].values; MINPTS = 5
```

**R** = events strictly after the mainshock and within the next 24 h: **3,505 events**, magnitude 0.59 to 5.5. The catalogue is less complete in the first hours after a large shock, so R describes *recorded* events only. `XR` is the (x, y) matrix, and `MINPTS = 5` is the DBSCAN density threshold.

![Analysis set R in kilometres](docs/images/dataset_ridgecrest_R_km.png)

**How to choose ε: the k-distance graph.**

```python
def kdist(X, m=MINPTS):
    return NearestNeighbors(n_neighbors=m).fit(X).kneighbors(X)[0][:, -1]
```

For each event, the distance to its 5th nearest event *counting itself* (the nearest neighbour of a training point is itself at distance 0). A point is a **core point** exactly when `kdist ≤ ε`. If you sort these distances, small values are dense areas and the sharp rise at the end is noise; a good ε sits at the "knee".

```python
def knee(a):
    a = np.sort(a); xs = np.arange(len(a)) / (len(a) - 1); ys = (a - a.min()) / (a.max() - a.min()); i = int(np.argmax(xs - ys))
```

A simple knee finder: scale the sorted curve to the unit square and take the point farthest below the straight chord joining its ends.

The three ε values used from here on are **0.5 km (smaller), 1.0 km (candidate), 2.0 km (larger)**.

```python
assert (core_flag == (kd <= EPS["candidate"])).all() and (core_flag == (cnt >= MINPTS)).all()
```

Three ways of finding core points (scikit-learn's `core_sample_indices_`, the k-distance rule, and a brute-force neighbour count) must agree for all 3,505 events.

![k-distance graph](figs/r2_kdist.png)

*How to read it: each blue point is an event sorted by its 5-distance (log scale). Horizontal lines are the three ε candidates. Events below a line are core points at that ε.*

