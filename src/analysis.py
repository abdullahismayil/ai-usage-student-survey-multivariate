"""
Generative-AI use, stress and academic success among university students.
Python replication of the logistic-regression and CATPCA parts of an
IST646 (Multivariate Statistical Methods, Hacettepe University) final project,
originally done in IBM SPSS 29.

Input : data/survey.sav   (NOT included in this repository - see data/README.md)
Output: results/*.csv and figures/*.png

Run   : python src/analysis.py [path/to/survey.sav]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.isotonic import IsotonicRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "survey.sav"
RES, FIG = ROOT / "results", ROOT / "figures"
RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

import pyreadstat
df, meta = pyreadstat.read_sav(str(PATH))
print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

# ---------------------------------------------------------------- Part 1: logit
# Outcome: GPA in 3.00-4.00 (1) vs lower (0). Reference categories: YZ=Az, Stres=Dusuk, Cinsiyet=Erkek
d = df.dropna(subset=["Basari", "YZ_Siklik", "Stres", "Cinsiyet"]).copy()
for c in ["YZ_Siklik", "Stres", "Cinsiyet"]:
    d[c] = d[c].astype(int)
fit = smf.logit("Basari ~ C(YZ_Siklik) + C(Stres) + C(Cinsiyet)", d).fit(disp=0)
null = smf.logit("Basari ~ 1", d).fit(disp=0)
n = len(d)
lr = 2 * (fit.llf - null.llf)
cs = 1 - np.exp(2 * (null.llf - fit.llf) / n)
nag = cs / (1 - np.exp(2 * null.llf / n))

names = ["Constant", "YZ use: Orta (vs Az)", "YZ use: Cok (vs Az)",
         "Stress: Orta (vs Dusuk)", "Stress: Yuksek (vs Dusuk)", "Gender: Kadin (vs Erkek)"]
ci = fit.conf_int()
coef = pd.DataFrame({
    "term": names, "B": fit.params.values, "SE": fit.bse.values,
    "Wald": (fit.params / fit.bse).values ** 2, "p": fit.pvalues.values,
    "Exp(B)": np.exp(fit.params.values),
    "CI95_low": np.exp(ci[0].values), "CI95_high": np.exp(ci[1].values)}).round(4)
coef.to_csv(RES / "logit_coefficients.csv", index=False)

# Hosmer-Lemeshow (10 groups of predicted risk, as in SPSS)
p = fit.predict(d)
# SPSS-style groups: ~10 groups of predicted risk, tied probabilities are never split
vals = np.sort(p.unique()); cnt = p.value_counts().sort_index()
grp, cum, gi = {}, 0, 0
for v in vals:
    grp[v] = gi; cum += cnt[v]
    if cum >= (gi + 1) * n / 10: gi += 1
g = p.map(grp)
hl_tab = d.assign(p=p, g=g).groupby("g").agg(obs=("Basari", "sum"), exp=("p", "sum"), n=("Basari", "size"))
hl = (((hl_tab.obs - hl_tab.exp) ** 2) / (hl_tab.exp * (1 - hl_tab.exp / hl_tab.n))).sum()
hl_df = g.nunique() - 2
hl_p = 1 - stats.chi2.cdf(hl, hl_df)

pred = (p >= 0.5).astype(int)
cm = pd.crosstab(d.Basari, pred).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
acc = (pred == d.Basari).mean()
cm.to_csv(RES / "logit_classification_table.csv")

summary = pd.DataFrame({"metric": ["N", "-2 log-likelihood", "Omnibus chi2 (df=5)", "Omnibus p",
                                   "Cox & Snell R2", "Nagelkerke R2", f"Hosmer-Lemeshow chi2 (df={hl_df})",
                                   "Hosmer-Lemeshow p", "Classification accuracy"],
                        "value": [n, -2 * fit.llf, lr, 1 - stats.chi2.cdf(lr, 5), cs, nag, hl, hl_p, acc]}).round(4)
summary.to_csv(RES / "logit_model_summary.csv", index=False)
print(summary.to_string(index=False)); print(coef.to_string(index=False))

# ----------------------------------------------------------------- Part 2: CATPCA
items = ["Odev", "Proje", "Sinav", "DeepResearch", "KisiselGelisim", "SosyalMedya", "Sohbet", "SanalAsistan"]
X = df[items].dropna().astype(int)
N, m = X.shape


def standardise(q):
    q = q - q.mean()
    return q / np.sqrt((q ** 2).mean())


def catpca(X, ndim=2, iters=500, tol=1e-10, seed=0):
    """Ordinal (monotone) CATPCA by alternating least squares (Gifi / PRINCALS-type).
    Loss = sum_j || Q_j - Z A_j' ||^2 with Q_j the monotone, standardised quantification of item j."""
    rng = np.random.default_rng(seed)
    Q = np.column_stack([standardise(X[c].values.astype(float)) for c in X])
    Z = np.linalg.qr(rng.normal(size=(N, ndim)))[0] * np.sqrt(N)
    prev = np.inf
    for _ in range(iters):
        # 1) object scores: leading principal components of the quantified data
        U, S, Vt = np.linalg.svd(Q, full_matrices=False)
        Z = U[:, :ndim] * np.sqrt(N)
        A = (Q.T @ Z) / N                      # component loadings (m x ndim)
        fitted = Z @ A.T
        # 2) optimal scaling: monotone regression of fitted values on the item categories
        for j, c in enumerate(X):
            x = X[c].values.astype(float)
            iso = IsotonicRegression(increasing=True).fit(x, fitted[:, j])
            q = iso.predict(x)
            Q[:, j] = standardise(q) if q.std() > 0 else q
        loss = ((Q - fitted) ** 2).sum() / (N * m)
        if abs(prev - loss) < tol:
            break
        prev = loss
    # final solution
    R = np.corrcoef(Q.T)
    ev = np.sort(np.linalg.eigvalsh(R))[::-1]
    return Q, R, ev


Q, R, ev = catpca(X)
alpha = lambda e: m / (m - 1) * (1 - 1 / e)
vaf = pd.DataFrame({"dimension": range(1, 9), "eigenvalue": ev, "pct_variance": ev / m * 100,
                    "cronbach_alpha": [alpha(e) if e > 1 else np.nan for e in ev]})
vaf.round(4).to_csv(RES / "catpca_eigenvalues.csv", index=False)
print(vaf.head(3).round(3))
tot = ev[:2].sum()
print(f"2-dim total eigenvalue {tot:.3f} ({tot / m * 100:.1f}%), Cronbach alpha (total) {alpha(tot):.3f}")
pd.DataFrame(R, index=items, columns=items).round(3).to_csv(RES / "catpca_transformed_correlations.csv")

# unrotated loadings, then oblimin (via factor_analyzer-free direct oblimin, delta=0)
w, V = np.linalg.eigh(R); o = np.argsort(w)[::-1]
L = V[:, o[:2]] * np.sqrt(w[o[:2]])
L = L * np.sign(L.sum(0))                       # make dimension 1 positive
pd.DataFrame(L, index=items, columns=["Dim1", "Dim2"]).round(3).to_csv(RES / "catpca_loadings_unrotated.csv")


def oblimin(L, gamma=0.0, iters=1000, tol=1e-9):
    """Direct oblimin (gradient projection, Jennrich 2002). Returns pattern loadings and factor correlations."""
    p, k = L.shape
    T = np.eye(k); al = 1.0
    def crit(T):
        Lr = L @ np.linalg.inv(T).T
        L2 = Lr ** 2
        C = np.eye(p) - gamma / p * np.ones((p, p))
        f = np.sum(L2 * (C @ L2 @ (np.ones((k, k)) - np.eye(k)))) / 4
        Gq = Lr * (C @ L2 @ (np.ones((k, k)) - np.eye(k)))
        G = -(Lr.T @ Gq @ np.linalg.inv(T)).T
        return f, G
    f, G = crit(T)
    for _ in range(iters):
        Gp = G - T @ np.diag(np.sum(T * G, axis=0))
        s = np.linalg.norm(Gp)
        if s < tol:
            break
        al *= 2
        for _ in range(10):
            X_ = T - al * Gp
            Tt = X_ / np.sqrt(np.sum(X_ ** 2, axis=0))
            ft, Gt = crit(Tt)
            if ft < f - 0.5 * s ** 2 * al:
                break
            al /= 2
        T, f, G = Tt, ft, Gt
    Lr = L @ np.linalg.inv(T).T
    Phi = T.T @ T
    return Lr, Phi


h = np.sqrt((L ** 2).sum(1, keepdims=True))
Lr, Phi = oblimin(L / h)
Lr = Lr * h
pd.DataFrame(Lr, index=items, columns=["Dim1", "Dim2"]).round(3).to_csv(RES / "catpca_pattern_oblimin.csv")
print("Component correlation after oblimin:", round(Phi[0, 1], 3))
pd.DataFrame(Phi, index=["Dim1", "Dim2"], columns=["Dim1", "Dim2"]).round(3).to_csv(RES / "catpca_component_correlation.csv")

# ------------------------------------------------------------------- Figures
fig, ax = plt.subplots(figsize=(6, 5.5))
for i, c in enumerate(items):
    ax.arrow(0, 0, L[i, 0], L[i, 1], head_width=0.015, color="#2b4c7e", length_includes_head=True)
    ax.text(L[i, 0] * 1.04, L[i, 1] * 1.04, c, fontsize=9)
ax.axhline(0, c="grey", lw=.5); ax.axvline(0, c="grey", lw=.5)
ax.set_xlim(0, 1.05); ax.set_ylim(-.8, .8); ax.set_xlabel("Dimension 1"); ax.set_ylabel("Dimension 2")
ax.set_title("CATPCA component loadings (unrotated)")
fig.tight_layout(); fig.savefig(FIG / "01_catpca_loadings.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.bar(range(1, 9), ev, color="#2b4c7e"); ax.axhline(1, c="firebrick", ls="--", lw=1)
ax.set_xlabel("Dimension"); ax.set_ylabel("Eigenvalue"); ax.set_title("CATPCA scree (Kaiser criterion = 1)")
fig.tight_layout(); fig.savefig(FIG / "02_catpca_scree.png", dpi=150); plt.close(fig)

orr = coef.iloc[1:]
fig, ax = plt.subplots(figsize=(7, 3.6))
y = np.arange(len(orr))[::-1]
ax.errorbar(orr["Exp(B)"], y, xerr=[orr["Exp(B)"] - orr["CI95_low"], orr["CI95_high"] - orr["Exp(B)"]], fmt="o", color="#2b4c7e", capsize=3)
ax.axvline(1, c="grey", ls="--"); ax.set_yticks(y); ax.set_yticklabels(orr["term"]); ax.set_xscale("log")
ax.set_xlabel("Odds ratio, 95% CI (log scale)"); ax.set_title("Odds of GPA 3.00-4.00")
fig.tight_layout(); fig.savefig(FIG / "03_odds_ratios.png", dpi=150); plt.close(fig)
