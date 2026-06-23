import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.14.3"},
}

cells = []
md   = lambda t: nbf.v4.new_markdown_cell(t)
code = lambda s: nbf.v4.new_code_cell(s)

ROOT = Path(r'e:\Academic & Research\Msc in Quantitative Management\Research Codes')

# ---------------------------------------------------------------------------
cells.append(md(
    "# Data Cleaning — Equity Portfolio VaR Research\n"
    "**Student:** Mathonsi Mphikeleli Mbongiseni (28574249) · UNISA MCom Quantitative Management"
))

# 1. Setup
cells.append(md("## 1. Setup"))
cells.append(code("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import yfinance as yf
import warnings
from scipy import stats
from scipy.stats import norm as sp_norm
from pathlib import Path

warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.6f}'.format)

ROOT        = Path(r'e:\\Academic & Research\\Msc in Quantitative Management\\Research Codes')
DATA_DIR    = ROOT / 'data'
FIGURES_DIR = ROOT / 'figures'

plt.rcParams.update({'figure.dpi': 120, 'axes.spines.top': False, 'axes.spines.right': False})

TICKERS = ['MSFT', 'AAPL', 'NVDA', 'IBM', 'CSCO', 'JPM', 'BAC', 'C', '^GSPC', '^IXIC']
NAMES   = {
    'MSFT': 'Microsoft', 'AAPL': 'Apple', 'NVDA': 'NVIDIA', 'IBM': 'IBM',
    'CSCO': 'Cisco', 'JPM': 'JPMorgan', 'BAC': 'BofA', 'C': 'Citigroup',
    '^GSPC': 'S&P 500', '^IXIC': 'NASDAQ',
}
START, END   = '2010-01-01', '2025-12-31'
PALETTE      = sns.color_palette('tab10', n_colors=len(TICKERS))
"""))

# 2. Download
cells.append(md("## 2. Raw Data"))
cells.append(code("""raw = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)
prices = raw['Close'].copy() if isinstance(raw.columns, pd.MultiIndex) else raw[['Close']].copy()
prices.index = pd.to_datetime(prices.index)
prices.index.name = 'Date'

print(f"{prices.shape[0]} trading days, {prices.shape[1]} assets")
print(f"{prices.index.min().date()} to {prices.index.max().date()}")
prices.head(3)
"""))

# 3. Index checks
cells.append(md("## 3. Index & Duplicate Checks"))
cells.append(code("""print("Monotonic index  :", prices.index.is_monotonic_increasing)
print("Duplicate dates  :", prices.index.duplicated().sum())
print("Weekend rows     :", (prices.index.dayofweek >= 5).sum())

diffs     = pd.Series(prices.index).diff().dropna()
big_gaps  = diffs[diffs > pd.Timedelta(days=5)]
print(f"Gaps > 5 days    : {len(big_gaps)}")
for end_date, gap in zip(prices.index[big_gaps.index], big_gaps.values):
    print(f"  before {end_date.date()} — {gap.days} day gap")
"""))

# 4. Missing values
cells.append(md("## 4. Missing Values"))
cells.append(code("""miss = pd.DataFrame({
    'count':      prices.isnull().sum(),
    'pct':        (prices.isnull().mean() * 100).round(4),
    'first_valid': [prices[c].first_valid_index() for c in prices.columns],
    'last_valid':  [prices[c].last_valid_index()  for c in prices.columns],
}).rename(index=NAMES)
print(miss.to_string())
"""))

cells.append(code("""fig, ax = plt.subplots(figsize=(14, 3))
sns.heatmap(prices.isnull().astype(int).rename(columns=NAMES).T,
            cbar=False, cmap='Reds', ax=ax)
ax.set_title('Missing value map  (red = NaN)')
ax.set_xlabel('Date index position')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'missing_value_map.png', bbox_inches='tight')
plt.show()
"""))

cells.append(code("""for ticker in prices.columns:
    s = prices[ticker]
    if not s.isnull().any():
        continue
    is_nan  = s.isnull()
    changes = is_nan.astype(int).diff().fillna(0)
    starts  = prices.index[changes == 1].tolist()
    ends    = prices.index[changes == -1].tolist()
    if is_nan.iloc[0]:  starts = [prices.index[0]] + starts
    if is_nan.iloc[-1]: ends   = ends + [prices.index[-1]]
    print(f"{NAMES[ticker]:15s}: {len(starts)} NaN block(s)")
    for s_, e_ in zip(starts, ends):
        print(f"  {s_.date()} to {e_.date()}  ({(e_ - s_).days + 1}d)")
"""))

# 5. Sanity checks
cells.append(md("## 5. Price Sanity Checks"))
cells.append(code("""# zero or negative prices
z_neg = (prices <= 0).sum().rename(NAMES)
print("Zero/negative prices:"); print(z_neg.to_string())

# extreme single-day moves
chg = prices.pct_change()
print("\\nSingle-day moves > 40%:")
found = False
for t in prices.columns:
    ex = chg[t][chg[t].abs() > 0.40].dropna()
    if len(ex):
        found = True
        print(f"  {NAMES[t]:15s}: {len(ex)} events")
        for d, v in ex.items():
            print(f"    {d.date()}  {v*100:+.2f}%")
if not found:
    print("  None found above threshold.")
"""))

# 6. Outlier detection
cells.append(md("## 6. Outliers in Log Returns"))
cells.append(code("""ret_raw = np.log(prices / prices.shift(1)).dropna()

Z_THR, MAD_THR = 4.0, 3.5
rows = []
for t in ret_raw.columns:
    r  = ret_raw[t].dropna()
    zs = np.abs(stats.zscore(r))
    med, mad = r.median(), np.median(np.abs(r - r.median()))
    mz = 0.6745 * (r - med) / mad if mad > 0 else pd.Series(0, index=r.index)
    rows.append({
        'Asset':       NAMES[t],
        'N':           len(r),
        'Z |>4|':      int((zs > Z_THR).sum()),
        'MAD |>3.5|':  int((np.abs(mz) > MAD_THR).sum()),
        'Max |ret| %': round(r.abs().max() * 100, 3),
        'Date':        r.abs().idxmax().date(),
    })

out_df = pd.DataFrame(rows).set_index('Asset')
print(out_df.to_string())
"""))

cells.append(code("""CRISIS = {'2020-03': 'COVID crash', '2020-02': 'COVID onset',
          '2008': 'GFC', '2009': 'GFC', '2010-05-06': 'Flash Crash'}

for t in ret_raw.columns:
    r  = ret_raw[t].dropna()
    zs = np.abs(stats.zscore(r))
    outs = r[zs > Z_THR].sort_values(key=abs, ascending=False)
    if outs.empty: continue
    print(f"\\n{NAMES[t]} ({t}): {len(outs)} outlier(s)")
    for dt, val in outs.items():
        tag = next((f'  [{lbl}]' for key, lbl in CRISIS.items() if key in str(dt)), '')
        print(f"  {dt.date()}  {val*100:+.4f}%{tag}")
"""))

cells.append(code("""focus = ['MSFT', 'NVDA', '^GSPC']
fig, axes = plt.subplots(3, 1, figsize=(14, 9))
for ax, t in zip(axes, focus):
    r  = ret_raw[t].dropna()
    zs = np.abs(stats.zscore(r))
    outs = r[zs > Z_THR]
    ax.plot(r.index, r * 100, linewidth=0.55, alpha=0.8, color='steelblue')
    if not outs.empty:
        ax.scatter(outs.index, outs * 100, color='firebrick', s=25, zorder=5,
                   label=f'|z|>{Z_THR:.0f} (n={len(outs)})')
    ax.axhline(0, color='grey', linewidth=0.4)
    ax.set_title(NAMES[t])
    ax.set_ylabel('Log return (%)')
    ax.legend(frameon=False, fontsize=8)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', rotation=30)
plt.suptitle('Outlier flags in log returns', y=1.01)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'outlier_flags.png', bbox_inches='tight')
plt.show()
"""))

# 7. Winsorisation
cells.append(md("## 7. Outlier Treatment"))
cells.append(code("""# extreme events are genuine (COVID, GFC) — retain raw returns for GARCH
# winsorise at 0.5/99.5 for LSTM feature set only
def winsorise(s, lo=0.005, hi=0.995):
    return s.clip(lower=s.quantile(lo), upper=s.quantile(hi))

ret_wins = ret_raw.apply(winsorise)

for t in ret_raw.columns:
    n = (ret_raw[t] != ret_wins[t]).sum()
    print(f"{NAMES[t]:15s}: {n} observations clipped")
"""))

# 8. Imputation + clean
cells.append(md("## 8. Missing Value Treatment & Final Clean"))
cells.append(code("""prices_clean = prices.ffill(limit=2).bfill().dropna(how='any')

print(f"Before: {len(prices)} rows")
print(f"After : {len(prices_clean)} rows  (dropped: {len(prices) - len(prices_clean)})")
print(f"Period: {prices_clean.index.min().date()} to {prices_clean.index.max().date()}")
print(f"NaN remaining: {prices_clean.isnull().sum().sum()}")
"""))

# 9. Validation
cells.append(md("## 9. Post-Clean Validation"))
cells.append(code("""checks = pd.DataFrame({
    NAMES[t]: {
        'N obs':         len(prices_clean[t]),
        'NaN':           prices_clean[t].isnull().sum(),
        'Min price':     round(prices_clean[t].min(), 2),
        'Max price':     round(prices_clean[t].max(), 2),
        'Zero/neg':      (prices_clean[t] <= 0).any(),
        'Dup dates':     prices_clean.index.duplicated().sum(),
    }
    for t in prices_clean.columns
}).T
print(checks.to_string())
print(f"\\nAll clean: {checks['NaN'].eq(0).all() and not checks['Zero/neg'].any()}")
"""))

# 10. Compute clean returns
cells.append(md("## 10. Clean Log Returns"))
cells.append(code("""ret_clean = np.log(prices_clean / prices_clean.shift(1)).dropna()
pct_clean = prices_clean.pct_change().dropna()

print(f"Shape: {ret_clean.shape}")
print(ret_clean.describe().rename(columns=NAMES).round(6).to_string())
"""))

# 11. Before / after plot
cells.append(code("""fig, axes = plt.subplots(2, 1, figsize=(14, 8))

for i, t in enumerate(prices.columns):
    fv = prices[t].first_valid_index()
    if fv: axes[0].plot(prices[t] / prices[t][fv], color=PALETTE[i], lw=0.7, alpha=0.8)
axes[0].set_title('Normalised prices — raw')
axes[0].set_ylabel('Normalised (base = 1)')

for i, t in enumerate(prices_clean.columns):
    axes[1].plot(prices_clean[t] / prices_clean[t].iloc[0],
                 color=PALETTE[i], lw=0.7, alpha=0.8, label=NAMES[t])
axes[1].set_title('Normalised prices — clean')
axes[1].set_ylabel('Normalised (base = 1)')
axes[1].legend(ncol=4, fontsize=7, frameon=False)
axes[1].xaxis.set_major_locator(mdates.YearLocator(2))
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[1].tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'before_after_cleaning.png', bbox_inches='tight')
plt.show()
"""))

cells.append(code("""fig, axes = plt.subplots(2, 2, figsize=(14, 7))
axes = axes.flatten()
pairs = [('NVDA', 'Raw'), ('NVDA', 'Winsorised'), ('^GSPC', 'Raw'), ('^GSPC', 'Winsorised')]
src   = {'Raw': ret_raw, 'Winsorised': ret_wins}

for ax, (t, label) in zip(axes, pairs):
    r = src[label][t].dropna()
    ax.hist(r * 100, bins=80, density=True, alpha=0.65, color=PALETTE[TICKERS.index(t)])
    x = np.linspace(r.min() * 100, r.max() * 100, 300)
    ax.plot(x, sp_norm.pdf(x, r.mean() * 100, r.std() * 100), 'k--', lw=1.1)
    ax.set_title(f'{NAMES[t]} — {label}  (kurt={float(r.kurtosis()):.2f})')
    ax.set_xlabel('Log return (%)')
plt.suptitle('Raw vs winsorised return distributions', y=1.01)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'distribution_raw_vs_wins.png', bbox_inches='tight')
plt.show()
"""))

# 12. Export
cells.append(md("## 11. Export"))
cells.append(code("""prices_clean.to_csv(DATA_DIR / 'prices_clean.csv')
ret_clean.to_csv(DATA_DIR / 'log_returns_clean.csv')
ret_wins.to_csv(DATA_DIR / 'log_returns_winsorised.csv')

for f in ['prices_clean.csv', 'log_returns_clean.csv', 'log_returns_winsorised.csv']:
    kb = (DATA_DIR / f).stat().st_size / 1024
    print(f"{f:40s}  {kb:6.1f} KB")
"""))

nb['cells'] = cells

out = ROOT / 'notebooks' / 'Data_Cleaning_VaR_Research.ipynb'
nbf.write(nb, str(out))
print(f"Written: {out}  ({len(cells)} cells)")
