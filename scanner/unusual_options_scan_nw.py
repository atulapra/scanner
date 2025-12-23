import yfinance as yf
import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------ USER SETTINGS ------------------
MAX_EXPIRIES_PER_TICKER = 8
LAST_PRICE_MAX = 1.5
VOL_MIN = 300
VOL_OI_MIN = 2.0

START_DATE_CUTOFF = "2024-12-29"
END_DATE_CUTOFF   = "2026-01-31"

HISTORY_LOOKBACK_DAYS = 14   # how far back to look for vol/OI trends
HISTORY_1W_DAYS       = 7    # "1 week" window


# ------------------ FILE NAMING ------------------
base_date = datetime.now().strftime('%Y-%m-%d')
prefix = f"unusual_options_scan_{base_date}"
ext = ".csv"

# Find next index for filename (ignore summary files)
existing_indices = []
for fname in os.listdir('.'):
    if fname.startswith(prefix) and fname.endswith(ext) and "summary" not in fname:
        try:
            num = int(fname[len(prefix)+1:-len(ext)])
            existing_indices.append(num)
        except ValueError:
            pass

next_index = max(existing_indices) + 1 if existing_indices else 1
SAVE_CSV = f"{prefix}_{next_index}{ext}"
SUMMARY_CSV = f"{prefix}_{next_index}_summary{ext}"

print(f"Next detailed file : {SAVE_CSV}")
print(f"Next summary file  : {SUMMARY_CSV}")


# ------------------ TICKERS ------------------
NASDAQ100 = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","COST","AFRM",
    "NFLX","PEP","ADBE","AMD","LIN","TMUS","CSCO","QCOM","TXN","AMAT",
    "INTU","HON","INTC","BKNG","SBUX","MU","AMGN","PDD","REGN","LRCX",
    "ADP","ISRG","ABNB","MDLZ","VRTX","ASML","GILD","ADI","PANW","KLAC",
    "PYPL","CRWD","CSX","WDAY","CHTR","MAR","NXPI","ROP","AEP","KDP",
    "MELI","FTNT","ORLY","SNPS","CDNS","MNST","CTAS","DXCM","PCAR","LULU",
    "MRVL","MCHP","ROST","EXC","ODFL","ADSK","ATVI","IDXX","EA",
    "PAYX","CTSH","TEAM","XEL","WDAY","DDOG","ZS","SPLK","BKR","ALGN",
    "AZN","CEG","VRSK","SIRI","PDD","LCID","RIVN","BIDU","JD","BMRN",
    "DOCU","VRSN","NTES","MRNA","ANSS","CSGP","CHKP","MTCH","CRWD","OKTA",
    "NEE","JNJ","SMCI","STZ","TMQ","PLTR","XYZ","HOOD","ORCL","UPST",
    "TSM","SHOP","SPOT","LLY","HIMS","UNH","DELL","COIN","OSCR","SNOW",
    "QUBT","RGTI","CRWV","RKLB","BA","QCOM","PANW","JPM","GS","BABA","BIDU",
    "USAR","ONON","VIX","OKLO","QS","CRML","MP","QBTS","JEF","GKOS","GSK","AMGN",
    "ROKU","RH","FCX","DASH","CHWY","CCJ","FI","TEAM","SBET","METC","AVAV",
    "MTSR","NTLA","ALAB","ALK","PINS","TEM","AZN","CE","WWW","TREX","LVS",
    "SNDK","BBAI","NNN","QURE","LENZ","A","SYM","KSS","EXEL","MDB", "CFLT", "MSTR", 
    "GEV", "SATS", "NVO", "ASTS", "WVE", "IONQ", "KTOS", "SERV", "IREN"
]

TICKERS = sorted(list(dict.fromkeys(NASDAQ100)))  # Deduplicate


# ------------------ HELPERS ------------------
def safe_option_chain(tkr, exp):
    """Return (calls, puts) for an expiry or (None, None) on failure."""
    try:
        oc = tkr.option_chain(exp)
        c = oc.calls.copy()
        p = oc.puts.copy()
        c["type"] = "CALL"
        p["type"] = "PUT"
        for df in (c, p):
            df["expiration"] = exp
        return c, p
    except Exception:
        return None, None


def pick_expiries(all_exps):
    """
    Filter expiries to be between START_DATE_CUTOFF and END_DATE_CUTOFF.
    Then pick the nearest MAX_EXPIRIES_PER_TICKER - 1, plus Jan-2026 if present.
    """
    low  = START_DATE_CUTOFF
    high = END_DATE_CUTOFF

    exps = [e for e in all_exps if low <= e <= high]
    exps_sorted = sorted(exps)

    # nearest expiries
    chosen = exps_sorted[:max(0, MAX_EXPIRIES_PER_TICKER - 1)]

    # include January 2026 expiry if exists
    jan26 = [e for e in exps_sorted if e.startswith("2026-01")]
    if jan26:
        pick = jan26[0]
        if pick not in chosen:
            chosen.append(pick)

    return chosen


def load_history_trends(today_str: str) -> pd.DataFrame:
    """
    Load past scanner CSVs and compute base stats for the last 1w and 2w
    per contractSymbol (avg volume & OI). We convert to relative changes
    later inside add_trend_columns.

    Returns a DataFrame with:
        contractSymbol, vol_1w_avg, vol_2w_avg, oi_1w_avg, oi_2w_avg
    """
    today_ts = pd.to_datetime(today_str).normalize()
    frames = []

    pattern = "unusual_options_scan_*.csv"
    for fname in glob.glob(pattern):
        # skip summary files
        if "summary" in fname:
            continue

        # expect: unusual_options_scan_YYYY-MM-DD_idx.csv
        try:
            basename = os.path.basename(fname)
            core = basename[len("unusual_options_scan_"):-4]  # strip prefix + ".csv"
            date_part = core.split("_")[0]                    # YYYY-MM-DD
            file_date = pd.to_datetime(date_part).normalize()
        except Exception:
            continue

        # only look at strictly earlier days, within the lookback window
        if file_date >= today_ts:
            continue
        days_ago = (today_ts - file_date).days
        if days_ago > HISTORY_LOOKBACK_DAYS:
            continue

        try:
            df = pd.read_csv(
                fname,
                usecols=["contractSymbol", "volume", "openInterest"]
            )
        except Exception:
            # if the file doesn't have those columns (e.g., older format), skip
            continue

        df["scan_date"] = file_date
        frames.append(df)

    if not frames:
        # No history – caller should handle empty df gracefully
        return pd.DataFrame(
            columns=[
                "contractSymbol",
                "vol_1w_avg", "vol_2w_avg",
                "oi_1w_avg", "oi_2w_avg",
            ]
        )

    hist = pd.concat(frames, ignore_index=True)
    hist["scan_date"] = pd.to_datetime(hist["scan_date"]).dt.normalize()
    hist["days_ago"] = (today_ts - hist["scan_date"]).dt.days

    # 1-week and 2-week subsets
    hist_1w = hist[hist["days_ago"] <= HISTORY_1W_DAYS]
    hist_2w = hist[hist["days_ago"] <= HISTORY_LOOKBACK_DAYS]

    # averages per contractSymbol
    g1 = hist_1w.groupby("contractSymbol").agg(
        vol_1w_avg=("volume", "mean"),
        oi_1w_avg=("openInterest", "mean"),
    )
    g2 = hist_2w.groupby("contractSymbol").agg(
        vol_2w_avg=("volume", "mean"),
        oi_2w_avg=("openInterest", "mean"),
    )

    trends = g1.join(g2, how="outer").reset_index()
    return trends


def add_trend_columns(df: pd.DataFrame, today_str: str) -> pd.DataFrame:
    """
    Join in 1w / 2w volume & OI increase metrics per contractSymbol.
    - vol_1w_inc = (today_vol / avg_vol_last_7d)  - 1
    - vol_2w_inc = (today_vol / avg_vol_last_14d) - 1
    - oi_1w_inc  = (today_oi  / avg_oi_last_7d)   - 1
    - oi_2w_inc  = (today_oi  / avg_oi_last_14d)  - 1
    """
    trends = load_history_trends(today_str)

    # If we have no history yet, just create empty columns and return
    if trends.empty:
        for col in ["vol_1w_inc", "vol_2w_inc", "oi_1w_inc", "oi_2w_inc"]:
            df[col] = np.nan
        return df

    df = df.merge(trends, on="contractSymbol", how="left")

    def rel_change(current, avg):
        current = float(current) if pd.notna(current) else np.nan
        avg = float(avg) if pd.notna(avg) and avg != 0 else np.nan
        if np.isnan(current) or np.isnan(avg):
            return np.nan
        return (current / avg) - 1.0

    df["vol_1w_inc"] = df.apply(
        lambda r: rel_change(r["volume"], r.get("vol_1w_avg", np.nan)), axis=1
    )
    df["vol_2w_inc"] = df.apply(
        lambda r: rel_change(r["volume"], r.get("vol_2w_avg", np.nan)), axis=1
    )
    df["oi_1w_inc"] = df.apply(
        lambda r: rel_change(r["openInterest"], r.get("oi_1w_avg", np.nan)), axis=1
    )
    df["oi_2w_inc"] = df.apply(
        lambda r: rel_change(r["openInterest"], r.get("oi_2w_avg", np.nan)), axis=1
    )

    # Drop intermediate avg columns so final CSV stays clean
    for col in ["vol_1w_avg", "vol_2w_avg", "oi_1w_avg", "oi_2w_avg"]:
        if col in df.columns:
            del df[col]

    return df


def scan_ticker(ticker):
    """Return per-contract hits for a single ticker (today)."""
    tkr = yf.Ticker(ticker)

    try:
        all_exps = tkr.options
    except Exception:
        return pd.DataFrame()

    if not all_exps:
        return pd.DataFrame()

    exps = pick_expiries(all_exps)
    rows = []

    for exp in exps:
        calls, puts = safe_option_chain(tkr, exp)
        if calls is None:
            continue

        df = pd.concat([calls, puts], ignore_index=True)

        # Ensure columns exist
        for col in ["lastPrice", "volume", "openInterest", "strike"]:
            if col not in df.columns:
                df[col] = 0

        df["vol_oi"] = df["volume"] / df["openInterest"].replace(0, 1)

        flt = (
            (df["lastPrice"] <= LAST_PRICE_MAX) &
            (df["volume"] >= VOL_MIN) &
            (df["vol_oi"] >= VOL_OI_MIN)
        )

        df = df.loc[
            flt,
            [
                "contractSymbol", "type", "strike", "lastPrice", "volume",
                "openInterest", "vol_oi", "expiration"
            ]
        ]
        if df.empty:
            continue

        df["ticker"] = ticker
        df["score"] = df["volume"] * df["vol_oi"]
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def build_ticker_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-ticker summary so it's human-readable:

    Ticker, call_strike_min, call_strike_max, call_volume_sum, call_oi_sum,
            put_strike_min,  put_strike_max,  put_volume_sum,  put_oi_sum,
            call_oi_2w_inc_avg, put_oi_2w_inc_avg,
            call_put_vol_ratio, call_put_oi_ratio,
            call_n_contracts,  put_n_contracts,
            total_volume, total_oi
    """
    # group by ticker + type (CALL/PUT)
    agg = (
        df.groupby(["ticker", "type"])
          .agg(
              strike_min=("strike", "min"),
              strike_max=("strike", "max"),
              volume_sum=("volume", "sum"),
              oi_sum=("openInterest", "sum"),
              oi_2w_inc_avg=("oi_2w_inc", "mean"),
              vol_2w_inc_avg=("vol_2w_inc", "mean"),
              n_contracts=("contractSymbol", "nunique"),
          )
          .reset_index()
    )

    # pivot types into columns
    pivoted = agg.pivot(index="ticker", columns="type")
    pivoted = pivoted.reset_index()

    # Flatten MultiIndex columns: ('strike_min','CALL') -> 'call_strike_min'
    flat_cols = []
    for col in pivoted.columns.to_flat_index():
        if isinstance(col, tuple):
            metric, opt_type = col
            if opt_type in ("CALL", "PUT"):
                flat_cols.append(f"{opt_type.lower()}_{metric}")
            else:
                flat_cols.append(str(metric))
        else:
            flat_cols.append(col)
    pivoted.columns = flat_cols

    # Ensure missing columns are present
    for col in [
        "call_strike_min", "call_strike_max", "call_volume_sum", "call_oi_sum",
        "put_strike_min",  "put_strike_max",  "put_volume_sum",  "put_oi_sum",
        "call_n_contracts","put_n_contracts",
        "call_oi_2w_inc_avg","put_oi_2w_inc_avg",
    ]:
        if col not in pivoted.columns:
            if col.endswith("_inc_avg"):
                pivoted[col] = np.nan
            else:
                pivoted[col] = 0.0

    # total flow + ratios
    pivoted["total_volume"] = pivoted["call_volume_sum"] + pivoted["put_volume_sum"]
    pivoted["total_oi"]     = pivoted["call_oi_sum"] + pivoted["put_oi_sum"]

    # Avoid division by zero in ratios
    pivoted["call_put_vol_ratio"] = np.where(
        pivoted["put_volume_sum"] > 0,
        pivoted["call_volume_sum"] / pivoted["put_volume_sum"],
        np.nan,
    )
    pivoted["call_put_oi_ratio"] = np.where(
        pivoted["put_oi_sum"] > 0,
        pivoted["call_oi_sum"] / pivoted["put_oi_sum"],
        np.nan,
    )

    # Sort: most active tickers first
    pivoted = pivoted.sort_values("total_volume", ascending=False)

    # Sort by call_put_vol_ratio (highest → lowest)
    pivoted = pivoted.sort_values(
        "call_put_vol_ratio",
        ascending=False,
        na_position="last"
    )

    # Reorder columns into a nice human-readable layout
    cols = [
        "ticker",
        "call_strike_min", "call_strike_max", "call_volume_sum", "call_oi_sum",
        "put_strike_min",  "put_strike_max",  "put_volume_sum",  "put_oi_sum",
        "call_oi_2w_inc_avg", "put_oi_2w_inc_avg",
        "call_put_vol_ratio", "call_put_oi_ratio",
        "call_n_contracts", "put_n_contracts",
        "total_volume", "total_oi",
    ]
    cols = [c for c in cols if c in pivoted.columns]
    pivoted = pivoted[cols]

    return pivoted


# ------------------ MAIN ------------------
def main():
    all_hits = []

    max_workers = min(20, len(TICKERS))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(scan_ticker, tk): tk for tk in TICKERS}

        for i, future in enumerate(as_completed(futures), 1):
            tk = futures[future]
            try:
                hits = future.result()
                if not hits.empty:
                    all_hits.append(hits)
                    print(f"[{i:3d}/{len(TICKERS)}] ✅ {tk} — found {len(hits)} matches")
                else:
                    print(f"[{i:3d}/{len(TICKERS)}] {tk} — no matches")
            except Exception as e:
                print(f"[{i:3d}/{len(TICKERS)}] ❌ {tk} error: {e}")

    if not all_hits:
        print("No matches found. Lower VOL_MIN or VOL_OI_MIN?")
        return

    df = pd.concat(all_hits, ignore_index=True)

    # Add 1w/2w volume/OI increase metrics (investor trend)
    df = add_trend_columns(df, base_date)

    # Save detailed contract-level CSV
    detailed_cols = [
        "ticker","type","strike","lastPrice","volume",
        "openInterest","vol_oi","expiration","contractSymbol","score",
        "vol_1w_inc","vol_2w_inc","oi_1w_inc","oi_2w_inc",
    ]
    detailed_cols = [c for c in detailed_cols if c in df.columns]
    df[detailed_cols].to_csv(SAVE_CSV, index=False)
    print(f"\nSaved {len(df)} detailed rows → {SAVE_CSV}")

    # Build per-ticker summary for human reading
    summary = build_ticker_summary(df)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved {len(summary)} ticker summaries → {SUMMARY_CSV}")

    # Print a small sample to console so you can quickly read direction
    print("\n=== Top 25 tickers by total volume (summary) ===")
    print(summary.head(25).to_string(index=False))

    # Optional: old-style top 10 by score
    per_ticker = (
        df.groupby("ticker")["score"]
          .max()
          .reset_index()
          .sort_values("score", ascending=False)
    )
    top10 = per_ticker["ticker"].head(10).tolist()
    print("\n=== Old-style Top 10 tickers by max score ===")
    for _, row in per_ticker[per_ticker["ticker"].isin(top10)].iterrows():
        print(f"{row['ticker']:>5}  score={row['score']:.1f}")


if __name__ == "__main__":
    main()
