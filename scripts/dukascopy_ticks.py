"""
Download historical tick data from Dukascopy and save it as CSV.

Usage:
    python dukascopy_ticks.py EURUSD 2024-01-02 2024-01-05 -o eurusd_ticks.csv
    python dukascopy_ticks.py EURUSD 2024-01-02 2024-01-05 --format mt5 -o EURUSD_ticks.csv

Output formats:
    mt5 (default) - MetaTrader 5 tick import format (Symbols -> Ticks -> Import):
                    <DATE>\t<TIME>\t<BID>\t<ASK>\t<LAST>\t<VOLUME>\t<FLAGS>
    raw           - Plain CSV with UTC timestamp, ask, bid, ask_volume, bid_volume

Requirements:
    pip install requests pandas
"""

import argparse
import lzma
import struct
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://datafeed.dukascopy.com/datafeed/{symbol}/{year:04d}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"

# Each tick record is 20 bytes, big-endian:
#   uint32 ms offset from hour start, uint32 ask, uint32 bid, float32 ask volume, float32 bid volume
TICK_STRUCT = struct.Struct(">IIIff")

# Prices are stored as integers; the divisor depends on the instrument's decimal places.
PRICE_DIVISORS = {
    # JPY pairs and a few others use 3 decimals
    "USDJPY": 1_000, "EURJPY": 1_000, "GBPJPY": 1_000, "AUDJPY": 1_000,
    "CADJPY": 1_000, "CHFJPY": 1_000, "NZDJPY": 1_000,
    # Metals / indices / crypto
    "XAUUSD": 1_000, "XAGUSD": 1_000,
    "BTCUSD": 10, "ETHUSD": 10,
    "USA500IDXUSD": 1_000, "USA30IDXUSD": 1_000, "USATECHIDXUSD": 1_000,
    "DEUIDXEUR": 1_000, "GBRIDXGBP": 1_000, "JPNIDXJPY": 1_000,
    "BRENTCMDUSD": 1_000, "LIGHTCMDUSD": 1_000,
}
DEFAULT_DIVISOR = 100_000  # 5-decimal FX pairs (EURUSD, GBPUSD, ...)

# MetaTrader 5 tick flags (ENUM_TICK_FLAGS)
TICK_FLAG_BID = 2
TICK_FLAG_ASK = 4


def build_url(symbol: str, dt: datetime) -> str:
    # Dukascopy months are zero-based (January = 00)
    return BASE_URL.format(
        symbol=symbol, year=dt.year, month=dt.month - 1, day=dt.day, hour=dt.hour
    )


def fetch_hour(session: requests.Session, symbol: str, dt: datetime, retries: int = 3) -> bytes | None:
    url = build_url(symbol, dt)
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 404:
                return None  # no data for this hour (weekend / holiday)
            resp.raise_for_status()
            return resp.content if resp.content else None
        except requests.RequestException as exc:
            if attempt == retries:
                print(f"  ! failed {url}: {exc}", file=sys.stderr)
                return None
            time.sleep(1.5 * attempt)
    return None


def decode_ticks(raw: bytes, hour_start: datetime, divisor: int) -> list[dict]:
    try:
        data = lzma.decompress(raw)
    except lzma.LZMAError:
        return []

    ticks = []
    for ms, ask, bid, ask_vol, bid_vol in TICK_STRUCT.iter_unpack(data):
        ticks.append(
            {
                "timestamp": hour_start + timedelta(milliseconds=ms),
                "ask": ask / divisor,
                "bid": bid / divisor,
                "ask_volume": round(ask_vol, 4),
                "bid_volume": round(bid_vol, 4),
            }
        )
    return ticks


def iter_hours(start: datetime, end: datetime):
    current = start.replace(minute=0, second=0, microsecond=0)
    while current < end:
        yield current
        current += timedelta(hours=1)


def download(symbol: str, start: datetime, end: datetime, delay: float = 0.2) -> pd.DataFrame:
    symbol = symbol.upper()
    divisor = PRICE_DIVISORS.get(symbol, DEFAULT_DIVISOR)
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (dukascopy-tick-downloader)"

    all_ticks: list[dict] = []
    hours = list(iter_hours(start, end))
    print(f"Downloading {symbol}: {len(hours)} hourly files from {start:%Y-%m-%d %H:00} to {end:%Y-%m-%d %H:00} UTC")

    for i, hour in enumerate(hours, 1):
        raw = fetch_hour(session, symbol, hour)
        if raw:
            ticks = decode_ticks(raw, hour, divisor)
            all_ticks.extend(ticks)
            print(f"  [{i}/{len(hours)}] {hour:%Y-%m-%d %H:00} -> {len(ticks):>6} ticks")
        else:
            print(f"  [{i}/{len(hours)}] {hour:%Y-%m-%d %H:00} -> no data")
        time.sleep(delay)

    if not all_ticks:
        return pd.DataFrame(columns=["timestamp", "ask", "bid", "ask_volume", "bid_volume"])

    df = pd.DataFrame(all_ticks)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def price_digits(symbol: str) -> int:
    divisor = PRICE_DIVISORS.get(symbol.upper(), DEFAULT_DIVISOR)
    return len(str(divisor)) - 1


def to_mt5(df: pd.DataFrame, symbol: str, tz_shift_hours: int = 0) -> pd.DataFrame:
    """
    Convert raw ticks into the MetaTrader 5 tick import layout.

    MT5 expects a tab-separated file with the header
        <DATE> <TIME> <BID> <ASK> <LAST> <VOLUME> <FLAGS>
    where DATE is YYYY.MM.DD and TIME is HH:MM:SS.mmm.
    FLAGS tells MT5 which fields changed in each tick (2 = bid, 4 = ask).
    """
    if df.empty:
        return pd.DataFrame(columns=["<DATE>", "<TIME>", "<BID>", "<ASK>", "<LAST>", "<VOLUME>", "<FLAGS>"])

    ts = df["timestamp"]
    if tz_shift_hours:
        ts = ts + pd.Timedelta(hours=tz_shift_hours)
    ts = ts.dt.tz_localize(None)

    digits = price_digits(symbol)
    bid = df["bid"].round(digits)
    ask = df["ask"].round(digits)

    bid_changed = bid.ne(bid.shift())
    ask_changed = ask.ne(ask.shift())
    flags = bid_changed.astype(int) * TICK_FLAG_BID + ask_changed.astype(int) * TICK_FLAG_ASK
    # The very first tick has no predecessor: mark both prices as fresh
    flags.iloc[0] = TICK_FLAG_BID | TICK_FLAG_ASK
    flags = flags.replace(0, TICK_FLAG_BID | TICK_FLAG_ASK)

    fmt = f"{{:.{digits}f}}".format
    return pd.DataFrame(
        {
            "<DATE>": ts.dt.strftime("%Y.%m.%d"),
            "<TIME>": ts.dt.strftime("%H:%M:%S.%f").str[:-3],
            "<BID>": bid.map(fmt),
            "<ASK>": ask.map(fmt),
            "<LAST>": "0",
            "<VOLUME>": "0",
            "<FLAGS>": flags.astype(int),
        }
    )


def parse_date(value: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(f"Invalid date: {value} (use YYYY-MM-DD or 'YYYY-MM-DD HH:MM')")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Dukascopy historical tick data.")
    parser.add_argument("symbol", help="Instrument, e.g. EURUSD, XAUUSD, USDJPY")
    parser.add_argument("start", type=parse_date, help="Start date (UTC), e.g. 2024-01-02")
    parser.add_argument("end", type=parse_date, help="End date (UTC, exclusive), e.g. 2024-01-05")
    parser.add_argument("-o", "--output", type=Path, help="Output CSV path (default: <SYMBOL>_<start>_<end>.csv)")
    parser.add_argument("--delay", type=float, default=0.2, help="Seconds to wait between requests")
    parser.add_argument(
        "--format", choices=("mt5", "raw"), default="mt5",
        help="Output layout: 'mt5' for MetaTrader 5 tick import (default), 'raw' for plain UTC CSV",
    )
    parser.add_argument(
        "--tz-shift", type=int, default=0,
        help="Hours to add to UTC timestamps so they match your broker's server time (mt5 format only), e.g. 2 or 3",
    )
    args = parser.parse_args()

    if args.end <= args.start:
        parser.error("end must be after start")

    df = download(args.symbol, args.start, args.end, args.delay)

    output = args.output or Path(f"{args.symbol.upper()}_{args.start:%Y%m%d}_{args.end:%Y%m%d}.csv")

    if args.format == "mt5":
        mt5_df = to_mt5(df, args.symbol, args.tz_shift)
        mt5_df.to_csv(output, index=False, sep="\t", lineterminator="\n")
    else:
        df.to_csv(output, index=False)

    print(f"\nSaved {len(df):,} ticks to {output} ({args.format} format)")


if __name__ == "__main__":
    main()
