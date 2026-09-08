# -*- coding: utf-8 -*-
"""market-risk-lab 공통 설정.

v0 = 기존 market-brief 시스템(reference/market_dashboard_v0.py)의 규칙을 그대로 동결한 것.
v0 상수는 절대 수정하지 않는다 — 영구 벤치마크다. 새 실험은 별도 이름으로 추가한다.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"
RESULTS_DIR = ROOT / "results"

# ------------------------------------------------------------------
# v0 동결 상수 (reference/market_dashboard_v0.py CONFIG/WEIGHTS/SC/DAILY_ONLY 와 동일해야 함;
# tests/test_parity.py 가 동일성을 검사한다)
# ------------------------------------------------------------------
V0 = {
    "fang": ["META", "AMZN", "NFLX", "GOOGL", "AAPL", "MSFT", "NVDA", "TSLA"],
    "watchlist": ["KORU", "SOXL", "SOXX", "AMD", "URNM", "URA", "SPCX", "TSLA",
                  "META", "NVDU", "QQQ", "NVDA", "VLO", "IBIT", "AMZN", "PLTR",
                  "MSFT", "SPXL", "XLE", "GOOGL", "PHO", "UDOW", "KO", "NFLX",
                  "BLK", "FAS", "AAPL", "AAL"],
    "period": "2y",              # 라이브 시스템의 SPY/VIX/BTC/FANG 조회 창 (재현: (asof-2y, asof] 달력 창 500~507거래일 — signals_v0.WINDOW_SPAN)
    "watch_period": "1y",        # 워치리스트 조회 창 (재현: (asof-1y, asof] 달력 창 ≈252거래일)
    "ma_window": 180,
    "lookback": {"daily": 5, "weekly": 4, "monthly": 3},
    "turn_k": {"daily": 3, "weekly": 2, "monthly": 2},
    "eod_ret": 0.001,
    "trend_cut": {"daily": 3, "weekly": 5, "monthly": 21},
    "trend_eps": 0.08,
    "band": 0.45,
}
V0_WEIGHTS = {"fang": 1.2, "macd": 1.5, "fg": 1.0, "vix": 0.8, "ma": 1.2,
              "eod": 1.0, "lead": 0.8, "btc": 1.0}
V0_SC = {"GREEN": 1.0, "R2G": 0.5, "AMBER": 0.0, "G2R": -0.5, "RED": -1.0}
V0_DAILY_ONLY = ("fg", "ma", "eod", "lead")
V0_SIGNALS = ("fang", "macd", "fg", "vix", "ma", "eod", "lead", "btc")
TONES = ("buy", "hold", "neutral", "caution", "reduce")
# 평가용 노출 맵 (VALIDATION.md 사전 등록): 톤 → 주식 비중
TONE_EXPOSURE = {"buy": 1.0, "hold": 1.0, "neutral": 1.0, "caution": 0.5, "reduce": 0.25}

# ------------------------------------------------------------------
# 데이터 캐시 대상 티커 (전부 yfinance, period="max", auto_adjust=True, 일괄 다운로드)
# ------------------------------------------------------------------
TICKERS = {
    "core": ["SPY", "^VIX", "BTC-USD"],
    "fang": V0["fang"],
    "watchlist": V0["watchlist"],
    # Phase 2 입력 (지금부터 캐시해 두면 백테스트 이력이 쌓인다)
    "breadth": ["RSP", "IWM", "^RUT", "^GSPC", "^SOX", "SMH", "QQQ",
                "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLY", "XLB"],
    "credit": ["HYG", "IEF", "LQD", "TLT", "JNK"],
    "rates": ["^TNX", "^IRX", "^FVX", "^TYX"],
    "macro": ["DX-Y.NYB", "UUP", "GC=F", "CL=F", "HG=F", "KRW=X"],
    "factors": ["MTUM", "SPHB", "SPLV"],
}
ALL_TICKERS = sorted({t for g in TICKERS.values() for t in g})
TWENTY_FOUR_SEVEN = {"BTC-USD"}          # 주말에도 봉이 생기는 자산 (SPY 기준일로 잘라내지 않음)

# CBOE 변동성 기간구조 (무료 CSV, 키 불필요)
CBOE_SERIES = {
    "VIX3M": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv",
    "VIX9D": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX9D_History.csv",
    "VVIX": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VVIX_History.csv",
    "SKEW": "https://cdn.cboe.com/api/global/us_indices/daily_prices/SKEW_History.csv",
}
# CNN Fear & Greed: 날짜를 붙인 graphdata 엔드포인트는 2020-08-03부터 이력 제공 (검증 2026-09-07)
CNN_FG_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{date}"
CNN_FG_SEED_DATE = "2020-08-01"
CNN_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://edition.cnn.com/markets/fear-and-greed",
    "Origin": "https://edition.cnn.com",
}

# ------------------------------------------------------------------
# 평가 목표변수 (VALIDATION.md 와 동일해야 함)
# ------------------------------------------------------------------
HORIZONS = (5, 20, 60)
DD_TARGETS = {"y_dd5_20": (0.05, 20), "y_dd10_60": (0.10, 60)}   # (낙폭 임계, 창 길이)
EPISODE_THRESHOLDS = (0.05, 0.10, 0.20)
BACKTEST_START = "2015-01-02"     # 6개 재현 가능 신호(P1·P2·P4·P5·P7·P9) 공통 구간 시작
HOLDOUT_START = "2024-09-01"      # 최종 검증 전까지 손대지 않는 구간 (Phase 2)

# 라이브 관련
ET = "America/New_York"
CT = "America/Chicago"
SITE_URL = "https://uncmac.github.io/market-risk-lab/"
