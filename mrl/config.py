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

# ------------------------------------------------------------------
# Phase 2 — 보정 모델 p2 / 결정층 v1 (ARCHITECTURE_PHASE2.md §3; v0 블록 불변)
# ------------------------------------------------------------------
P2 = {
    "dd": 0.05, "h": 20,                                  # = DD_TARGETS["y_dd5_20"]
    "features": ("x_vix", "x_har", "x_ma"),
    "har_lookbacks": (1, 5, 22), "har_weights": (1/3, 1/3, 1/3), "var_floor": 1e-8,   # (1bp 일변동)^2
    "ma_window": V0["ma_window"],                         # 180 — v0 와 같은 창(새 상수 없음)
    "vix_ffill_limit": 3,                                 # SPY 세션에 VIX 없으면 ≤3세션 ffill, 초과 시 NaN
    "C": 1.0, "purge": 20, "train_start": "1993-10-14",
    "first_refit": "2003-01-02", "first_refit_sensitivity": "1999-01-04",
    "first_refit_rule": {"min_rows": 2000, "min_episode_depth": 0.20},
    "boot_block": 40, "n_boot": 4000, "hac_lag": 19, "phase_offsets": 20,
    "reliability_bins": (0.0, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.40, 0.50, 1.0),
    "n_eff_div": 20, "param_band_refits": 5,
    "budget": 5, "param_count": 4,
    "eras": (("1993-01-29", "1999-12-31"), ("2000-01-01", "2007-12-31"), ("2008-01-01", "2012-12-31"),
             ("2013-01-01", "2019-12-31"), ("2020-01-01", "2024-08-30")),
    "har_train_start_sensitivity": "1996-01-02",
}
# 평가 블록 (달력 경계, [a, b) 반개구간; 세션은 SPY 인덱스로 결정) — §7
BLOCKS_24 = [("2003-01-01", "2005-01-01"), ("2005-01-01", "2007-01-01"), ("2007-01-01", "2009-01-01"),
             ("2009-01-01", "2011-01-01"), ("2011-01-01", "2013-01-01"), ("2013-01-01", "2015-01-01"),
             ("2015-01-01", "2017-01-01"), ("2017-01-01", "2019-01-01"), ("2019-01-01", "2021-01-01"),
             ("2021-01-01", "2023-01-01"), ("2023-01-01", HOLDOUT_START)]                      # 11개, 마지막 20개월
BLOCKS_18 = [("2003-01-01", "2004-07-01"), ("2004-07-01", "2006-01-01"), ("2006-01-01", "2007-07-01"),
             ("2007-07-01", "2009-01-01"), ("2009-01-01", "2010-07-01"), ("2010-07-01", "2012-01-01"),
             ("2012-01-01", "2013-07-01"), ("2013-07-01", "2015-01-01"), ("2015-01-01", "2016-07-01"),
             ("2016-07-01", "2018-01-01"), ("2018-01-01", "2019-09-01"), ("2019-09-01", "2021-05-01"),
             ("2021-05-01", "2023-01-01"), ("2023-01-01", HOLDOUT_START)]                      # 14개: 18개월×10 + 20개월×4
BLOCKS_24_FROM_1999 = [("1999-01-01", "2001-01-01"), ("2001-01-01", "2003-01-01")] + BLOCKS_24  # 13개(민감도)
DECISION_P2 = {"enter_caution": 1.5, "exit_caution": 1.2, "enter_reduce": 2.5, "exit_reduce": 2.0,
               "dwell": 5, "churn_alert": 12, "kpi_max_switches_per_year": 12}
DECISION_P2_SENSITIVITY = {"wide": {"enter_caution": 1.75, "exit_caution": 1.25, "enter_reduce": 3.0, "exit_reduce": 2.25},
                           "symmetric_dwell": {"dwell_escalate": 5}, "no_dwell": {"dwell": 0}}
P2_STATES = ("normal", "caution", "reduce")
STATE_TO_TONE = {"normal": "hold", "caution": "caution", "reduce": "reduce"}   # TONE_EXPOSURE 그대로 재사용
MODEL_P2_PATH = RESULTS_DIR / "model_p2.json"
HOLDOUT_UNLOCK_PATH = RESULTS_DIR / "holdout_unlock.json"

# ------------------------------------------------------------------
# Phase 3 — 비중 p3 / 그림자 등록부 / 킬룰 (ARCHITECTURE_PHASE3.md §3; v0·P2 블록 불변)
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Phase 3 — 비중 p3 / 그림자 등록부 / 킬룰 (ARCHITECTURE_PHASE3.md §3; v0·P2 블록 불변)
# ------------------------------------------------------------------
P3 = {
    "sigma_model": "ewma", "ewma_lambda": 0.94, "ewma_seed_sessions": 60,   # RiskMetrics 상수 — 적합 아님
    "sigma_sensitivity": "har_fc_20",                                        # 적합 HAR 예측은 표시·민감도 S-HAR 전용
    "k_slow": 3.5, "k_fast": 2.0,                                            # σ_T = D_max/k_slow (헤드라인) ; k_fast 는 낙관선 병기
    "vol_grid": (0.06, 0.08, 0.10, 0.12, 0.15),                              # σ_T 격자 (내림)
    "d_max_default": 0.35,                                                   # 기본 가족 예산 → σ_T 10%
    "w_min": 0.25, "w_max": 1.0, "grid": 0.05, "band": 0.10,                 # 바닥/상한/격자/무거래 밴드
    "floor_after_multiplier": True,                                          # 곱 뒤 [w_min, w_max] 재클립
    "cadence": "weekly", "escalate_immediately": True,                       # 주 마지막 세션 점검 + 결정층 격상 시 즉시 하향
    "cost_bps": 5, "cash_return": 0.0,
    "eval_start": "2003-01-02", "vol_only_start": "1993-10-14", "v0_start": BACKTEST_START,
    "reference_windows": (63, 126, 252, 756), "churn_ceiling": 24,          # 비중 변경 상한(252세션)
}
HMM_P3 = {
    "obs": ("r100", "ln_rv10"), "rv_window": 10, "k_states": 2, "cov": "full",
    "init_A": ((0.98, 0.02), (0.05, 0.95)), "tol": 1e-6, "max_iter_first": 300, "max_iter_refit": 100,
    "cov_jitter": 1e-6, "clip": 1e-4, "param_count": 12,                     # μ 2×2 + Σ 2×3 + A 2 (π = 정상분포, 자유도 0)
    "guard": {"p_min": 0.90, "p_max": 0.999, "gap_lnvol": 0.3, "occ_min": 0.05},
    "turn_h": 20, "platt_C": 1.0, "train_start": P2["train_start"], "purge": P2["purge"],
    "first_refit": P2["first_refit"], "tol_theta": 1e-7, "tol_prob": 1e-7,
}
ENSEMBLE_P3 = {
    "members": ("p2", "M1", "H"), "member_order": ("H", "M1"),               # 채택 검정 순서(고정, 재배열 금지)
    "fresh_block_months": 3, "fresh_blocks_min": 11, "admit_frac": 8 / 11, "reeval_every_blocks": 4,
    "disagree_flag": {"width": 0.20, "sessions": 5}, "K_s_cap": 12, "K_u_cap": 15,
}
KILL_P3 = {
    "min_months": 36, "min_episodes": 8, "max_months": 60, "reeval_months": 12,     # 2단계(§8.4)
    "block": P2["boot_block"], "n_boot": P2["n_boot"], "seed": 0, "ci": 0.95,
    "dd": 0.05, "confirm_sessions": 20, "target": "y_dd5_20", "reference": "p2_clim", "criterion": "point",
    "rearm": {"min_months": 12, "ci_lo_gt": 0.0},                            # 복귀: 새 장부 항목 + 전체 라이브 창 CI 하한 > 0
}
P3_DRIFT = {                                                                 # 임계 = 2003~24 참조분포 p5/p95 (post hoc; 라이브로 재조정 금지)
    "D1_p_level": {"window": 120, "lo": 0.04, "hi": 0.35},
    "D2_bss": {"red": {"window": 756, "lt": -0.05}, "yellow": {"window": 252, "lt": -0.10}},
    "D3_vol_fc": {"window": 60, "har_log_mae_gt": 0.50, "ewma_bias": (-0.35, 0.35)},
    "D4_vol_target": {"window": 60, "ratio_gt": 1.5}, "D4b_budget": True,
    "D5_stuck": {"window": 252, "warn_occ_gt": 0.75, "avg_w_lt": 0.30},
    "D6_churn": {"window": 252, "gt": 24},
    "D7_parity": {"sessions": 20, "dp_gt": 0.01},                             # exit 1
    "D8_coverage": {"min_n_eff": 12, "hit_lt": 0.5},
    "D9_hmm": {"window": 756, "auc_lt": 0.5},
    "D10_replay": {"dp_gt": 1e-6},                                            # 주간 재현(코드·자료 개정 탐지)
    "D11_feature_range": {"sessions": 5},                                     # x_vix·x_har·x_ma·x_hmm 가 1993~2024 범위 밖
}
SCENARIO_P3 = {"bins": P2["reliability_bins"], "min_n_eff": 20, "n_eff_div": P2["n_eff_div"], "episode_split": False,
               "h": P2["h"], "z": {"1s": 1.0, "80": 1.2816, "90": 1.645}, "dd_levels": (0.05, 0.10, 0.15, 0.20)}
P3_DEPLOY_MODES = ("info_only", "tones")                                     # p2 와 같은 두 값; 유효 모드 = 둘의 AND
MODEL_P3_PATH = RESULTS_DIR / "model_p3.json"
HMM_P3_PATH = RESULTS_DIR / "hmm_p3.json"
KILL_RECORD_PATH = RESULTS_DIR / "kill_record.json"
KILL_MANUAL_PATH = RESULTS_DIR / "kill_manual.json"
ALARMS_PATH = RESULTS_DIR / "alarms.csv"
TRACK_P3_PATH = RESULTS_DIR / "track_record_p3.json"
