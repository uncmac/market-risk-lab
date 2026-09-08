#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
market_dashboard.py — 매매 원칙 신호 대시보드 (yfinance 연동, v0.2)

사용법 (터미널):
    pip install yfinance pandas tzdata
    python market_dashboard.py                 # 실데이터 (yfinance + CNN F&G)
    python market_dashboard.py --demo          # 모의 데이터로 파이프라인 검증
    python market_dashboard.py --fg 9          # F&G 수동 입력 (API 실패 시)
    python market_dashboard.py -o out.html     # 출력 파일 지정

사용법 (Jupyter Notebook):
    from market_dashboard import run           # .py를 노트북과 같은 폴더에 두기
    run()                                      # 실데이터 → market_dashboard.html
    run(demo=True)                             # 모의 데이터
    run(show=True)                             # 노트북 셀 안에 결과 표시
    # 코드 전체를 셀에 붙여넣은 경우에도 그대로 실행됨 (커널 인자 -f 무시)

출력: market_dashboard.html (자체 완결 HTML, 브라우저에서 열기)

휴장일 처리:
    일간 봉은 거래일에만 존재하므로 "기준일 = 마지막 봉 날짜"가 자동으로
    직전 거래일이 된다. 예: 2026-07-26(일)에 실행하면 2026-07-24(금) 종가 기준.
    헤더에 기준일과 휴장 여부를 명시한다. BTC는 주말에도 거래되므로
    최신(주말 포함) 데이터를 그대로 쓴다 — P9(BTC 선행)의 취지와 일치.
"""
import argparse, json, sys, time, warnings
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# 설정 — 임계값·가중치·워치리스트는 여기서 수정 (백테스트로 보정할 것)
# ============================================================
CONFIG = {
    "fang": ["META", "AMZN", "NFLX", "GOOGL", "AAPL", "MSFT", "NVDA", "TSLA"],
    # 사용자 워치리스트 (SPY는 비교 기준이라 제외, SPX는 지수 심볼이라 제외)
    "watchlist": ["KORU", "SOXL", "SOXX", "AMD", "URNM", "URA", "SPCX", "TSLA",
                  "META", "NVDU", "QQQ", "NVDA", "VLO", "IBIT", "AMZN", "PLTR",
                  "MSFT", "SPXL", "XLE", "GOOGL", "PHO", "UDOW", "KO", "NFLX",
                  "BLK", "FAS", "AAPL", "AAL"],
    # 숏 비중 표시 대상 (티커, 한글명, 영문명). SpaceX는 비상장이라 숏 데이터가 없음
    # → SpaceX 지분을 보유한 상장 펀드 DXYZ(Destiny Tech100)로 대체 표시
    "short_watch": [("TSLA", "테슬라", "Tesla"), ("NVDA", "엔비디아", "NVIDIA"),
                    ("AMD", "AMD", "AMD"), ("MU", "마이크론", "Micron"),
                    ("DXYZ", "SpaceX 프록시", "SpaceX proxy")],
    "period": "2y",            # 일간 데이터 조회 기간
    "ma_window": 180,          # 180일선 (P5)
    "lookback": {"daily": 5, "weekly": 4, "monthly": 3},   # RS/모멘텀 봉수
    "turn_k": {"daily": 3, "weekly": 2, "monthly": 2},     # MACD 반전 확인 봉수
    "eod_ret": 0.001,          # 마감 30분 수익률 임계 (P6, +0.1%)
    "trend_cut": {"daily": 3, "weekly": 5, "monthly": 21}, # 종합점수 방향 비교 시점(일)
    "trend_eps": 0.08,
    "band": 0.45,              # 종합 판정 밴드
}
WEIGHTS = {"fang": 1.2, "macd": 1.5, "fg": 1.0, "vix": 0.8, "ma": 1.2,
           "eod": 1.0, "lead": 0.8, "btc": 1.0}
SC = {"GREEN": 1.0, "R2G": 0.5, "AMBER": 0.0, "G2R": -0.5, "RED": -1.0}
DAILY_ONLY = ("fg", "ma", "eod", "lead")  # 일간 전용(P3·P5·P6·P7): 주간·월간에서 제외
ST = {
    "GREEN": {"key": "GREEN", "label": "GREEN",       "color": "#22c55e", "desc": "상승 우호"},
    "R2G":   {"key": "R2G",   "label": "RED → GREEN", "color": "#2dd4bf", "desc": "바닥·개선 전환"},
    "AMBER": {"key": "AMBER", "label": "AMBER",       "color": "#eab308", "desc": "중립·관망"},
    "G2R":   {"key": "G2R",   "label": "GREEN → RED", "color": "#fb923c", "desc": "고점·악화 전환"},
    "RED":   {"key": "RED",   "label": "RED",         "color": "#ef4444", "desc": "하락 위험"},
}
# 미국 증시 휴장일 (NYSE, 관측일 기준). 매년 말 다음 해 날짜 추가 필요.
# 조기폐장일(반일장)은 포함하지 않음. 목록에 없는 해가 되면 휴장일이
# "데이터 지연"으로 표시될 뿐 동작은 유지된다.
US_MARKET_HOLIDAYS = {
    # 2026
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    # 2027
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
    "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
}
# 다국어: 대시보드는 한국어·영어 두 벌을 렌더링하고 CSS 토글로 전환한다.
LANGS = ("ko", "en")
DOW = {"ko": ["월", "화", "수", "목", "금", "토", "일"],
       "en": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]}
DOW_KR = DOW["ko"]  # 콘솔 로그용
TF_LABEL = {"ko": {"daily": "일간", "weekly": "주간", "monthly": "월간"},
            "en": {"daily": "Daily", "weekly": "Weekly", "monthly": "Monthly"}}


def fmt(v, d=1):
    return ("+" if v > 0 else "") + f"{v:.{d}f}"


def _news_ts(ts):
    """뉴스 타임스탬프를 'MM-DD HH:MM'으로. epoch·ISO·RSS 형식 모두 대응."""
    try:
        if ts is None:
            return ""
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(int(ts)).strftime("%m-%d %H:%M")
        s = str(ts)
        if "T" in s:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).strftime("%m-%d %H:%M")
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(s).strftime("%m-%d %H:%M")
    except Exception:
        return ""


def fetch_news(limit=5):
    """P8: 실시간 주요 시장 뉴스 Top N.
    1순위 yfinance(SPY 관련 뉴스), 실패·부족 시 Google News RSS 폴백."""
    items = []
    try:
        import yfinance as yf
        for n in (yf.Ticker("SPY").news or []):
            c = n.get("content", n)  # 신·구 스키마 모두 대응
            title = c.get("title") or n.get("title")
            if not title:
                continue
            cu, ctu = c.get("canonicalUrl"), c.get("clickThroughUrl")
            link = ((cu.get("url") if isinstance(cu, dict) else None)
                    or (ctu.get("url") if isinstance(ctu, dict) else None)
                    or c.get("link") or n.get("link") or "")
            pv = c.get("provider")
            pub = (pv.get("displayName") if isinstance(pv, dict) else None) or c.get("publisher") or n.get("publisher") or ""
            ts = c.get("pubDate") or n.get("providerPublishTime")
            items.append({"title": str(title), "link": str(link),
                          "publisher": str(pub), "ts": _news_ts(ts)})
            if len(items) >= limit:
                break
    except Exception as e:
        print(f"[경고] yfinance 뉴스 조회 실패: {e}")
    if len(items) < limit:
        try:
            import requests
            import xml.etree.ElementTree as ET
            r = requests.get(
                "https://news.google.com/rss/search?q=stock%20market&hl=en-US&gl=US&ceid=US:en",
                headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            for it in ET.fromstring(r.content).iter("item"):
                t = it.findtext("title")
                if not t or any(t == x["title"] for x in items):
                    continue
                sc = it.find("source")
                items.append({"title": t, "link": it.findtext("link") or "",
                              "publisher": sc.text if sc is not None else "",
                              "ts": _news_ts(it.findtext("pubDate"))})
                if len(items) >= limit:
                    break
        except Exception as e:
            print(f"[경고] Google News RSS 조회 실패: {e}")
    return items[:limit]


def _demo_news():
    return [
        {"title": "관세 협상 재개 기대에 뉴욕 증시 사흘째 반등 (모의)", "publisher": "DEMO", "ts": "07-24 16:10", "link": ""},
        {"title": "연준 금리 동결 시사, 국채금리 하락 (모의)", "publisher": "DEMO", "ts": "07-24 14:02", "link": ""},
        {"title": "반도체 수출 규제 완화 검토 보도 (모의)", "publisher": "DEMO", "ts": "07-24 11:45", "link": ""},
        {"title": "비트코인 주말 반등, 위험자산 선호 회복 (모의)", "publisher": "DEMO", "ts": "07-26 09:30", "link": ""},
        {"title": "대형 기술주 실적 발표 주간 개막 (모의)", "publisher": "DEMO", "ts": "07-26 08:00", "link": ""},
    ]


def now_eastern():
    """미 동부 현재 시각. Windows에 tzdata가 없으면 pytz → 로컬 시간 순으로 폴백."""
    try:
        return datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        try:
            import pytz
            return datetime.now(pytz.timezone("America/New_York"))
        except Exception:
            print("[경고] 시간대 DB 없음 → 로컬 시간으로 기준일 판정 (pip install tzdata 권장)")
            return datetime.now()


# ============================================================
# 캘린더 · 수급 — 옵션 만기일 / 윈도우 드레싱 / 숏 비중 (참고 정보, 점수 미반영)
# ============================================================
def _third_friday(y, m):
    """해당 월의 3번째 금요일(미국 월간 옵션 만기일)."""
    from datetime import date
    first = date(y, m, 1)
    return date(y, m, 1 + (4 - first.weekday()) % 7 + 14)


def market_calendar(today):
    """이번 달 옵션 만기일(OPEX)·다음 달 만기일과 월말 윈도우 드레싱 예상 구간.
    주말만 제외하고 미국 휴장일은 미반영(만기일이 휴장이면 실제로는 전일로 이동)."""
    from datetime import date

    def dinfo(d):
        return {"date": str(d), "dow": d.weekday()}  # 요일은 렌더링 때 언어별로 표기

    opex = _third_friday(today.year, today.month)
    y2, m2 = (today.year + 1, 1) if today.month == 12 else (today.year, today.month + 1)
    opex_next = _third_friday(y2, m2)
    # 윈도우 드레싱: 월말 마지막 4거래일 (기관 리밸런싱·성과 보고용 매매 집중 구간)
    mend = (pd.Timestamp(today) + pd.offsets.MonthEnd(0)).date()
    bd = pd.bdate_range(date(today.year, today.month, 1), mend)
    wd_s, wd_e = bd[-4].date(), bd[-1].date()
    return {
        "opex": {**dinfo(opex), "dday": (opex - today).days,
                 "triple": today.month in (3, 6, 9, 12), "passed": opex < today},
        "opex_next": {**dinfo(opex_next), "dday": (opex_next - today).days,
                      "triple": m2 in (3, 6, 9, 12)},
        "wd": {"start": dinfo(wd_s), "end": dinfo(wd_e), "dday": (wd_s - today).days,
               "active": wd_s <= today <= wd_e,
               "quarter": today.month in (3, 6, 9, 12)},
    }


def fetch_shorts():
    """숏 비중(NASDAQ/NYSE 집계, 월 2회 발표 → 약 2주 지연) — yfinance info 필드 사용."""
    import yfinance as yf
    out = []
    for t, name, name_en in CONFIG["short_watch"]:
        try:
            info = yf.Ticker(t).info
            spf = info.get("shortPercentOfFloat")
            ss, sp = info.get("sharesShort"), info.get("sharesShortPriorMonth")
            asof = info.get("dateShortInterest")
            out.append({
                "ticker": t, "name": name, "name_en": name_en,
                "pct": round(spf * 100, 1) if spf else None,          # 유동주식 대비 숏 %
                "dtc": info.get("shortRatio"),                        # 숏 커버 소요일
                "chg": round((ss / sp - 1) * 100, 1) if ss and sp else None,  # 숏 주식수 전월 대비 %
                "asof": datetime.fromtimestamp(int(asof)).strftime("%Y-%m-%d") if asof else "",
            })
        except Exception as e:
            print(f"[경고] 숏 비중 {t} 조회 실패 → 제외: {e}")
    return out


def _demo_shorts():
    return [
        {"ticker": "TSLA", "name": "테슬라", "name_en": "Tesla", "pct": 17.8, "dtc": 2.4, "chg": 6.3, "asof": "2026-07-15"},
        {"ticker": "NVDA", "name": "엔비디아", "name_en": "NVIDIA", "pct": 1.1, "dtc": 1.2, "chg": -3.1, "asof": "2026-07-15"},
        {"ticker": "AMD", "name": "AMD", "name_en": "AMD", "pct": 4.6, "dtc": 1.9, "chg": 2.2, "asof": "2026-07-15"},
        {"ticker": "MU", "name": "마이크론", "name_en": "Micron", "pct": 3.2, "dtc": 2.1, "chg": -1.4, "asof": "2026-07-15"},
        {"ticker": "DXYZ", "name": "SpaceX 프록시", "name_en": "SpaceX proxy", "pct": 8.9, "dtc": 3.5, "chg": 11.0, "asof": "2026-07-15"},
    ]


# ============================================================
# 데이터 계층
# ============================================================
def fetch_real(fg_manual=None):
    """yfinance + CNN F&G에서 실데이터 수집."""
    import yfinance as yf

    def hist(t, period=CONFIG["period"], interval="1d"):
        """일시적 조회 실패(레이트리밋 등)에 대비해 최대 3회 재시도."""
        last = None
        for attempt in range(3):
            try:
                df = yf.Ticker(t).history(period=period, interval=interval, auto_adjust=True)
                if not df.empty:
                    return df
                last = RuntimeError(f"{t} 데이터 없음")
            except Exception as e:
                last = e
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"{t} 조회 실패 (3회 시도): {last}")

    spy = hist("SPY")
    spy.index = spy.index.tz_localize(None)
    vix = hist("^VIX")["Close"]
    vix.index = vix.index.tz_localize(None)
    btc = hist("BTC-USD")["Close"]          # 주말 포함
    btc.index = btc.index.tz_localize(None)

    fang = {}
    for t in CONFIG["fang"]:
        try:
            s = hist(t)["Close"]
            s.index = s.index.tz_localize(None)
            fang[t] = s
        except Exception as e:
            print(f"[경고] FANG {t} 조회 실패 → 제외: {e}")  # 바스켓 평균이라 일부 누락 허용
    if not fang:
        raise RuntimeError("FANG 전 종목 조회 실패")
    fang = pd.DataFrame(fang).dropna()

    watch = {}
    for t in CONFIG["watchlist"]:
        try:
            s = hist(t, period="1y")["Close"]
            s.index = s.index.tz_localize(None)
            watch[t] = s
        except Exception as e:
            print(f"[경고] 워치리스트 {t} 조회 실패 → 제외: {e}")
    watch = pd.DataFrame(watch)

    # P6: 마감 30분 순매수 — 30분봉의 마지막 봉 수익률 (최근 ~30거래일)
    intr = hist("SPY", period="30d", interval="30m")
    last_bar = intr.groupby(intr.index.date).tail(1)
    ret30 = last_bar["Close"] / last_bar["Open"] - 1
    idx30 = pd.to_datetime([d.date() for d in last_bar.index])
    eod_bool = pd.Series((ret30 > CONFIG["eod_ret"]).values, index=idx30)
    eod_vals = pd.Series((ret30 * 100).values, index=idx30)  # 마감 30분 수익률(%)

    # P3: CNN Fear & Greed (비공식 엔드포인트, 실패 시 None)
    fg = None
    if fg_manual is not None:
        fg = {"score": fg_manual, "d_daily": 0, "d_weekly": 0, "d_monthly": 0, "hist": []}
    else:
        try:
            import requests
            # CNN은 단순 User-Agent를 차단하고 HTML/빈 응답을 주는 경우가 많음
            # → 실제 브라우저 수준 헤더로 요청하고, JSON이 아니면 즉시 원인 출력
            headers = {
                "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                               "AppleWebKit/537.36 (KHTML, like Gecko) "
                               "Chrome/126.0.0.0 Safari/537.36"),
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://edition.cnn.com/markets/fear-and-greed",
                "Origin": "https://edition.cnn.com",
            }
            r = requests.get(
                "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
                headers=headers, timeout=10)
            if r.status_code != 200 or not r.text.lstrip().startswith("{"):
                raise RuntimeError(f"HTTP {r.status_code}, 응답 앞부분 {r.text[:60]!r}")
            jall = r.json()
            j = jall["fear_and_greed"]
            sc = round(j["score"])
            hist_fg = [float(p["y"]) for p in
                       jall.get("fear_and_greed_historical", {}).get("data", [])][-180:]
            fg = {"score": sc,
                  "d_daily": round(j["score"] - j["previous_close"]),
                  "d_weekly": round(j["score"] - j["previous_1_week"]),
                  "d_monthly": round(j["score"] - j["previous_1_month"]),
                  "hist": hist_fg}
        except Exception as e:
            print(f"[경고] CNN F&G 조회 실패 → N/A 처리, run(fg=값) 또는 --fg 로 수동 입력: {e}")

    return {"spy": spy, "vix": vix, "btc": btc, "fang": fang,
            "watch": watch, "eod": eod_bool, "eod_vals": eod_vals, "fg": fg,
            "news": fetch_news(), "shorts": fetch_shorts(), "mode": "live"}


def make_demo():
    """모의 데이터: 급락 후 V-반전 초입(바닥 형성) 국면을 합성.
    기준일이 금요일(2026-07-24)로 끝나도록 만들어 휴장일 로직도 함께 검증."""
    rng = np.random.default_rng(7)
    end = pd.Timestamp("2026-07-24")
    days = pd.bdate_range(end=end, periods=420)

    # SPY: 완만한 상승 → 25일 급락(-18%) → 트로프(8거래일 전) → 반등
    n = len(days)
    trough_i = n - 5
    crash_i = trough_i - 25
    px = np.empty(n)
    px[0] = 520.0
    for i in range(1, n):
        if i < crash_i:
            mu = 0.0006
        elif i < trough_i:
            mu = -0.0079          # 급락 구간
        else:
            mu = 0.0115           # 날카로운 V 반등
        px[i] = px[i - 1] * (1 + mu + rng.normal(0, 0.004))
    close = pd.Series(px, index=days)
    op = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.002, n))
    hi = np.maximum(op, close) * (1 + abs(rng.normal(0, 0.003, n)))
    lo = np.minimum(op, close) * (1 - abs(rng.normal(0, 0.003, n)))
    spy = pd.DataFrame({"Open": op, "High": hi, "Low": lo, "Close": close}, index=days)

    # FANG: 급락은 더 깊게, 반등은 트로프 3일 전부터 더 강하게 (P1)
    fang = {}
    for k, t in enumerate(CONFIG["fang"]):
        f = np.empty(n); f[0] = 100.0
        for i in range(1, n):
            if i < crash_i:
                mu = 0.0008
            elif i < trough_i - 3:
                mu = -0.0118
            else:
                mu = 0.0215
            f[i] = f[i - 1] * (1 + mu + rng.normal(0, 0.006))
        fang[t] = pd.Series(f, index=days)
    fang = pd.DataFrame(fang)

    # VIX: 급락 때 45까지 치솟고 정점 후 하락 중 (P4)
    v = np.full(n, 15.0)
    for i in range(1, n):
        if crash_i <= i < trough_i:
            v[i] = min(46, v[i - 1] * 1.055)
        elif i >= trough_i:
            v[i] = max(38, v[i - 1] * 0.985)
        else:
            v[i] = 15 + rng.normal(0, 0.6)
    vix = pd.Series(v, index=days)

    # BTC: 주말 포함, 주식보다 5일 먼저 반등 (P9) — 오늘(일요일)까지 데이터 존재
    bdays = pd.date_range(days[0], pd.Timestamp("2026-07-26"), freq="D")
    b = np.empty(len(bdays)); b[0] = 60000.0
    btc_trough = bdays.get_indexer([days[trough_i - 5]], method="nearest")[0]
    for i in range(1, len(bdays)):
        if i < btc_trough - 30:
            mu = 0.0008
        elif i < btc_trough:
            mu = -0.010
        else:
            mu = 0.012
        b[i] = b[i - 1] * (1 + mu + rng.normal(0, 0.008))
    btc = pd.Series(b, index=bdays)

    # 워치리스트: 5종목은 방어+반등(주도주 후보), 나머지는 시장 추종
    watch = {}
    for k, t in enumerate(CONFIG["watchlist"]):
        strong = k < 5
        w = np.empty(n); w[0] = 100.0
        for i in range(1, n):
            if i < crash_i:
                mu = 0.0008
            elif i < trough_i:
                mu = -0.004 if strong else -0.009
            else:
                mu = 0.016 if strong else 0.009
            w[i] = w[i - 1] * (1 + mu + rng.normal(0, 0.005))
        watch[t] = pd.Series(w, index=days)
    watch = pd.DataFrame(watch)

    # 마감 30분 매수: 최근 4거래일 연속 (P6)
    eod = pd.Series(False, index=days[-30:])
    eod.iloc[-4:] = True
    ev = rng.normal(-0.02, 0.06, 30)
    ev[-4:] = [0.18, 0.22, 0.15, 0.27]
    eod_vals = pd.Series(ev, index=days[-30:])

    fgh = np.clip(np.r_[np.linspace(52, 14, 20), np.linspace(13, 6, 6),
                        np.linspace(7, 9, 4)] + rng.normal(0, 1.5, 30), 2, 95)
    fg = {"score": 9, "d_daily": 4, "d_weekly": -4, "d_monthly": -12,
          "hist": [float(x) for x in fgh]}
    return {"spy": spy, "vix": vix, "btc": btc, "fang": fang,
            "watch": watch, "eod": eod, "eod_vals": eod_vals, "fg": fg,
            "news": _demo_news(), "shorts": _demo_shorts(), "mode": "demo"}


# ============================================================
# 지표
# ============================================================
def resample_close(s, freq):
    """일간 종가 → 주간(금요일 마감)/월간 종가."""
    if freq == "weekly":
        return s.resample("W-FRI").last().dropna()
    if freq == "monthly":
        return s.resample("ME").last().dropna()
    return s


def macd_phase(close, turn_k):
    """MACD(12,26,9) 히스토그램 국면 분류 (P2)."""
    e12 = close.ewm(span=12, adjust=False).mean()
    e26 = close.ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    sig = macd.ewm(span=9, adjust=False).mean()
    hist = (macd - sig).dropna()
    d = hist.diff().dropna()
    up = dn = 0
    for x in d.iloc[::-1]:
        if x > 0 and dn == 0:
            up += 1
        elif x < 0 and up == 0:
            dn += 1
        else:
            break
    h0 = hist.iloc[-1]
    if h0 < 0 and up >= turn_k:
        return "turn", up, hist
    if h0 > 0 and dn >= turn_k:
        return "roll", dn, hist
    if h0 >= 0:
        return "up", up, hist
    return "down", dn, hist


def ret(s, k):
    """마지막 값의 k봉 수익률(%)."""
    if len(s) <= k:
        return 0.0
    return float((s.iloc[-1] / s.iloc[-1 - k] - 1) * 100)


# ============================================================
# 메트릭 빌드 — 타임프레임별, cut(최근 n일 제외)으로 과거 시점 재현
# ============================================================
def build_metrics(data, tf, cut=0):
    lb = CONFIG["lookback"][tf]
    spy_d = data["spy"]["Close"]
    if cut:
        spy_d = spy_d.iloc[:-cut]
    asof = spy_d.index[-1]

    fang_d = data["fang"].loc[:asof]
    vix_d = data["vix"].loc[:asof]
    btc_d = data["btc"]                       # BTC는 주말 포함 최신 유지
    if cut:
        btc_d = btc_d.loc[:asof]
    watch_d = data["watch"].loc[:asof]

    # 타임프레임 변환
    spy = resample_close(spy_d, tf)
    basket = resample_close(fang_d.div(fang_d.iloc[0]).mean(axis=1), tf)
    vix = resample_close(vix_d, tf)
    btc = resample_close(btc_d, tf)

    # P5: 180일선 (일간 기준, 전 타임프레임 공통)
    ma_ser = spy_d.rolling(CONFIG["ma_window"]).mean().dropna()
    ma_dist = float((spy_d.iloc[-1] / ma_ser.iloc[-1] - 1) * 100)
    ma_slope = float((ma_ser.iloc[-1] / ma_ser.iloc[-21] - 1) * 100) if len(ma_ser) > 21 else 0.0
    below = ma_dist <= 0

    # P1
    fang_rs = ret(basket, lb) - ret(spy, lb)
    # P2
    phase, streak, hist = macd_phase(spy, CONFIG["turn_k"][tf])
    # P4
    vix_now = float(vix.iloc[-1])
    vix_d_ = float(vix.iloc[-1] - vix.iloc[-1 - lb]) if len(vix) > lb else 0.0
    # P6 (일간 고유 신호 — 전 타임프레임 공통 표기)
    eod = data["eod"]
    if cut:
        eod = eod[eod.index <= asof]
    streak_eod = 0
    for x in eod.iloc[::-1]:
        if x:
            streak_eod += 1
        else:
            break
    eod20 = int(eod.iloc[-20:].sum())
    # P7 (일간 기준)
    leaders = []
    spy_hi = spy_d.rolling(60).max().iloc[-1]
    spy_dd = float(spy_d.iloc[-1] / spy_hi - 1)
    for t in watch_d.columns:
        s = watch_d[t].dropna()
        if len(s) < 65:          # 상장 초기 등 데이터 부족 종목 제외
            continue
        if below:
            dd = float(s.iloc[-1] / s.rolling(60).max().iloc[-1] - 1)
            if dd >= 0.6 * spy_dd and ret(s, 5) > 0:
                leaders.append(t)
        else:
            if ret(s, 20) - ret(spy_d, 20) > 3:
                leaders.append(t)
    # P9
    btc_mom = ret(btc, {"daily": 7, "weekly": 4, "monthly": 3}[tf])
    up_days = 0
    for i in range(len(btc_d) - 1, 0, -1):
        if btc_d.iloc[i] > btc_d.iloc[i - 1]:
            up_days += 1
        else:
            break
    btc_turned = btc_mom > 2 and up_days >= 2
    # P3
    fg = data["fg"]
    fg_score = fg["score"] if fg else None
    fg_delta = fg[f"d_{tf}"] if fg else 0

    return {
        "belowMA": below, "maDist": ma_dist, "maSlope": ma_slope,
        "fangRS": fang_rs, "macd": phase, "histStreak": streak,
        "fg": fg_score, "fgD": fg_delta,
        "vix": vix_now, "vixD": vix_d_,
        "eod": streak_eod, "eod20": eod20, "leaders": leaders,
        "btcMom": btc_mom, "btcTurned": btc_turned, "btcUpDays": up_days,
        "lb": lb,
    }


# ============================================================
# 신호 평가 — 프로토타입 로직 이식 (P1~P9)
# ============================================================
def assess(m):
    s = {}
    # P1 FANG 상대강도
    if m["belowMA"]:
        s["fang"] = "R2G" if m["fangRS"] >= 1.5 else "RED" if m["fangRS"] <= -1.5 else "AMBER"
    else:
        s["fang"] = "G2R" if m["fangRS"] <= -2 else "GREEN" if m["fangRS"] >= 0 else "AMBER"
    # P2 MACD
    s["macd"] = {"down": "RED", "turn": "R2G", "up": "GREEN", "roll": "G2R"}[m["macd"]]
    # P3 F&G
    if m["fg"] is None:
        s["fg"] = "AMBER"
    elif m["fg"] <= 10:
        s["fg"] = "R2G"
    elif m["fg"] <= 25:
        s["fg"] = "R2G" if m["fgD"] > 0 else "RED"
    elif m["fg"] >= 75:
        s["fg"] = "G2R" if m["fgD"] < 0 else "GREEN"
    elif m["fg"] >= 55:
        s["fg"] = "GREEN"
    else:
        s["fg"] = "AMBER"
    # P4 VIX
    if m["vix"] >= 40:
        s["vix"] = "R2G" if m["vixD"] <= 0 else "RED"
    elif m["vix"] >= 28:
        s["vix"] = "RED"
    elif m["vix"] <= 17:
        s["vix"] = "GREEN"
    elif m["vixD"] >= 4 and not m["belowMA"]:
        s["vix"] = "G2R"
    else:
        s["vix"] = "AMBER"
    # P5 180일선
    s["ma"] = "R2G" if m["maDist"] <= 0 else "AMBER" if m["maDist"] <= 3 else "GREEN"
    # P6 장마감 매수
    if m["belowMA"]:
        s["eod"] = "R2G" if m["eod"] >= 3 else "RED" if m["eod"] == 0 else "AMBER"
    else:
        s["eod"] = "GREEN" if m["eod"] >= 3 else "AMBER"
    # P7 주도주
    n = len(m["leaders"])
    if n >= 5:
        s["lead"] = "R2G" if m["belowMA"] else "GREEN"
    elif n <= 2:
        s["lead"] = "RED" if m["belowMA"] else "AMBER"
    else:
        s["lead"] = "AMBER"
    # P9 BTC
    if m["belowMA"]:
        s["btc"] = "R2G" if m["btcTurned"] else "RED"
    elif m["btcMom"] < -3:
        s["btc"] = "G2R"
    else:
        s["btc"] = "GREEN" if m["btcMom"] > 0 else "AMBER"
    return s


def composite(states, fg_missing):
    tot = w = 0.0
    for k, st in states.items():
        if k == "fg" and fg_missing:      # F&G 결측 시 가중치에서 제외
            continue
        tot += SC[st] * WEIGHTS[k]
        w += WEIGHTS[k]
    return tot / w


def overall(score, trend):
    b = CONFIG["band"]
    if score >= b:
        return "G2R" if trend < 0 else "GREEN"
    if score <= -b:
        return "R2G" if trend > 0 else "RED"
    return "R2G" if trend > 0 else "G2R" if trend < 0 else "AMBER"


TONE_COLORS = {"buy": "#22c55e", "hold": "#2dd4bf", "caution": "#eab308",
               "reduce": "#ef4444", "neutral": "#8a94a8"}


SIGNAL_NAMES = {
    "ko": {"fang": "FANG 상대강도", "macd": "V-바닥 / MACD 반전", "fg": "CNN Fear & Greed",
           "vix": "불확실성 정점", "ma": "SPY vs 180일선", "eod": "장마감 기관 매수",
           "lead": "차기 주도주 후보", "btc": "BTC 선행 지표"},
    "en": {"fang": "FANG Relative Strength", "macd": "V-Bottom / MACD Reversal", "fg": "CNN Fear & Greed",
           "vix": "Uncertainty Peak", "ma": "SPY vs 180-Day MA", "eod": "Late-Day Institutional Buying",
           "lead": "Next Leadership Candidates", "btc": "BTC Leading Indicator"},
}


def plain_digest(d, st, lang="ko"):
    """종합 판정 바로 아래에 붙는 '오늘의 신호 한 줄 요약'.
    개별 신호를 누구나 읽을 수 있는 평이한 말로 한 줄씩 풀어 쓴다."""
    out = []
    ko = lang == "ko"
    names = SIGNAL_NAMES[lang]

    def add(num, key, text):
        out.append({"num": num, "state": st.get(key, "AMBER"),
                    "name": names[key], "text": text})

    if d["belowMA"]:
        t = ({"R2G": "대형 기술주들이 시장보다 먼저 오르기 시작했습니다. 바닥에서 자주 나오는 모습입니다.",
              "RED": "대형 기술주들이 시장보다 더 크게 빠지고 있습니다."} if ko else
             {"R2G": "Big Tech has started rising ahead of the market — a pattern often seen near bottoms.",
              "RED": "Big Tech is falling harder than the market."}).get(
            st.get("fang"),
            "대형 기술주들의 힘은 시장과 비슷한 수준입니다." if ko else "Big Tech is moving roughly in line with the market.")
    else:
        t = ({"G2R": "대형 기술주들이 먼저 힘이 빠지고 있어 조심할 때입니다.",
              "GREEN": "대형 기술주들이 시장을 잘 이끌고 있습니다."} if ko else
             {"G2R": "Big Tech is losing steam first — time to be careful.",
              "GREEN": "Big Tech is leading the market well."}).get(
            st.get("fang"),
            "대형 기술주들의 힘은 시장과 비슷한 수준입니다." if ko else "Big Tech is moving roughly in line with the market.")
    add(1, "fang", t)

    add(2, "macd", ({"turn": "떨어지던 힘이 줄어들고 방향이 위로 바뀌는 중입니다.",
                     "up": "가격을 밀어 올리는 힘이 유지되고 있습니다.",
                     "roll": "오르던 힘이 꺾이기 시작했습니다.",
                     "down": "아직 내려가는 힘이 더 셉니다."} if ko else
                    {"turn": "Downward pressure is fading and momentum is turning up.",
                     "up": "Upward momentum is holding.",
                     "roll": "The rally's momentum has started to roll over.",
                     "down": "Downward pressure still dominates."})[d["macd"]])

    v = d["fg"]
    if v is None:
        t = "투자 심리 지수는 오늘 확인하지 못했습니다." if ko else "The sentiment index could not be read today."
    elif v <= 10:
        t = (f"투자 심리가 {v}점으로 극도로 얼어붙었습니다. 역설적으로 바닥이 가까울 때 나오는 수치입니다." if ko else
             f"Sentiment is extremely fearful at {v} — paradoxically a reading that often appears near bottoms.")
    elif v <= 25:
        t = (f"투자 심리가 {v}점으로, 시장에 겁먹은 사람이 많습니다." if ko else
             f"Sentiment is fearful at {v}; many investors are scared.")
    elif v >= 75:
        t = (f"투자 심리가 {v}점으로 과열입니다. 모두가 낙관적일 때를 조심해야 합니다." if ko else
             f"Sentiment is overheated at {v}. Be careful when everyone is optimistic.")
    elif v >= 55:
        t = f"투자 심리는 {v}점으로 밝은 편입니다." if ko else f"Sentiment is upbeat at {v}."
    else:
        t = f"투자 심리는 {v}점으로 중간 수준입니다." if ko else f"Sentiment is neutral at {v}."
    add(3, "fg", t)

    vv, vd = d["vix"], d["vixD"]
    if vv >= 40 and vd <= 0:
        t = "시장의 불안이 최고조를 지나 가라앉기 시작했습니다." if ko else "Market fear looks past its peak and is starting to subside."
    elif vv >= 40:
        t = "시장이 매우 불안한 상태입니다." if ko else "The market is highly stressed."
    elif vv >= 28:
        t = "시장이 꽤 불안한 상태입니다." if ko else "The market is fairly nervous."
    elif vv <= 17:
        t = "시장 분위기는 차분합니다." if ko else "The market mood is calm."
    else:
        t = "시장의 불안 수준은 보통입니다." if ko else "Market anxiety is at a normal level."
    add(4, "vix", t)

    dist, slope = d["maDist"], d["maSlope"]
    if dist <= 0:
        t = (f"S&P 500이 장기 평균선보다 {abs(dist):.1f}% 아래에 있습니다. " + (
            "좋은 주식을 싸게 담아 볼 수 있는 구간입니다." if slope > 0
            else "싸게 살 구간이지만 평균선도 내려가고 있어 서두를 필요는 없습니다.")) if ko else (
            f"The S&P 500 is {abs(dist):.1f}% below its long-term average. " + (
                "A zone where good stocks can be picked up cheaply." if slope > 0
                else "A bargain zone, but the average itself is falling — no need to rush."))
    else:
        t = (f"S&P 500이 장기 평균선보다 {dist:.1f}% 위에서 순항하고 있습니다." if ko else
             f"The S&P 500 is cruising {dist:.1f}% above its long-term average.")
    add(5, "ma", t)

    stk = d["eod"]
    if stk >= 3:
        t = (f"장 마감 직전에 큰손들이 {stk}일째 사들이고 있습니다. 바닥 근처에서 자주 보이는 움직임입니다." if ko else
             f"Big players have been buying into the close for {stk} straight days — a pattern often seen near bottoms.")
    elif stk > 0:
        t = (f"마감 직전 매수가 {stk}일째 보이지만 아직 확실하지 않습니다." if ko else
             f"Late-day buying has shown up for {stk} day(s), but it is not yet conclusive.")
    else:
        t = "마감 직전의 큰손 매수는 보이지 않습니다." if ko else "No institutional buying into the close today."
    add(6, "eod", t)

    ld = d["leaders"]
    if ld:
        t = (("잘 버티며 먼저 오르는 종목들(" + ", ".join(ld[:3])
              + (" 등" if len(ld) > 3 else "") + ")이 다음 상승의 후보입니다.") if ko else
             ("Names holding up and rising first (" + ", ".join(ld[:3])
              + (", among others" if len(ld) > 3 else "") + ") are candidates to lead the next advance."))
    else:
        t = "아직 특별히 눈에 띄는 후보 종목은 없습니다." if ko else "No standout candidates yet."
    add(7, "lead", t)

    if d["belowMA"] and d["btcTurned"]:
        t = ("비트코인이 먼저 반등을 시작했습니다. 주식시장이 며칠 뒤 따라오는 경우가 많았습니다." if ko else
             "Bitcoin has started rebounding first — stocks have often followed a few days later.")
    elif d["belowMA"]:
        t = "비트코인에서도 아직 반등 신호는 없습니다." if ko else "No rebound signal from Bitcoin yet either."
    elif st.get("btc") == "G2R":
        t = "비트코인이 먼저 흔들리고 있어 경계 신호입니다." if ko else "Bitcoin is wobbling first — a caution signal."
    elif st.get("btc") == "GREEN":
        t = "비트코인도 순항하고 있습니다." if ko else "Bitcoin is cruising as well."
    else:
        t = "비트코인은 큰 움직임이 없습니다." if ko else "Bitcoin shows no major move."
    add(9, "btc", t)
    return out


def combo_advice(mo, wk, dy, lang="ko"):
    """월간/주간/일간 종합 상태 조합 → (국면명, 권장 행동, 톤).
    위에서부터 첫 일치 규칙 적용. 규칙 추가·수정 지점 (문구는 ko/en 쌍으로 유지)."""
    UP, DN = ("GREEN", "R2G"), ("RED", "G2R")
    ko = lang == "ko"
    rules = [
        (mo in DN and wk in DN and dy in DN,
         ("시장 전체가 내리막", "주식은 줄이고 현금을 확보하세요. 잘 버티는 종목만 지켜보기") if ko else
         ("Broad market in decline", "Reduce stock exposure and raise cash. Just watch the names holding up"), "reduce"),
        (dy == "R2G" and wk in DN and mo in DN,
         ("바닥 신호가 살짝 보임", "아직 확실하지 않습니다. 며칠 더 지켜보고, 사더라도 아주 조금만") if ko else
         ("Faint bottoming signal", "Not confirmed yet. Watch a few more days; if you buy, buy only a little"), "caution"),
        (wk == "R2G" and dy in UP and mo != "GREEN",
         ("바닥을 다지는 중", "관심 우량주를 소액으로 나눠서 사기 시작 (1단계)") if ko else
         ("Building a bottom", "Start buying quality watchlist names in small installments (stage 1)"), "buy"),
        (mo in ("RED", "AMBER", "G2R") and wk in UP and dy == "GREEN",
         ("반등이 확인됨", "사는 양을 한 단계 늘리기 (2단계)") if ko else
         ("Rebound confirmed", "Step up the buying one notch (stage 2)"), "buy"),
        (mo == "R2G" and (wk in DN or dy in DN),
         ("신호가 엇갈림", "큰 흐름은 좋아지는데 단기가 흔들립니다. 추가 매수는 잠시 멈추고 기다리기") if ko else
         ("Mixed signals", "The big trend is improving but the short term is shaky. Pause new buying and wait"), "caution"),
        (mo == "R2G",
         ("큰 흐름이 좋아지는 중", "목표한 비중까지 조금씩 늘려 가기") if ko else
         ("Big trend improving", "Gradually build toward your target allocation"), "buy"),
        (mo == "GREEN" and wk not in DN and dy in ("RED", "R2G"),
         ("상승장 속 잠깐 쉬어가는 구간", "좋은 종목을 싸게 살 기회입니다. 나눠서 매수") if ko else
         ("Brief pause within an uptrend", "A chance to buy good names cheaply. Buy in installments"), "buy"),
        (mo == "GREEN" and wk == "GREEN" and dy == "G2R",
         ("단기 과열 뒤 주춤", "새로 사는 건 잠시 쉬고, 가진 것은 그대로 유지") if ko else
         ("Stalling after short-term overheating", "Hold off on new buys; keep what you own"), "caution"),
        (mo == "GREEN" and wk in DN,
         ("조정이 올 수 있음", "이익 난 것 일부는 팔고, 새 매수는 중단") if ko else
         ("A correction may be coming", "Take some profits and stop new buying"), "reduce"),
        (mo == "GREEN" and wk in ("GREEN", "AMBER") and dy in ("GREEN", "AMBER"),
         ("꾸준한 상승 흐름", "그대로 보유. 쉬어가는 구간이 오면 추가 매수 고려") if ko else
         ("Steady uptrend", "Stay invested. Consider adding on pullbacks"), "hold"),
    ]
    for cond, (name, act), tone in rules:
        if cond:
            return name, act, tone
    avg = (SC[mo] + SC[wk] + SC[dy]) / 3
    if avg >= 0.35:
        return (("좋은 쪽에 가까움", "그대로 보유") if ko else ("Leaning positive", "Stay invested")) + ("hold",)
    if avg <= -0.35:
        return (("나쁜 쪽에 가까움", "방어 위주로. 현금 확보") if ko else ("Leaning negative", "Play defense. Raise cash")) + ("reduce",)
    return (("방향이 뚜렷하지 않음", "일단 관망하며 가진 것만 유지") if ko else
            ("No clear direction", "Wait and see; just hold what you own")) + ("neutral",)


# ============================================================
# 카드 텍스트
# ============================================================
def brief_texts(d, w, m, lang="ko"):
    """카드 문구. metric = 오늘 값, sub = 평가(주간·월간까지 종합한 해석),
    rule = 기준(판정 임계값). 문장은 완결형으로 기술한다. 문구는 ko/en 쌍."""
    ko = lang == "ko"
    ph = ({"down": "하락 계속", "turn": lambda x: f"바닥 반전 {x}봉째",
           "up": "상승 흐름", "roll": lambda x: f"고점 이탈 {x}봉째"} if ko else
          {"down": "Still falling", "turn": lambda x: f"Bottom turn, bar {x}",
           "up": "Uptrend", "roll": lambda x: f"Rolling over, bar {x}"})

    def phase(mm):
        p = ph[mm["macd"]]
        return p(mm["histStreak"]) if callable(p) else p

    L_D, L_W, L_M = (("일간", "주간", "월간") if ko else ("Daily", "Weekly", "Monthly"))
    below = d["belowMA"]

    # P1 FANG
    f_d, f_w, f_m = d["fangRS"], w["fangRS"], m["fangRS"]
    f_line = f"{L_D} {fmt(f_d)}%p · {L_W} {fmt(f_w)}%p · {L_M} {fmt(f_m)}%p."
    if below and f_d >= 1.5:
        f_msg = (("월간은 아직 뒤처지지만 일간부터 앞서기 시작해 바닥 반등 초기 신호로 해석됩니다."
                  if f_m < 0 else "세 주기 모두 지수를 앞서고 있어 반등이 자리 잡는 신호입니다.") if ko else
                 ("Monthly still lags, but daily has started to lead — an early sign of a bottoming rebound."
                  if f_m < 0 else "All three timeframes are beating the index — the rebound is taking hold."))
    elif below and f_d <= -1.5:
        f_msg = ("FANG이 낙폭을 주도하고 있어 하락이 더 이어질 가능성이 큽니다." if ko else
                 "FANG is leading the decline — further downside is likely.")
    elif not below and f_d <= -2:
        f_msg = ("상승장에서 FANG이 먼저 뒤처지기 시작해 고점 경계 신호입니다." if ko else
                 "FANG is starting to lag in an uptrend — a topping warning.")
    elif not below and f_d >= 0:
        f_msg = (("장기와 단기 모두 FANG이 앞서고 있어 상승 흐름이 건강합니다."
                  if f_m >= 0 else "장기 열세는 남아 있으나 단기 주도력은 유지되고 있습니다.") if ko else
                 ("FANG leads on both long and short horizons — a healthy uptrend."
                  if f_m >= 0 else "Long-term weakness remains, but short-term leadership is intact."))
    else:
        f_msg = ("뚜렷한 방향성이 없어 판단을 유보합니다." if ko else
                 "No clear direction — judgment reserved.")

    # P2 MACD
    p_d, p_w, p_m = d["macd"], w["macd"], m["macd"]
    c_line = f"{L_D} {phase(d)} · {L_W} {phase(w)} · {L_M} {phase(m)}."
    if p_d in ("turn", "up") and p_m == "down":
        c_msg = ("단기부터 차례로 돌아서는 전형적인 바닥 형성 순서입니다." if ko else
                 "Short-term turning first — the classic bottoming sequence.")
    elif p_d == "up" and p_w == "up" and p_m == "up":
        c_msg = ("세 주기 모두 상승 흐름을 유지하고 있습니다." if ko else
                 "All three timeframes remain in uptrends.")
    elif p_d in ("roll", "down") and p_m == "up":
        c_msg = ("장기 추세는 살아 있으나 단기가 먼저 꺾여 조정 초기 가능성이 있습니다." if ko else
                 "The long-term trend is intact, but the short term has rolled over first — possibly an early correction.")
    elif p_d == "down" and p_w == "down":
        c_msg = ("전 주기에서 하락 압력이 이어지고 있습니다." if ko else
                 "Downward pressure persists across timeframes.")
    else:
        c_msg = ("주기별 신호가 엇갈려 추가 확인이 필요합니다." if ko else
                 "Signals are mixed across timeframes — needs more confirmation.")

    # P3 F&G (일간 전용, 주·월 변화폭으로 맥락 제공)
    if d["fg"] is None:
        fg_txt = "N/A"
        fg_msg = ("조회에 실패했습니다. run(fg=값)으로 수동 입력할 수 있습니다." if ko else
                  "Lookup failed. You can enter it manually with run(fg=value).")
    else:
        v = d["fg"]
        zone = (("극단 공포" if v <= 10 else "공포" if v <= 25
                 else "과열" if v >= 75 else "낙관" if v >= 55 else "중립") if ko else
                ("extreme fear" if v <= 10 else "fear" if v <= 25
                 else "overheated" if v >= 75 else "optimistic" if v >= 55 else "neutral"))
        fg_txt = f"{v} ({fmt(d['fgD'], 0)})"
        dw, dm = w["fgD"], m["fgD"]
        if dm < 0 and dw > 0:
            tail = (f"한 달 전보다 {abs(dm)} 낮지만 최근 일주일 사이 {dw} 회복했습니다." if ko else
                    f"Still {abs(dm)} below a month ago, but up {dw} over the past week.")
        elif dw > 0 and dm > 0:
            tail = "심리가 꾸준히 개선되고 있습니다." if ko else "Sentiment keeps improving."
        elif dw < 0 and dm < 0:
            tail = "심리가 계속 위축되고 있습니다." if ko else "Sentiment keeps deteriorating."
        else:
            tail = (f"일주일 변화 {fmt(dw, 0)}, 한 달 변화 {fmt(dm, 0)}입니다." if ko else
                    f"Change: {fmt(dw, 0)} over a week, {fmt(dm, 0)} over a month.")
        fg_msg = (f"현재 {v}, {zone} 구간입니다. " if ko else f"Now {v}, in the {zone} zone. ") + tail

    # P4 VIX
    vv, vd, vm = d["vix"], d["vixD"], m["vixD"]
    vzone = (("공포 정점" if vv >= 40 else "높은 변동성" if vv >= 28
              else "안정" if vv <= 17 else "보통") if ko else
             ("panic peak" if vv >= 40 else "high volatility" if vv >= 28
              else "calm" if vv <= 17 else "normal"))
    if vv >= 40 and vd <= 0:
        v_msg = (f"현재 {vv:.0f}, 3개월간 {fmt(vm, 0)} 급등했으나 최근 5일은 {fmt(vd, 0)}로 꺾여 공포가 정점을 지나는 신호입니다." if ko else
                 f"Now {vv:.0f}, up {fmt(vm, 0)} in 3 months but {fmt(vd, 0)} over the last 5 days — fear looks past its peak.")
    elif vv <= 17:
        v_msg = (f"현재 {vv:.0f}로 변동성이 낮게 유지되어 시장이 안정적입니다." if ko else
                 f"Now {vv:.0f}; volatility stays low and the market is stable.")
    elif vd > 0 and vv >= 28:
        v_msg = (f"현재 {vv:.0f}, 변동성이 계속 확대되고 있어 위험 구간입니다." if ko else
                 f"Now {vv:.0f}; volatility keeps expanding — a risk zone.")
    else:
        v_msg = (f"현재 {vv:.0f}, {vzone} 구간입니다. 5일 {fmt(vd, 0)} · 3개월 {fmt(vm, 0)} 변화로 큰 방향성은 없습니다." if ko else
                 f"Now {vv:.0f}, in the {vzone} zone. 5-day {fmt(vd, 0)} · 3-month {fmt(vm, 0)} — no big direction.")

    # P5 180일선
    dist, slope = d["maDist"], d["maSlope"]
    if dist <= 0:
        ma_msg = ((f"SPY가 180일선보다 {abs(dist):.1f}% 아래에 있습니다. "
                   + ("선 자체는 상승을 유지하고 있어 되돌림 매수 구간으로 해석됩니다."
                      if slope > 0 else "선의 기울기도 꺾여 있어 신중한 분할 접근이 필요합니다.")) if ko else
                  (f"SPY is {abs(dist):.1f}% below its 180-day line. "
                   + ("The line itself still rises, so this reads as a pullback buying zone."
                      if slope > 0 else "The line's slope has also turned down — approach cautiously in installments.")))
    else:
        ma_msg = ((f"SPY가 180일선보다 {dist:.1f}% 위에 있으며, "
                   + ("선의 기울기도 상승을 유지하고 있습니다." if slope > 0
                      else "선의 기울기는 눕기 시작했습니다.")) if ko else
                  (f"SPY is {dist:.1f}% above its 180-day line, "
                   + ("and the line keeps rising." if slope > 0
                      else "but the line has started to flatten.")))

    # P6 장마감 매수
    st, n20 = d["eod"], d["eod20"]
    if st >= 3:
        e_msg = (f"마감 30분 순매수가 {st}일째 이어지고 있습니다. 최근 20거래일 중 {n20}일 관측되어 기관성 매집 신호에 해당합니다." if ko else
                 f"Net buying in the last 30 minutes has run {st} straight days — seen on {n20} of the last 20 sessions, an institutional accumulation signal.")
    elif st > 0:
        e_msg = (f"마감 30분 순매수가 {st}일째입니다. 최근 20거래일 중 {n20}일 관측되어 아직 추세로 보기는 이릅니다." if ko else
                 f"Late-30-minute net buying for {st} day(s); {n20} of the last 20 sessions — too early to call a trend.")
    else:
        e_msg = (f"오늘 기준 연속 매수는 없습니다. 최근 20거래일 중 {n20}일 관측됐습니다." if ko else
                 f"No buying streak as of today; observed on {n20} of the last 20 sessions.")

    # P7 주도주
    ld = d["leaders"]
    if ld:
        l_msg = ((f"{len(ld)}종목이 조건을 통과했습니다: " if ko else
                  f"{len(ld)} names passed the screen: ") + " · ".join(ld[:8]) + ".")
        if below and len(ld) >= 5:
            l_msg += (" 하락장에서 후보군이 두터워 차기 주도군이 형성되는 신호입니다." if ko else
                      " A deep candidate pool in a downturn — a sign the next leadership group is forming.")
    else:
        l_msg = "현재 조건을 통과한 종목이 없습니다." if ko else "No names currently pass the screen."

    # P9 BTC
    b_d, b_w, b_m = d["btcMom"], w["btcMom"], m["btcMom"]
    b_line = ((f"7일 {fmt(b_d)}% · 4주 {fmt(b_w)}% · 3개월 {fmt(b_m)}%.") if ko else
              (f"7d {fmt(b_d)}% · 4w {fmt(b_w)}% · 3m {fmt(b_m)}%."))
    if below and d["btcTurned"]:
        b_msg = (("중기 조정 속에서 단기 반등이 시작되어 주식시장 선행 신호로 해석됩니다."
                  if b_m < 0 else "전 구간이 상승으로 돌아서 위험선호 회복 신호입니다.") if ko else
                 ("A short-term rebound has begun within the mid-term correction — read as a leading signal for stocks."
                  if b_m < 0 else "All horizons have turned up — risk appetite is recovering."))
    elif below:
        b_msg = ("아직 선행 반등 신호는 나타나지 않았습니다." if ko else
                 "No leading rebound signal yet.")
    elif b_d > 0 and b_m > 0:
        b_msg = ("전 구간 상승으로 위험선호가 유지되고 있습니다." if ko else
                 "Up across horizons — risk appetite holds.")
    elif b_d < -3:
        b_msg = ("BTC가 먼저 밀리기 시작해 경계 신호입니다." if ko else
                 "BTC is slipping first — a caution signal.")
    else:
        b_msg = "큰 방향성 없이 횡보하고 있습니다." if ko else "Drifting sideways with no clear direction."

    NM = SIGNAL_NAMES[lang]
    return {
        "fang": {"num": 1, "name": NM["fang"], "metric": f"RS {fmt(f_d)}%p",
                 "sub": f_line + " " + f_msg,
                 "rule": ("하락장에서는 상대수익 +1.5%p 이상이면 바닥 신호, −1.5%p 이하이면 하락 주도로 판정합니다. 상승장에서 −2%p 이하로 뒤처지면 고점 경고입니다." if ko else
                          "In a downturn, relative return ≥ +1.5%p is a bottoming signal and ≤ −1.5%p means FANG leads the decline. In an uptrend, lagging by ≤ −2%p is a topping warning.")},
        "macd": {"num": 2, "name": NM["macd"], "metric": phase(d),
                 "sub": c_line + " " + c_msg,
                 "rule": ("MACD(12,26,9) 히스토그램이 저점 이후 3봉(주간·월간은 2봉) 연속 개선되면 바닥 신호, 고점 이후 같은 길이로 약화되면 경고로 판정합니다." if ko else
                          "A MACD(12,26,9) histogram improving 3 consecutive bars off a low (2 for weekly/monthly) is a bottoming signal; weakening the same length off a high is a warning.")},
        "fg": {"num": 3, "name": NM["fg"], "metric": fg_txt,
               "sub": fg_msg,
               "rule": ("지수가 10 이하이면 바닥 근접(역발상 매수 구간), 25 이하이면 공포, 75 이상이면 과열 경계로 판정합니다." if ko else
                        "Index ≤ 10 means near a bottom (contrarian buy zone), ≤ 25 fear, ≥ 75 overheating caution.")},
        "vix": {"num": 4, "name": NM["vix"], "metric": f"VIX {vv:.0f} ({fmt(vd, 0)})",
                "sub": v_msg,
                "rule": ("VIX가 40 이상에서 꺾이면 전환 신호, 28 이상이면 위험, 17 이하이면 안정으로 판정합니다. 추후 뉴스 심리 분석으로 대체할 예정입니다." if ko else
                         "VIX rolling over from above 40 is a turning signal; ≥ 28 is risky, ≤ 17 stable. To be replaced by news-sentiment analysis later.")},
        "ma": {"num": 5, "name": NM["ma"], "metric": f"180D {fmt(dist)}%",
               "sub": ma_msg,
               "rule": ("SPY가 180일선 이하로 내려오면 우량주 분할매수 구간, +3%를 넘으면 정상 상승 추세로 판정합니다." if ko else
                        "SPY below the 180-day line marks a zone to accumulate quality names in installments; above +3% is a normal uptrend.")},
        "eod": {"num": 6, "name": NM["eod"], "metric": (f"{st}일 연속" if ko else f"{st}-day streak"),
                "sub": e_msg,
                "rule": ("마감 30분봉 수익률이 +0.1%를 넘는 날이 3일 이상 이어지면 기관성 매집 신호로 판정합니다." if ko else
                         "Three or more consecutive days with the final 30-minute bar returning over +0.1% counts as institutional accumulation.")},
        "lead": {"num": 7, "name": NM["lead"], "metric": (f"{len(ld)}종목 포착" if ko else f"{len(ld)} names flagged"),
                 "sub": l_msg,
                 "rule": ("하락장에서는 낙폭이 SPY의 60% 이내로 방어되고 최근 5일 수익률이 플러스인 종목을, 상승장에서는 20일 상대수익이 +3%p를 넘는 종목을 후보로 선정합니다." if ko else
                          "In a downturn: names whose drawdown stays within 60% of SPY's and whose 5-day return is positive. In an uptrend: names with 20-day relative return over +3%p.")},
        "btc": {"num": 9, "name": NM["btc"], "metric": f"{fmt(b_d)}% " + ("(7일)" if ko else "(7d)"),
                "sub": b_line + " " + b_msg,
                "rule": ("하락장에서 BTC 모멘텀이 +2% 이상이고 2일 연속 상승하면 주식시장 선행 반등 신호로 판정합니다." if ko else
                         "In a downturn, BTC momentum ≥ +2% with 2 consecutive up days is a leading rebound signal for stocks.")},
    }


# ============================================================
# HTML 렌더링 — 서버사이드 정적 생성 (JavaScript 0줄)
#   Jupyter의 HTML 뷰어는 <script>를 차단하지만 CSS는 그대로 두므로,
#   탭 전환을 CSS(숨긴 라디오 + :checked)로 구현해 어디서 열어도 보이게 함.
# ============================================================
import math
import html as _h

_CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;500;700&family=IBM+Plex+Mono:wght@400;600&display=swap');
:root{--bg:#0b1220;--card:#111a2b;--line:#1e2a40;--tx:#e6eaf2;--mut:#8a94a8;--dim:#5c6a82;}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--tx);font-family:'IBM Plex Sans KR',system-ui,sans-serif;font-size:14px}
.mono{font-family:'IBM Plex Mono',ui-monospace,monospace}
.wrap{position:relative;max-width:1000px;margin:0 auto;padding:20px 16px 48px}
/* 언어 토글: 숨긴 체크박스 + :checked (JS 없이 동작 — Jupyter 뷰어 호환) */
.lang-sw{position:absolute;left:-9999px}
.lang{display:flex;flex-direction:column;gap:20px}
.lang-en{display:none}
.lang-sw:checked ~ .wrap .lang-ko{display:none}
.lang-sw:checked ~ .wrap .lang-en{display:flex}
.lang-btn{position:absolute;top:22px;right:16px;z-index:5;display:inline-flex;border:1px solid var(--line);border-radius:999px;overflow:hidden;cursor:pointer;font-size:12px;user-select:none;background:var(--card)}
.lang-btn .opt{padding:5px 12px;color:var(--mut)}
.lang-btn .opt-ko{background:#1e2a40;color:var(--tx);font-weight:600}
.lang-sw:checked ~ .wrap .lang-btn .opt-ko{background:transparent;color:var(--mut);font-weight:400}
.lang-sw:checked ~ .wrap .lang-btn .opt-en{background:#1e2a40;color:var(--tx);font-weight:600}
/* 좁은 화면: 절대위치 버튼이 헤더 첫 줄과 겹치므로 흐름 배치(헤더 위, 오른쪽 정렬)로 전환 */
@media(max-width:640px){.lang-btn{position:static;display:flex;width:max-content;margin:0 0 12px auto}}
.eyebrow{font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:var(--dim)}
h1{font-size:24px;margin-top:4px}
.sub{color:var(--mut);margin-top:4px}
.asof{margin-top:10px;display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.tag{border:1px solid var(--line);border-radius:999px;padding:4px 12px;font-size:12px;color:var(--mut)}
.tag b{color:var(--tx)}
.tag.warn{border-color:#eab30855;color:#eab308;background:#eab30814}
.tag.mock{border-color:#fb923c66;color:#fb923c;background:#fb923c14}
/* 상태 인디케이터: 왼쪽 RED → 오른쪽 GREEN 미니 바
   정적 상태 = 바 위 해당 위치 마커, 전환 상태 = 바 위 방향 화살표 */
.ind{display:inline-flex;flex-direction:column;flex:none;gap:2px}
.ind-top{display:flex;align-items:center;justify-content:center;height:8px}
.ind.lg .ind-top{height:11px}
.ind-bar{position:relative;height:6px;border-radius:999px;background:linear-gradient(90deg,#ef4444,#eab308,#22c55e)}
.ind.lg .ind-bar{height:8px}
.ind-mk{position:absolute;top:50%;transform:translate(-50%,-50%);width:3px;height:11px;border-radius:2px;background:#fff;box-shadow:0 0 3px rgba(0,0,0,.7)}
.ind.lg .ind-mk{width:4px;height:14px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:20px}
.row{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:12px}
.score{font-size:28px;font-weight:600}
.sind-row{display:flex;flex-wrap:wrap;gap:22px;margin-top:24px}
.sind{display:inline-flex;align-items:center;gap:8px}
.sind-lb{font-size:12px;color:var(--mut)}
.sind.on .sind-lb{color:var(--tx);font-weight:600}
.sind-bar{position:relative;display:inline-block;width:110px;height:6px;border-radius:999px;background:linear-gradient(90deg,#ef4444,#eab308,#22c55e)}
.sind-ar{position:absolute;top:-12px;transform:translateX(-50%)}
.verdict{font-size:19px;font-weight:700;margin-top:16px}
.v-act{margin-top:4px;font-size:13px}
.drow{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px;font-size:13px;line-height:1.6}
.dno{flex:none;width:22px;font-weight:600}
.dnm{flex:none;color:var(--mut)}
.dtx{flex:1;min-width:200px}
.nrow{display:flex;flex-wrap:wrap;gap:10px;align-items:baseline;margin-top:10px}
.nno{color:var(--dim);font-size:12px}
.ntt{flex:1;min-width:200px;font-size:13px;line-height:1.5}
.ntt a{color:var(--tx);text-decoration:none}
.ntt a:hover{text-decoration:underline}
.nmeta{font-size:11px;color:var(--dim);white-space:nowrap}
.note{font-size:12px;color:var(--dim);margin-top:8px;line-height:1.6}
.ov{padding:16px 18px}
.ov-left{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.ov-score{display:flex;align-items:baseline;gap:8px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px;display:flex;flex-direction:column;gap:10px}
.card .hd{display:flex;justify-content:space-between;align-items:flex-start;gap:8px}
.card .pn{font-size:11px;color:var(--dim)}
.card .nm{font-weight:600;margin-top:2px}
.card .mv{font-size:18px;font-weight:600}
.card .ms{font-size:12px;color:var(--mut);margin-top:2px}
.card .rule{font-size:12px;color:var(--dim);line-height:1.6}
.crow{display:flex;flex-wrap:wrap;gap:10px;align-items:baseline;margin-top:10px;font-size:13px}
.clb{flex:none;width:150px;color:var(--mut)}
.cvl{flex:1;min-width:200px;line-height:1.6}
.cvl b{color:var(--tx)}
.dday{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:1px 8px;font-size:11px;color:var(--mut);margin-left:6px}
.dday.hot{border-color:#fb923c66;color:#fb923c;background:#fb923c14}
.stblwrap{overflow-x:auto;margin-top:10px}
.stbl{width:100%;border-collapse:collapse;font-size:13px}
.stbl th{color:var(--dim);font-weight:500;text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);font-size:12px;white-space:nowrap}
.stbl td{padding:6px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
.stbl tr:last-child td{border-bottom:0}
.foot{font-size:12px;color:var(--dim);line-height:1.7}
@media(max-width:520px){.score{font-size:26px}.combo-verdict{font-size:18px}}
"""


def _esc(x):
    return _h.escape(str(x), quote=True)


def _arrow_svg(direction, color, w, h):
    """전환 방향 화살표. 'r' = 초록(오른쪽) 방향, 'l' = 빨강(왼쪽) 방향."""
    y = h / 2
    if direction == "r":
        return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
                f'<line x1="2" y1="{y}" x2="{w-8}" y2="{y}" stroke="{color}" stroke-width="2" stroke-linecap="round"/>'
                f'<path d="M{w-8} {y-3.5} L{w-1.5} {y} L{w-8} {y+3.5} Z" fill="{color}"/></svg>')
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            f'<line x1="8" y1="{y}" x2="{w-2}" y2="{y}" stroke="{color}" stroke-width="2" stroke-linecap="round"/>'
            f'<path d="M8 {y-3.5} L1.5 {y} L8 {y+3.5} Z" fill="{color}"/></svg>')


def _ind(k, lg=False):
    """상태 인디케이터: RED-AMBER-GREEN 그라데이션 바(왼쪽 red, 오른쪽 green).
    GREEN/AMBER/RED = 바 위 위치 마커, R2G/G2R = 바 위쪽에 진행 방향 화살표."""
    w, h = (64, 11) if lg else (40, 8)
    if k == "R2G":
        top, mk = _arrow_svg("r", ST["GREEN"]["color"], w, h), ""
    elif k == "G2R":
        top, mk = _arrow_svg("l", ST["RED"]["color"], w, h), ""
    else:
        pos = {"RED": 8, "AMBER": 50, "GREEN": 92}[k]
        top = ""
        mk = '<span class="ind-mk" style="left:' + str(pos) + '%"></span>'
    return ('<span class="ind' + (' lg' if lg else '') + '" style="width:' + str(w) + 'px">'
            + '<span class="ind-top">' + top + '</span>'
            + '<span class="ind-bar">' + mk + '</span></span>')


def _score_ind(T):
    """타임프레임 하나의 소형 점수 바: 마커 = 점수 위치, 전환이면 위에 방향 화살표."""
    k = T["overall"]
    pct = (T["score"] + 1) / 2 * 100
    ar = ''
    if k in ("R2G", "G2R"):
        d = 'r' if k == "R2G" else 'l'
        ac = ST["GREEN" if d == 'r' else "RED"]["color"]
        apct = min(88.0, max(12.0, pct))
        ar = ('<span class="sind-ar" style="left:' + f"{apct:.1f}" + '%">'
              + _arrow_svg(d, ac, 22, 9) + '</span>')
    return ('<span class="sind-bar">' + ar
            + '<span class="ind-mk" style="left:' + f"{pct:.1f}" + '%"></span></span>')


def _overall_panel(tfs_all, combo, digest, lang="ko"):
    """오늘의 종합 판정: 일간 점수 + 월간·주간·일간 소형 점수 바 + 국면·권장 행동."""
    ko = lang == "ko"
    T = tfs_all["daily"]
    c = ST[T["overall"]]["color"]
    dg = ''
    if digest:
        rows = ''.join('<div class="drow"><span class="dno mono" style="color:'
                       + ST[x["state"]]["color"] + '">P' + str(x["num"]) + '</span>'
                       + '<span class="dnm">(' + _esc(x.get("name", "")) + ')</span>'
                       + '<span class="dtx">' + _esc(x["text"]) + '</span></div>' for x in digest)
        dg = ('<div class="eyebrow mono" style="margin-top:18px">'
              + ('오늘의 신호 한 줄 요약' if ko else "Today's Signals in Plain Words") + '</div>'
              + rows)
    tone_c = TONE_COLORS.get(combo["tone"], "#8a94a8")
    groups = ''
    for t in ("monthly", "weekly", "daily"):
        groups += ('<span class="sind' + (' on' if t == "daily" else '') + '">'
                   + '<span class="sind-lb mono">' + TF_LABEL[lang][t] + '</span>'
                   + _score_ind(tfs_all[t]) + '</span>')
    return ('<div class="panel ov"><div class="row">'
            + '<div class="ov-left"><span class="eyebrow mono">'
            + ('오늘의 종합 판정' if ko else "Today's Overall Verdict") + '</span></div>'
            + '<div class="ov-score"><span class="score mono" style="color:' + c + '">'
            + fmt(T["score"] * 100) + '</span>'
            + '<span class="note mono" style="margin-top:0">/ ±100</span></div></div>'
            + '<div class="sind-row">' + groups + '</div>'
            + '<div class="verdict" style="color:' + tone_c + '">' + _esc(combo["name"]) + '</div>'
            + '<div class="v-act">' + ('권장 행동: ' if ko else 'Suggested action: ')
            + _esc(combo["action"]) + '</div>'
            + dg + '</div>')


def _card(sig, lang="ko"):
    ko = lang == "ko"
    c = ST[sig["state"]]["color"]
    return ('<div class="card"><div class="hd">'
            + '<div><div class="pn mono"><span style="color:' + c
            + ';font-weight:700">P' + str(sig["num"]) + '</span> · w ' + str(sig["weight"]) + '</div>'
            + '<div class="nm">' + _esc(sig["name"]) + '</div></div>'
            + _ind(sig["state"], True) + '</div>'
            + '<div><div class="mv mono">' + _esc(sig["metric"]) + '</div>'
            + '<div class="ms">' + ('평가: ' if ko else 'Assessment: ') + _esc(sig["sub"]) + '</div></div>'
            + '<div class="rule">' + ('기준: ' if ko else 'Rule: ') + _esc(sig["rule"]) + '</div></div>')


def _cal_panel(payload, lang="ko"):
    """캘린더 & 수급 패널: 옵션 만기일 / 윈도우 드레싱 / 숏 비중 (점수 미반영 참고 정보)."""
    cal = payload.get("calendar")
    if not cal:
        return ''
    ko = lang == "ko"
    op, opn, wd = cal["opex"], cal["opex_next"], cal["wd"]

    def fd(x):
        return x["date"] + ' (' + DOW[lang][x["dow"]] + ')'

    def dd(n, hot=False):
        if n == 0:
            t = "오늘" if ko else "Today"
        elif n > 0:
            t = f"D-{n}"
        else:
            t = f"{-n}일 지남" if ko else f"{-n}d ago"
        return '<span class="dday' + (' hot' if hot else '') + ' mono">' + t + '</span>'

    tw = (' · <b>트리플 위칭</b>(지수 선물·옵션 동시 만기, 변동성 확대 주의)' if ko else
          ' · <b>Triple witching</b> (index futures & options expire together — expect volatility)')
    if op["passed"]:
        opex_html = ((('이번 달 만기(' + fd(op) + ')는 지났습니다 → 다음 만기 <b>') if ko else
                      ("This month's expiration (" + fd(op) + ") has passed → next <b>"))
                     + fd(opn) + '</b>' + dd(opn["dday"], opn["dday"] <= 5)
                     + (tw if opn["triple"] else ''))
    else:
        opex_html = ('<b>' + fd(op) + '</b>' + dd(op["dday"], op["dday"] <= 5)
                     + (tw if op["triple"] else '')
                     + (' · 다음 달: ' if ko else ' · Next month: ') + fd(opn))
    wd_html = ('<b>' + fd(wd["start"]) + ' ~ ' + fd(wd["end"]) + '</b> '
               + ('(월말 마지막 4거래일)' if ko else '(last 4 trading days of the month)')
               + (dd(0) if wd["active"] else dd(wd["dday"], 0 < wd["dday"] <= 3))
               + ((' · <b>분기말</b>이라 리밸런싱 효과가 큰 달입니다' if wd["quarter"]
                   else ' · 월말 리밸런싱 수준') if ko else
                  (' · <b>Quarter-end</b> — rebalancing effects run stronger this month' if wd["quarter"]
                   else ' · Ordinary month-end rebalancing')))

    rows = ''
    for s in payload.get("shorts") or []:
        pct = f'{s["pct"]:.1f}%' if s.get("pct") is not None else 'N/A'
        dtc = ((f'{s["dtc"]:.1f}일' if ko else f'{s["dtc"]:.1f}d')
               if s.get("dtc") is not None else 'N/A')
        chg = (fmt(s["chg"]) + '%') if s.get("chg") is not None else 'N/A'
        hot = s.get("pct") is not None and s["pct"] >= 10
        nm = s["name"] if ko else s.get("name_en") or s["name"]
        rows += ('<tr><td>' + _esc(nm)
                 + ' <span class="mono" style="color:var(--dim)">' + _esc(s["ticker"]) + '</span></td>'
                 + '<td class="mono"' + (' style="color:#fb923c;font-weight:600"' if hot else '') + '>' + pct + '</td>'
                 + '<td class="mono">' + dtc + '</td>'
                 + '<td class="mono">' + chg + '</td>'
                 + '<td class="mono" style="color:var(--dim)">' + _esc(s.get("asof") or "") + '</td></tr>')
    th = (('<tr><th>종목</th><th>숏 비중(유동주식 대비)</th><th>숏 커버 소요일</th>'
           '<th>숏 주식수 전월 대비</th><th>집계 기준일</th></tr>') if ko else
          ('<tr><th>Name</th><th>Short % of float</th><th>Days to cover</th>'
           '<th>Shares short vs prior month</th><th>As of</th></tr>'))
    tbl = (('<div class="stblwrap"><table class="stbl">' + th + rows + '</table></div>') if rows
           else '<div class="note">'
                + ('숏 비중 조회 실패 — 네트워크 확인 후 다시 실행하세요.' if ko else
                   'Short-interest lookup failed — check the network and rerun.') + '</div>')

    return ('<div class="panel"><div class="eyebrow mono">'
            + ('캘린더 &amp; 수급 체크 (참고 정보 · 종합점수 미반영)' if ko else
               'Calendar &amp; Positioning Check (reference only · not scored)') + '</div>'
            + '<div class="crow"><span class="clb">'
            + ('옵션 만기일 (OPEX)' if ko else 'Options Expiration (OPEX)')
            + '</span><span class="cvl">' + opex_html + '</span></div>'
            + '<div class="crow"><span class="clb">'
            + ('윈도우 드레싱 예상' if ko else 'Window Dressing Window')
            + '</span><span class="cvl">' + wd_html + '</span></div>'
            + tbl
            + '<div class="note">'
            + ('숏 비중은 거래소 공식 집계(월 2회 발표, 약 2주 지연) 기준이라 실시간이 아닙니다. '
               '숏 비중 10% 이상은 주황색으로 표시하며, 숏 커버 소요일(Days to Cover)이 길수록 숏스퀴즈 가능성이 커집니다. '
               '만기일·월말 날짜는 주말만 제외한 계산이라 미국 휴장일과 겹치면 실제로는 직전 거래일로 이동합니다. '
               'SpaceX는 비상장 기업이라 숏 데이터가 존재하지 않아, SpaceX 지분을 보유한 상장 펀드 DXYZ(Destiny Tech100)로 대체 표시합니다.' if ko else
               'Short interest comes from the exchanges\' official tally (published twice a month, ~2-week lag) — not real-time. '
               'Short % of float at 10% or more is shown in orange; the longer the days-to-cover, the higher the squeeze potential. '
               'Expiration and month-end dates exclude weekends only, so a date falling on a US market holiday actually moves to the prior trading day. '
               'SpaceX is private and has no short data, so DXYZ (Destiny Tech100), a listed fund holding SpaceX, is shown as a proxy.')
            + '</div></div>')


def _render_body(payload, lang):
    """한 언어분의 본문(헤더~푸터). render_html이 두 언어를 모두 담아 토글한다."""
    ko = lang == "ko"
    a = payload["asof"]
    tx = payload["texts"][lang]

    def fdow(i):
        return DOW[lang][i]

    stt = a.get("status", "current")
    if stt == "weekend":
        asof_tag = ('<span class="tag warn">오늘 ' + a["today"] + ' (' + fdow(a["today_dow"])
                    + ')은 주말 휴장 → 직전 영업일 종가 기준</span>') if ko else (
            '<span class="tag warn">Today ' + a["today"] + ' (' + fdow(a["today_dow"])
            + ') is a weekend — based on the prior business day\'s close</span>')
    elif stt == "pre_open":
        asof_tag = ('<span class="tag warn">오늘 개장 전 → 직전 영업일 종가 기준</span>' if ko else
                    '<span class="tag warn">Before today\'s open — based on the prior business day\'s close</span>')
    elif stt == "holiday":
        # 주의(ko): '휴장일 →' 문자열은 서버 휴장 판정 grep이 사용
        asof_tag = ('<span class="tag warn">오늘 ' + a["today"] + ' (' + fdow(a["today_dow"])
                    + ')은 휴장일 → 직전 영업일 종가 기준</span>') if ko else (
            '<span class="tag warn">Today ' + a["today"] + ' (' + fdow(a["today_dow"])
            + ') is a market holiday — based on the prior business day\'s close</span>')
    elif stt == "intraday":
        asof_tag = ('<span class="tag warn">장중 실행 → 오늘 데이터는 미완성 장중 가격 (종가 아님)</span>' if ko else
                    '<span class="tag warn">Generated intraday — today\'s data is an incomplete intraday price (not the close)</span>')
    elif stt == "stale":
        # 주의(ko): 이 문구에 '휴장일 →'이 들어가면 안 됨 (서버 휴장 판정 grep과 충돌)
        asof_tag = ('<span class="tag warn">데이터 지연 → 직전 영업일 종가 기준</span>' if ko else
                    '<span class="tag warn">Data delayed — based on the prior business day\'s close</span>')
    else:
        asof_tag = ''  # 마감 후 실행: 오늘자 확정 종가가 반영된 정상 상태
    mode_tag = ('<span class="tag mock">' + ('모의 데이터' if ko else 'Mock data') + '</span>'
                if payload["mode"] == "demo" else '')

    head = ('<header><div class="eyebrow mono">Morning Brief · v2.6 · '
            + ('Mock' if payload["mode"] == "demo" else 'Live') + ' Data</div>'
            + "<h1>Jaeyoung Cho's Morning Brief</h1>"
            + '<div class="sub">'
            + ('구독자를 위한 데일리 시장 브리핑 · P1~P9 매매 원칙 기반' if ko else
               'A daily market brief for subscribers · based on trading principles P1–P9') + '</div>'
            + '<div class="asof"><span class="tag">'
            + ('기준일 ' if ko else 'As of ') + '<b>' + a["equity"] + ' (' + fdow(a["equity_dow"]) + ')</b> '
            + (('장중' if stt == "intraday" else '종가') if ko else
               ('intraday' if stt == "intraday" else 'close')) + '</span>'
            + asof_tag
            + '<span class="tag">' + ('BTC 기준 ' if ko else 'BTC as of ')
            + '<b>' + a["btc"] + ' (' + fdow(a["btc_dow"]) + ')</b> · '
            + ('24시간 거래라 주말·휴일에도 최신' if ko else 'trades 24/7, so current even on weekends/holidays')
            + '</span>' + mode_tag + '</div></header>')

    panel = _overall_panel(payload["tfs"], tx["combo"], tx.get("digest") or [], lang)

    grid = ('<div><div class="eyebrow mono" style="margin-bottom:8px">'
            + ('개별 신호 상세 (참고 자료) · 평가는 주간·월간 흐름까지 반영' if ko else
               'Signal Details (reference) · assessments reflect weekly & monthly flows too') + '</div>'
            + '<div class="grid">' + ''.join(_card(s, lang) for s in tx["signals"]) + '</div></div>')

    nrows = ''
    for i, nw in enumerate(payload.get("news") or [], 1):
        t = _esc(nw.get("title", ""))
        link = nw.get("link") or ""
        ttl = ('<a href="' + _esc(link) + '" target="_blank" rel="noopener">' + t + '</a>') if link else t
        meta = ' · '.join(x for x in (_esc(nw.get("publisher", "")), _esc(nw.get("ts", ""))) if x)
        nrows += ('<div class="nrow"><span class="nno mono">' + str(i) + '</span>'
                  + '<span class="ntt">' + ttl + '</span>'
                  + '<span class="nmeta mono">' + meta + '</span></div>')
    news = ('<div class="panel"><div class="eyebrow mono">'
            + ('P8 · 주요 시장 뉴스 Top 5' if ko else 'P8 · Top 5 Market News') + '</div>'
            + (nrows if nrows else '<div class="note">'
               + ('뉴스 조회 실패 — 네트워크 확인 후 다시 실행' if ko else
                  'News lookup failed — check the network and rerun') + '</div>')
            + '<div class="note">'
            + ('기준: 선거·전쟁·미중 갈등 등 대형 이벤트는 과거 유사 패턴을 따르는 경우가 많습니다(P8). 해석은 구독자 판단이며 점수에는 포함하지 않습니다.' if ko else
               'Rule: big events — elections, wars, US-China tension — often follow past patterns (P8). Interpretation is up to the reader and is not scored.')
            + '</div></div>')

    foot = ('<footer class="panel"><div class="foot">'
            + ('종합점수(-100~+100) = Σ(신호 상태점수 × 가중치) ÷ Σ가중치이며, ±45 밴드와 최근 방향으로 판정합니다. '
               '주간·월간 점수는 타임프레임 연동 신호(P1·P2·P4·P9)로 계산하며, 세 주기의 시차 자체가 정보입니다. '
               '임계값과 가중치는 자리표시자로, 실데이터 백테스트로 보정이 필요합니다. '
               '국면 문구와 권장 행동은 combo_advice()에서 수정합니다. '
               'CNN Fear &amp; Greed 조회 실패 시 해당 신호는 일간 점수에서 자동 제외됩니다. '
               '생성 시각: ' if ko else
               'Composite score (−100 to +100) = Σ(state score × weight) ÷ Σweights, judged with a ±45 band and the recent direction. '
               'Weekly/monthly scores use the timeframe-linked signals (P1·P2·P4·P9); the lag between the three horizons is itself information. '
               'Thresholds and weights are placeholders pending a backtest on real data. '
               'Regime wording and suggested actions are edited in combo_advice(). '
               'If the CNN Fear &amp; Greed lookup fails, that signal is dropped from the daily score automatically. '
               'Generated: ')
            + payload["generated"] + '</div></footer>')

    return head + panel + _cal_panel(payload, lang) + news + grid + foot


def render_html(payload):
    toggle = '<input type="checkbox" id="lang-sw" class="lang-sw" aria-label="한국어 / English">'
    btn = ('<label for="lang-sw" class="lang-btn mono">'
           '<span class="opt opt-ko">한국어</span><span class="opt opt-en">English</span></label>')
    return ('<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">'
            + '<meta name="viewport" content="width=device-width, initial-scale=1">'
            + "<title>Jaeyoung Cho's Morning Brief</title><style>" + _CSS + '</style></head><body>'
            + toggle
            + '<div class="wrap">' + btn
            + '<div class="lang lang-ko">' + _render_body(payload, "ko") + '</div>'
            + '<div class="lang lang-en">' + _render_body(payload, "en") + '</div>'
            + '</div></body></html>')


# ============================================================
# 메인
# ============================================================
def jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return round(float(x), 4)
    if isinstance(x, float):
        return round(x, 4)
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def build_payload(demo=False, fg=None):
    """데이터 수집 → 신호 계산 → payload(dict) 반환. 대시보드·이메일 공용."""
    data = make_demo() if demo else fetch_real(fg)

    # ---- 기준일 / 휴장일 판정 ----
    now_et = now_eastern()
    today = now_et.date()
    eq_asof = data["spy"].index[-1].date()
    btc_asof = data["btc"].index[-1].date()
    # 데이터가 오늘자가 아니면 사유를 구분: 주말 / 개장 전(평일 아침) / 휴장일
    if eq_asof >= today:
        # 오늘자 봉이 있음: 장중이면 아직 미완성 봉(종가 아님), 마감 후면 확정 종가
        if today.weekday() < 5 and (9, 30) <= (now_et.hour, now_et.minute) < (16, 0):
            mkt_status = "intraday"
        else:
            mkt_status = "current"
    elif today.weekday() >= 5:
        mkt_status = "weekend"
    elif (now_et.hour, now_et.minute) < (9, 30):
        mkt_status = "pre_open"
    elif str(today) in US_MARKET_HOLIDAYS:
        mkt_status = "holiday"
    else:
        # 평일 장 시작 후인데 오늘자 봉이 없고 휴장일 목록에도 없음
        # → 야후 응답 지연으로 간주 (휴장일로 오판해 갱신·메일을 막지 않도록 구분)
        mkt_status = "stale"
        print("[경고] 개장 시간인데 오늘자 봉 없음 → 데이터 지연으로 처리 "
              "(실제 휴장일이면 US_MARKET_HOLIDAYS에 날짜를 추가하세요)")

    tfs, mets, states = {}, {}, {}
    for tf in ("daily", "weekly", "monthly"):
        m_now = build_metrics(data, tf, cut=0)
        m_prev = build_metrics(data, tf, cut=CONFIG["trend_cut"][tf])
        st_now, st_prev = assess(m_now), assess(m_prev)
        if tf != "daily":
            for k in DAILY_ONLY:
                st_now.pop(k, None)
                st_prev.pop(k, None)
        fg_missing = m_now["fg"] is None
        sc_now = composite(st_now, fg_missing)
        sc_prev = composite(st_prev, fg_missing)
        dsc = sc_now - sc_prev
        trend = 1 if dsc > CONFIG["trend_eps"] else -1 if dsc < -CONFIG["trend_eps"] else 0
        tfs[tf] = {"score": sc_now, "trend": trend, "overall": overall(sc_now, trend)}
        mets[tf], states[tf] = m_now, st_now
        print(f"[{TF_LABEL['ko'][tf]}] 종합 {sc_now*100:+.1f} → {tfs[tf]['overall']}"
              f"  (방향 {trend:+d}, 상태: " +
              ", ".join(f"{k}:{v}" for k, v in st_now.items()) + ")")

    mo, wk, dy = (tfs[t]["overall"] for t in ("monthly", "weekly", "daily"))
    texts = {}
    for lg in LANGS:  # 한국어·영어 두 벌 생성 → HTML 토글로 전환
        tx = brief_texts(mets["daily"], mets["weekly"], mets["monthly"], lg)
        cname, caction, ctone = combo_advice(mo, wk, dy, lg)
        texts[lg] = {
            "signals": [{**tx[k], "id": k, "weight": WEIGHTS[k], "state": states["daily"][k]}
                        for k in states["daily"]],
            "digest": plain_digest(mets["daily"], states["daily"], lg),
            "combo": {"mo": mo, "wk": wk, "dy": dy,
                      "name": cname, "action": caction, "tone": ctone},
        }
    ck = texts["ko"]["combo"]
    print(f"[통합] 월간 {mo} · 주간 {wk} · 일간 {dy} → {ck['name']}: {ck['action']}")

    payload = jsonable({
        "mode": data["mode"],
        "news": data.get("news", []),
        "shorts": data.get("shorts", []),
        "calendar": market_calendar(today),
        "asof": {
            "equity": str(eq_asof), "equity_dow": eq_asof.weekday(),
            "today": str(today), "today_dow": today.weekday(),
            "status": mkt_status,
            "btc": str(btc_asof), "btc_dow": btc_asof.weekday(),
        },
        "tfs": tfs, "texts": texts, "states": ST,
        "generated": now_et.strftime("%Y-%m-%d %H:%M ET"),
    })
    note = {"current": "",
            "intraday": " — 장중 실행, 오늘 데이터는 미완성 장중 가격(종가 아님)",
            "weekend": " — 주말 휴장, 직전 영업일 종가 기준",
            "pre_open": " — 개장 전, 직전 영업일 종가 기준",
            "holiday": " — 휴장일, 직전 영업일 종가 기준",
            "stale": " — 데이터 지연, 직전 영업일 종가 기준"}[mkt_status]
    print(f"\n기준일: {eq_asof} ({DOW_KR[eq_asof.weekday()]})" + note)
    return payload


# ============================================================
# 가족 공유 — GitHub Pages 자동 게시
#   최초 1회 설정(터미널): gh auth login 후 저장소 생성·push (설정돼 있으면 자동 동작)
#   게시 주소: https://<GitHub아이디>.github.io/market-brief/
# ============================================================
PAGES_REPO = r"C:\Users\CJY_Laptop\market-brief"   # 게시용 git 저장소 폴더


def publish(src="market_dashboard.html"):
    """대시보드를 GitHub Pages 저장소에 index.html로 복사한 뒤 commit·push."""
    import os, shutil, subprocess
    if not os.path.isdir(os.path.join(PAGES_REPO, ".git")):
        print(f"[안내] 웹 게시 생략 — {PAGES_REPO} 저장소가 없습니다 (최초 설정 필요)")
        return False
    shutil.copy(src, os.path.join(PAGES_REPO, "index.html"))
    for cmd in (["git", "add", "-A"],
                ["git", "commit", "-m",
                 "dashboard update " + datetime.now().strftime("%Y-%m-%d %H:%M")],
                # 서버 자동 갱신과 충돌 시 방금 만든 로컬 파일 우선 (-X theirs = rebase에서 로컬 커밋 편)
                ["git", "pull", "--rebase", "-X", "theirs", "--quiet"],
                ["git", "push"]):
        r = subprocess.run(cmd, cwd=PAGES_REPO, capture_output=True, text=True)
        if r.returncode != 0:
            msg = ((r.stdout or "") + (r.stderr or "")).strip()
            if cmd[1] == "commit" and "nothing to commit" in msg:
                print("[안내] 대시보드 변경 없음 — 웹 게시 생략")
                return True
            print(f"[경고] 웹 게시 실패 ({' '.join(cmd)}): {msg[:300]}")
            return False
    print("웹 게시 완료 → 1~2분 뒤 사이트에 반영됩니다")
    return True


def run(demo=False, fg=None, out="market_dashboard.html", show=False, web=True):
    """대시보드 HTML 생성·저장. Jupyter: run(show=True) 등. web=True면 GitHub Pages에도 자동 게시."""
    payload = build_payload(demo, fg)
    with open(out, "w", encoding="utf-8") as f:
        f.write(render_html(payload))
    print(f"저장 완료 → {out}")
    if web and not demo:
        publish(out)
    if show:
        try:
            import html as _html
            from IPython.display import HTML, display
            doc = open(out, encoding="utf-8").read()
            display(HTML('<iframe srcdoc="' + _html.escape(doc, quote=True)
                         + '" style="width:100%;height:900px;border:0;background:#0b1220"></iframe>'))
        except Exception as e:
            print(f"[안내] 노트북 내 표시 실패({e}) → 브라우저에서 {out} 파일을 직접 여세요")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="모의 데이터로 실행")
    ap.add_argument("--fg", type=int, default=None, help="Fear & Greed 수동 입력")
    ap.add_argument("-o", "--out", default="market_dashboard.html")
    # Jupyter 커널이 넘기는 -f kernel-xxx.json 같은 미지 인자는 무시
    args, _unknown = ap.parse_known_args()
    run(demo=args.demo, fg=args.fg, out=args.out)


if __name__ == "__main__":
    main()