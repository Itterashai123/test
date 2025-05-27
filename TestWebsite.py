import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time
import requests
from bs4 import BeautifulSoup

# Global untuk hasil
results = []
kondisi_table = []
condition_names = [
    "Breakout",
    "Candle Bullish",
    "Volume Up",
    "MACD Cross",
    "MACD Positive",
    "RSI < 60",
    "Trend Up",
    "ADX > 20",           # Kondisi ke-8
    "Volume 2 Hari Naik"  # Kondisi ke-9
]

    
def get_berita_kontan(ticker_nama, max_berita=5):
    url = f"https://www.kontan.co.id/search/?search={ticker_nama}&Button_search="
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers)
        soup = BeautifulSoup(resp.text, 'html.parser')
        hasil = []
        # Cari semua <li> di dalam <div class="list-berita">
        for item in soup.select('div.list-berita ul li')[:max_berita]:
            # Judul dan link
            judul_tag = item.select_one('div.sp-hl h1 a')
            judul = judul_tag.text.strip() if judul_tag else "Tanpa Judul"
            link = judul_tag['href'] if judul_tag else "#"
            # Waktu (misal: | 8 Jam 31 Menit lalu)
            waktu_tag = item.select_one('div.fs14 span.font-gray')
            waktu = waktu_tag.text.strip() if waktu_tag else ""
            hasil.append((judul, link, waktu))
        return hasil
    except Exception as e:
        return [("Gagal mengambil berita", "#", str(e))]

def get_berita_investing(ticker_nama, max_berita=5):
    """
    Ambil berita terbaru dari Investing.com Indonesia berdasarkan nama saham.
    """
    import requests
    from bs4 import BeautifulSoup

    query = ticker_nama.replace('.JK', '').replace(' ', '%20')
    url = f"https://id.investing.com/search/?q={query}&tab=news"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": "https://id.investing.com/",
        "Connection": "keep-alive"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        hasil = []
        for item in soup.select('div.js-search-results-news article')[:max_berita]:
            judul_tag = item.select_one('a.title')
            judul = judul_tag.text.strip() if judul_tag else "Tanpa Judul"
            link = "https://id.investing.com" + judul_tag['href'] if judul_tag else "#"
            waktu_tag = item.select_one('span.date')
            waktu = waktu_tag.text.strip() if waktu_tag else ""
            hasil.append((judul, link, waktu))
        return hasil
    except Exception as e:
        return [("Gagal mengambil berita Investing.com", "#", str(e))]

@st.cache_data(ttl=3600)
def calculate_macd(data):
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

@st.cache_data(ttl=3600)
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

@st.cache_data(ttl=3600)
def calculate_bollinger_bands(data, window=20, no_of_std=2):
    rolling_mean = data['Close'].rolling(window).mean()
    rolling_std = data['Close'].rolling(window).std()
    upper_band = rolling_mean + (rolling_std * no_of_std)
    lower_band = rolling_mean - (rolling_std * no_of_std)
    return upper_band, rolling_mean, lower_band

@st.cache_data(ttl=3600)
def calculate_atr(data, window=14):
    high_low = data['High'] - data['Low']
    high_close_prev = (data['High'] - data['Close'].shift(1)).abs()
    low_close_prev = (data['Low'] - data['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr

@st.cache_data(ttl=3600)
def get_stock_data(ticker):
    data = yf.download(ticker, period='1y', interval='1d', group_by=None)
    # Jika kolom MultiIndex, ambil hanya kolom level kedua (Price)
    if isinstance(data.columns, pd.MultiIndex):
        data = data[ticker]  # Ambil kolom level kedua: 'Open', 'High', dst
    return data

def check_data_coverage(data, ticker):
    if data.empty:
        st.write(f"{ticker}: ❌ Data kosong.")
        return
    first_date = data.index.min()
    last_date = data.index.max()
    day_span = (last_date - first_date).days
    if day_span < 50:
        st.write(f"{ticker}: ⚠️ Hanya mencakup {day_span} hari kalender ({len(data)} hari bursa).")
    else:
        st.write(f"{ticker}: ✅ {day_span} hari kalender, {len(data)} hari bursa.")


@st.cache_data(ttl=3600)
def load_idx_tickers():
    try:
        df = pd.read_csv('idx_stocks.csv')
        if 'Ticker' not in df.columns:
            st.error("CSV tidak memiliki kolom 'Ticker'")
            return []
        return df['Ticker'].dropna().unique().tolist()
    except FileNotFoundError:
        if data.empty:
            st.write(f"{ticker}: ❌ Data kosong.")
            return
        first_date = data.index.min()
        last_date = data.index.max()
        day_span = (last_date - first_date).days
        if day_span < 50:
            st.write(f"{ticker}: ⚠️ Hanya mencakup {day_span} hari kalender ({len(data)} hari bursa).")
        else:
            st.write(f"{ticker}: ✅ {day_span} hari kalender, {len(data)} hari bursa.")
        
def calculate_adx(data, window=14):
    high = data['High']
    low = data['Low']
    close = data['Close']
    plus_dm = high.diff()
    minus_dm = low.diff().abs()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1/window).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/window).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/window).mean()
    return adx

def analyze_stock(ticker, min_conditions, min_atr=10.0):
    data = get_stock_data(ticker)
    if data.empty or len(data) < 50:
        return None

    macd, signal = calculate_macd(data)
    rsi = calculate_rsi(data)
    upper_band, middle_band, lower_band = calculate_bollinger_bands(data)
    atr = calculate_atr(data)
    adx = calculate_adx(data)

    data['MACD'] = macd
    data['Signal'] = signal
    data['RSI'] = rsi
    data['UpperBB'] = upper_band
    data['LowerBB'] = lower_band
    data['ATR'] = atr
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['VolumeMA20'] = data['Volume'].rolling(window=20).mean()
    data['ADX'] = adx

    # Filter ATR minimum
    if data['ATR'].iloc[-1] < min_atr:
        return None

    # Filter harga dan volume
    close_today = data['Close'].iloc[-1]
    avg_volume = data['Volume'].rolling(5).mean().iloc[-1]
    if not (100 < close_today < 10000):
        return None
    if avg_volume < 1_000_000:
        return None

    # --- Tambahkan filter breakout volume ---
    breakout_volume = data['Volume'].iloc[-1] > 1.5 * data['VolumeMA20'].iloc[-1]
    if not breakout_volume:
        return None
    # ---------------------------------------

    i = len(data) - 1
    window = 50
    if i + 1 < window:
        resistance = data['High'].iloc[:i+1].max()
    else:
        resistance = data['High'].iloc[i-window+1:i+1].max()

    close_today = data['Close'].iloc[i]
    open_today = data['Open'].iloc[i]
    volume_today = data['Volume'].iloc[i]
    volume_ma5 = data['Volume'].rolling(5).mean().iloc[i]
    sma50_today = data['SMA50'].iloc[i]
    adx_today = data['ADX'].iloc[i]

    body = abs(close_today - open_today)
    candle_range = data['High'].iloc[i] - data['Low'].iloc[i]
    mid_range = data['Low'].iloc[i] + 0.5 * candle_range

    candle_bullish = (
        (close_today > open_today) and
        (close_today > mid_range) and
        (body > 0.5 * candle_range)
    )

    if pd.isna(resistance):
        breakout = False
    else:
        if i >= 1:
            prev_close = data['Close'].iloc[i-1]
            breakout = (close_today >= resistance * 1.01) and (prev_close >= resistance * 1.005)
        else:
            breakout = close_today >= resistance * 1.01

    macd_today = data['MACD'].iloc[i]
    signal_today = data['Signal'].iloc[i]
    macd_positive = macd_today > 0 and signal_today > 0
    macd_cross = macd_today > signal_today
    rsi_today = data['RSI'].iloc[i]
    rsi_ok = rsi_today < 60
    trend_up = False if pd.isna(sma50_today) else close_today > sma50_today
    volume_up = False if pd.isna(volume_ma5) else volume_today > volume_ma5 * 1.5

    # Tambahkan ADX sebagai kondisi ke-8
    adx_ok = adx_today > 20

    # Konfirmasi volume naik 2 hari berturut-turut
    volume_up_2days = (
        (i >= 2) and
        (data['Volume'].iloc[i] > data['Volume'].iloc[i-1]) and
        (data['Volume'].iloc[i-1] > data['Volume'].iloc[i-2])
    )

    # Tambahkan ke daftar kondisi
    conditions = [
        breakout,           # 1
        candle_bullish,     # 2
        volume_up,          # 3
        macd_cross,         # 4
        macd_positive,      # 5
        rsi_ok,             # 6
        trend_up,           # 7
        adx_ok,             # 8
        volume_up_2days     # 9
    ]
    # Jika ingin volume_up_2days opsional, jangan tambahkan ke condition_names

    total_conditions = sum(conditions)
    kondisi_table.append({
        "Ticker": ticker,
        **{name: cond for name, cond in zip(condition_names, conditions)},
        "Total": total_conditions
    })

    with st.expander(f"Detail {ticker}"):
        st.write(f"{ticker}: {total_conditions} kondisi terpenuhi.")
        check_data_coverage(data, ticker)
        st.dataframe(data.tail())

    if total_conditions >= min_conditions:
        return {
            'Ticker': ticker,
            'Date': data.index[i].strftime('%Y-%m-%d'),
            'Close': close_today,
            'Conditions Met': total_conditions,
            'Breakout': breakout,
            'Volume Up': volume_up,
            'MACD Cross': macd_cross,
            'RSI': rsi_today,
            'Trend Up': trend_up,
            'Resistance': resistance,
            'SMA50': sma50_today,
            'Data': data
        }
    else:
        return None

def find_swing_high_low(data, window=20):
    data['SwingHigh'] = data['High'][(data['High'] == data['High'].rolling(window=window, center=True).max())]
    data['SwingLow'] = data['Low'][(data['Low'] == data['Low'].rolling(window=window, center=True).min())]
    return data

def detect_swing_signals(data):
    data = find_swing_high_low(data)
    buy_signals = []
    sell_signals = []

    for i in range(20, len(data)):
        today_close = data['Close'].iloc[i]
        today_volume = data['Volume'].iloc[i]
        avg_volume = data['Volume'].iloc[i-5:i].mean()
        today_rsi = data['RSI'].iloc[i]
        macd = data['MACD'].iloc[i]
        signal = data['Signal'].iloc[i]

        prev_swing_high = data['SwingHigh'].iloc[i-20:i].max()
        prev_swing_low = data['SwingLow'].iloc[i-20:i].min()

        # BUY signal
        if (
            today_close > prev_swing_high and
            today_volume > avg_volume and
            today_rsi < 65 and
            macd > signal
        ):
            buy_signals.append(data.index[i])
        else:
            buy_signals.append(None)

        # SELL signal
        if today_rsi > 70 or today_close < prev_swing_low:
            sell_signals.append(data.index[i])
        else:
            sell_signals.append(None)

    # Pad awal agar panjang sama
    data['BuySignal'] = [None]*20 + buy_signals
    data['SellSignal'] = [None]*20 + sell_signals
    return data

def rekomendasi_tp_sl(entry_price, atr, data=None, tp_atr=2, sl_atr=1, tp_pct=0.05, sl_pct=0.03):
    """
    Rekomendasi TP/SL berbasis ATR, persentase, dan support level (jika data diberikan).
    """
    tp_atr_val = entry_price + tp_atr * atr
    sl_atr_val = entry_price - sl_atr * atr
    tp_pct_val = entry_price * (1 + tp_pct)
    sl_pct_val = entry_price * (1 - sl_pct)

    # Pilih TP yang lebih konservatif (lebih dekat ke entry)
    tp_final = min(tp_atr_val, tp_pct_val)
    # Pilih SL yang lebih konservatif (lebih dekat ke entry)
    sl_final = max(sl_atr_val, sl_pct_val)

    # Jika data disediakan, cari support (Swing Low) terakhir
    support_level = None
    if data is not None and 'SwingLow' in data.columns:
        recent_supports = data['SwingLow'].dropna()
        if not recent_supports.empty:
            support_level = recent_supports.iloc[-1]
            # Jika support lebih dekat ke entry daripada SL ATR/persen, gunakan support
            if support_level > sl_final and support_level < entry_price:
                sl_final = support_level

    return {
        "TP (ATR)": round(tp_atr_val, 2),
        "SL (ATR)": round(sl_atr_val, 2),
        "TP (%)": round(tp_pct_val, 2),
        "SL (%)": round(sl_pct_val, 2),
        "TP Final": round(tp_final, 2),
        "SL Final": round(sl_final, 2),
        "Support SL": round(support_level, 2) if support_level else None
    }

def trailing_stop_exit(prices, atr, trailing_pct=0.03):
    price = prices.iloc[-1]
    dynamic_pct = min(trailing_pct, 0.5 * atr / price)
    highest = prices.cummax()
    trailing_stop = highest * (1 - dynamic_pct)
    exit_signal = prices < trailing_stop
    return exit_signal

def trailing_exit_conditions(data):
    """
    Return True jika salah satu exit condition terpenuhi:
    - Close < MA5
    - RSI menurun 3 hari berturut-turut
    - MACD histogram menurun 3 hari berturut-turut
    """
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MACD_Hist'] = data['MACD'] - data['Signal']

    close_below_ma5 = data['Close'].iloc[-1] < data['MA5'].iloc[-1]

    rsi_down = all(data['RSI'].iloc[-i] < data['RSI'].iloc[-i-1] for i in range(1, 4))
    macd_hist_down = all(data['MACD_Hist'].iloc[-i] < data['MACD_Hist'].iloc[-i-1] for i in range(1, 4))

    return close_below_ma5 or rsi_down or macd_hist_down

# UI
st.title("Sinyal Trading Semi-Otomatis IDX Full Scan")

st.markdown("""
Scan seluruh saham IDX dan tampilkan sinyal BUY yang memenuhi minimal 5 dari 9 kondisi berikut:
1. **Breakout resistance 50 hari +1%** (validasi 2 hari berturut-turut close di atas resistance)
2. **Candle bullish signifikan** (close > mid-range & body > 50% candle)
3. **Volume naik >1.5x rata-rata 5 hari**
4. **MACD cross** (MACD > Signal)
5. **MACD positif** (MACD > 0 dan Signal > 0)
6. **RSI < 60**
7. **Harga di atas SMA50**
8. **ADX > 20** (trend kuat)
9. **Volume naik 2 hari berturut-turut** (konfirmasi akumulasi)
""")

if st.button("Reset Cache"):
    st.cache_data.clear()

tickers = load_idx_tickers()
batch_size = st.slider("Jumlah saham yang ingin di-scan:", min_value=20, max_value=len(tickers), value=50)
min_conditions = st.slider("Minimal jumlah kondisi untuk sinyal BUY:", min_value=1, max_value=9, value=5)
selected_tickers = tickers[:batch_size]

with st.spinner(f"Mengambil dan memproses data {batch_size} saham..."):
    results.clear()
    kondisi_table.clear()
    for ticker in selected_tickers:
        try:
            res = analyze_stock(ticker, min_conditions)
            if res:
                results.append(res)
        except Exception as e:
            st.write(f"Error proses {ticker}: {e}")

# Tampilkan tabel kondisi dan tabel sinyal BUY di atas
if kondisi_table:
    st.subheader("Tabel Kondisi Saham (True = kondisi terpenuhi)")
    st.dataframe(pd.DataFrame(kondisi_table))
else:
    st.warning("Tidak ada saham yang berhasil diproses dalam batch ini.")

if results:
    df = pd.DataFrame(results).drop(columns=['Data'])
    st.subheader(f"Sinyal BUY ditemukan ({len(results)} saham)")
    st.dataframe(df)
    st.download_button(
        "Download hasil ke CSV", 
        data=df.to_csv(index=False), 
        file_name="sinyal_buy.csv", 
        mime='text/csv',
        key="download_no_tier"  # <-- Add unique key here
    )
# ...existing code...

# Detail saham & visualisasi
if results:
    selected_ticker = st.selectbox("Pilih saham untuk lihat grafik:", [r['Ticker'] for r in results])
    selected_data = next(r['Data'] for r in results if r['Ticker'] == selected_ticker)
    selected_resistance = next(r['Resistance'] for r in results if r['Ticker'] == selected_ticker)
    selected_sma50 = next(r['SMA50'] for r in results if r['Ticker'] == selected_ticker)

    # Grafik Harga dengan Sinyal Swing (harian)
    selected_data = detect_swing_signals(selected_data)
    buy_dates = selected_data[selected_data['BuySignal'].notna()]
    sell_dates = selected_data[selected_data['SellSignal'].notna()]

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(selected_data.index, selected_data['Close'], label='Close', color='blue')
    ax.axhline(selected_resistance, color='red', linestyle='--', label='Resistance 50 hari')
    ax.plot(selected_data.index, selected_data['SMA50'], label='SMA 50', color='orange')
    ax.scatter(buy_dates.index, selected_data.loc[buy_dates.index, 'Close'], label='Buy', color='green', marker='^', s=100)
    ax.scatter(sell_dates.index, selected_data.loc[sell_dates.index, 'Close'], label='Sell', color='red', marker='v', s=100)
    ax.set_title(f"Grafik Harga & Sinyal Swing {selected_ticker}")
    ax.legend()
    st.pyplot(fig)

    # --- Tambahan: Grafik Timeframe Lebih Kecil (1 Jam) ---
    st.markdown("### Grafik Intraday (1 Jam) untuk Entry Presisi")
    intraday_1h = yf.download(selected_ticker, period='5d', interval='1h')
    if not intraday_1h.empty:
        fig_1h, ax_1h = plt.subplots(figsize=(10,3))
        ax_1h.plot(intraday_1h.index, intraday_1h['Close'], label='Close 1H', color='blue')
        ax_1h.set_title(f"{selected_ticker} - Close 1 Jam (5 Hari Terakhir)")
        ax_1h.legend()
        st.pyplot(fig_1h)
    else:
        st.info("Data intraday 1 jam tidak tersedia untuk saham ini.")

    # --- Grafik Intraday 30 Menit ---
    st.markdown("### Grafik Intraday (30 Menit) untuk Entry Presisi")
    intraday_30m = yf.download(selected_ticker, period='5d', interval='30m')
    if not intraday_30m.empty:
        fig_30m, ax_30m = plt.subplots(figsize=(10,3))
        ax_30m.plot(intraday_30m.index, intraday_30m['Close'], label='Close 30m', color='green')
        ax_30m.set_title(f"{selected_ticker} - Close 30 Menit (5 Hari Terakhir)")
        ax_30m.legend()
        st.pyplot(fig_30m)
    else:
        st.info("Data intraday 30 menit tidak tersedia untuk saham ini.")

    # --- END Tambahan ---

    # Rekomendasi TP/SL otomatis
    selected_data = detect_swing_signals(selected_data)
    last_atr = selected_data['ATR'].iloc[-1]
    entry = selected_data['Close'].iloc[-1]
    tp_sl = rekomendasi_tp_sl(entry, last_atr, data=selected_data)
    st.markdown("### Rekomendasi Take Profit & Stop Loss (Otomatis, ATR & Support)")
    st.write(pd.DataFrame([tp_sl]))

    # Risk:Reward Ratio
    rr_atr = (tp_sl["TP (ATR)"] - entry) / (entry - tp_sl["SL (ATR)"]) if (entry - tp_sl["SL (ATR)"]) != 0 else None
    rr_pct = (tp_sl["TP (%)"] - entry) / (entry - tp_sl["SL (%)"]) if (entry - tp_sl["SL (%)"]) != 0 else None
    st.markdown("**Risk:Reward Ratio**")
    st.write(f"ATR-based: {rr_atr:.2f} | Persentase: {rr_pct:.2f}")

    # Grafik RSI
    fig2, ax2 = plt.subplots(figsize=(10,2))
    ax2.plot(selected_data.index, selected_data['RSI'], label='RSI', color='purple')
    ax2.axhline(70, color='red', linestyle='--', linewidth=1)
    ax2.axhline(30, color='green', linestyle='--', linewidth=1)
    ax2.set_title(f"RSI {selected_ticker}")
    st.pyplot(fig2)

    # Grafik MACD
    fig3, ax3 = plt.subplots(figsize=(10,2))
    ax3.plot(selected_data.index, selected_data['MACD'], label='MACD', color='blue')
    ax3.plot(selected_data.index, selected_data['Signal'], label='Signal', color='orange')
    ax3.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax3.set_title(f"MACD {selected_ticker}")
    ax3.legend()
    st.pyplot(fig3)

    # Trailing Stop Info
    last_atr = selected_data['ATR'].iloc[-1]
    trailing_signal = trailing_stop_exit(selected_data['Close'], last_atr)
    if trailing_signal.iloc[-1]:
        st.warning("⚠️ Trailing Stop Exit: Harga sudah turun >3% dari puncak terakhir!")
    else:
        st.success("Trailing Stop belum kena.")

    # --- Info GAP Open Hari Ini ---
    if len(selected_data) > 1:
        today_open = selected_data['Open'].iloc[-1]
        prev_close = selected_data['Close'].iloc[-2]
        gap = today_open - prev_close
        gap_pct = (gap / prev_close) * 100 if prev_close != 0 else 0
        if abs(gap_pct) > 0.5:  # threshold 0.5% untuk dianggap signifikan
            st.warning(f"GAP open hari ini: {gap:+.2f} ({gap_pct:+.2f}%)")
        else:
            st.info(f"Tidak ada GAP open signifikan hari ini. (Gap: {gap:+.2f}, {gap_pct:+.2f}%)")
    else:
        st.info("Data tidak cukup untuk menghitung GAP open hari ini.")
else:
    st.info("Tidak ada sinyal BUY hari ini di batch ini.")

def assign_detailed_tier_v2(row):
    conds = {
        'Breakout': row.get('Breakout', False),
        'Volume': row.get('Volume Up', False),
        'RSI': row.get('RSI', False),
        'MACD': row.get('MACD Cross', False),
        'MA': row.get('Trend Up', False),
        'Golden Cross': False  # Not used in your current logic, set as needed
    }
    total = row.get('Conditions Met', 0)

    has_breakout = conds.get('Breakout', False)
    has_volume = conds.get('Volume', False)
    has_momentum = conds.get('RSI', False) or conds.get('MACD', False)
    has_ma = conds.get('MA', False) or conds.get('Golden Cross', False)

    if total in (8, 9):
        if has_breakout and has_volume and has_ma:
            if has_momentum:
                return '🟢 A++ (Strong Buy – All Confirmed + Momentum)'
            else:
                return '🟢 A+ (Strong Buy – All Confirmed)'
        elif has_breakout and has_volume:
            return '🟢 A (Buy – Breakout Confirmed)'
        else:
            return '🟢 A− (Buy – Technical Fullset)'
    elif total in (7, 6):
        if has_breakout and has_volume:
            return '🟡 B+ (Buy – Strong Momentum)'
        elif has_breakout or has_volume:
            return '🟡 B (Buy – Moderate Momentum)'
        else:
            return '🟡 B− (Buy – Weak Confirmation)'
    elif total == 5:
        if has_breakout:
            return '🟠 C+ (Watchlist – 4 with Breakout)'
        elif has_volume or has_momentum:
            return '🟠 C (Watchlist – 4 Mix)'
        else:
            return '🟠 C− (Watchlist – 4 Lemah)'
    elif total == 4:
        if has_breakout:
            return '🔵 D+ (Hold – 3 Breakout)'
        if has_volume or has_ma or has_momentum:
            return '🔵 D+ (Hold – 3 Mix Bagus)'
        else:
            return '🔵 D (Hold – 3 Random)'
    elif total == 3:
        return '🔴 E+ (Avoid – 2 Kondisi Lemah)'
    elif total == 2:
        if has_breakout or has_volume:
            return '🔴 E (Avoid – 1 Breakout/Volume)'
        else:
            return '🔴 E (Avoid – 1 Sinyal Tunggal)'
    else:
        return '⚫ F (Ignore – No Signal)'

# Setelah membuat df dari results, tambahkan kolom Tier:
if results:
    df = pd.DataFrame(results).drop(columns=['Data'])
    df['Tier'] = df.apply(assign_detailed_tier_v2, axis=1)
    st.subheader(f"Sinyal BUY ditemukan ({len(results)} saham)")
    st.dataframe(df)
    st.download_button(
        "Download hasil ke CSV",
        data=df.to_csv(index=False),
        file_name="sinyal_buy.csv",
        mime='text/csv',
        key="download_with_tier"
    )

    # Trailing TP/Exit Condition
    if trailing_exit_conditions(selected_data):
        st.warning("⚠️ Exit Condition terpenuhi: Close < MA5, RSI/Histogram MACD menurun!")
    else:
        st.success("Belum ada sinyal exit (Trailing TP/Exit Condition).")

    # --- Berita Terkini Saham Ini (Google News/Kontan) ---
    st.markdown("### Berita Terkini Saham Ini (Kontan.co.id)")
    berita = get_berita_kontan(selected_ticker.replace('.JK', ''))
    if berita:
        for judul, link, waktu in berita:
            st.markdown(f"- [{judul}]({link})  \n<sub>{waktu}</sub>", unsafe_allow_html=True)
    else:
        st.info("Belum ada berita terbaru untuk saham ini.")

    # --- Berita Terkini Investing.com Indonesia ---
    st.markdown("### Berita Terkini Saham Ini (Investing.com Indonesia)")
    berita_investing = get_berita_investing(selected_ticker.replace('.JK', ''))
    if berita_investing:
        for judul, link, waktu in berita_investing:
            st.markdown(f"- [{judul}]({link})  \n<sub>{waktu}</sub>", unsafe_allow_html=True)
    else:
        st.info("Belum ada berita terbaru untuk saham ini di Investing.com.")
