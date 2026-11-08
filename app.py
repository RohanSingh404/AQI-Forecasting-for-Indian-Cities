import streamlit as st, pandas as pd, numpy as np, joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="India AQI Forecast", layout="wide")

CATS = [
    (50,  "Good",         "#009865", "Air quality is fine. Enjoy outdoor activities."),
    (100, "Satisfactory", "#8fc31f", "Sensitive people may feel minor breathing discomfort."),
    (200, "Moderate",     "#f0c20f", "People with asthma, heart or lung disease should limit long outdoor exertion."),
    (300, "Poor",         "#ff7f27", "Avoid prolonged outdoor activity. Sensitive groups should stay indoors."),
    (400, "Very Poor",    "#e53935", "Avoid outdoor exertion. Wear an N95 mask if you go out."),
    (500, "Severe",       "#7b1fa2", "Health emergency. Stay indoors and use an air purifier if possible."),
]
TRAIN_END = pd.Timestamp("2019-12-24")

BP = {
 "PM2.5": [(0,30,0,50),(30,60,50,100),(60,90,100,200),(90,120,200,300),(120,250,300,400),(250,380,400,500)],
 "PM10":  [(0,50,0,50),(50,100,50,100),(100,250,100,200),(250,350,200,300),(350,430,300,400),(430,510,400,500)],
 "NO2":   [(0,40,0,50),(40,80,50,100),(80,180,100,200),(180,280,200,300),(280,400,300,400),(400,500,400,500)],
 "SO2":   [(0,40,0,50),(40,80,50,100),(80,380,100,200),(380,800,200,300),(800,1600,300,400),(1600,2000,400,500)],
 "CO":    [(0,1,0,50),(1,2,50,100),(2,10,100,200),(10,17,200,300),(17,34,300,400),(34,50,400,500)],
 "O3":    [(0,50,0,50),(50,100,50,100),(100,168,100,200),(168,208,200,300),(208,748,300,400),(748,1000,400,500)],
 "NH3":   [(0,200,0,50),(200,400,50,100),(400,800,100,200),(800,1200,200,300),(1200,1800,300,400),(1800,2400,400,500)],
}
NAMES = {"PM2.5":"PM2.5","PM10":"PM10","NO2":"NO₂","SO2":"SO₂","CO":"CO","O3":"O₃","NH3":"NH₃"}

def categorize(a):
    for hi, name, color, advice in CATS:
        if a <= hi:
            return name, color, advice
    return CATS[-1][1:]

def sub_index(x, bands):
    if pd.isna(x):
        return np.nan
    for clo, chi, ilo, ihi in bands:
        if x <= chi:
            return ilo + (max(x, clo) - clo) * (ihi - ilo) / (chi - clo)
    return 500.0

@st.cache_resource
def load_models():
    return joblib.load("artifacts/aqi_models.joblib")

@st.cache_data
def load_data():
    fe = pd.read_csv("artifacts/features.csv", parse_dates=["Date"])
    hist = pd.read_csv("artifacts/history.csv", parse_dates=["Date"])
    return fe, hist

bundle = load_models()
feats = bundle["feats"]
fe, hist = load_data()

st.title("India AQI Forecast")
st.caption("Pick a city and a start date to see the next 7 days of forecast AQI with health advice.")

city = st.sidebar.selectbox("City", sorted(fe["City"].unique()))
valid = fe[fe["City"] == city].dropna(subset=["AQI_lag14", "AQI_roll30"])
dmin, dmax = valid["Date"].min(), valid["Date"].max()
sel = st.sidebar.date_input("Forecast from (date)", value=dmax.date(),
                            min_value=dmin.date(), max_value=dmax.date())
sel = pd.Timestamp(sel)

row = valid[valid["Date"] == sel]
if row.empty:
    st.warning("No record for this date (gap in the data). Try a nearby date.")
    st.stop()

live = sel == dmax
models = bundle["models"] if live else bundle["holdout"]
X = row[feats].astype(float)
base = float(row["AQI"].iloc[0])

preds = [float(np.clip(base + models[h].predict(X)[0], 0, 500)) for h in range(1, 8)]
dates = pd.date_range(sel + pd.Timedelta(days=1), periods=7)
fc = pd.DataFrame({"Date": dates, "Forecast": np.round(preds).astype(int)})
fc["Category"] = fc["Forecast"].apply(lambda a: categorize(a)[0])
series = hist[hist["City"] == city].set_index("Date")["AQI"]
fc["Actual"] = fc["Date"].map(series)

if live:
    st.info("Latest date in the dataset (1 Jul 2020): this is a true forecast, so there are no actual values yet.")
elif sel <= TRAIN_END:
    st.warning("This date falls in the model's training period, so the forecast is in-sample and looks better than it would on new data.")
else:
    st.success("This date is after the training period (Dec 2019), so the forecast is out-of-sample.")

c1, c2, c3 = st.columns(3)
c1.metric(f"Observed on {sel:%d %b %Y}", int(base))
c2.metric("Next day (forecast)", int(fc["Forecast"][0]))
c3.metric("Day 7 (forecast)", int(fc["Forecast"][6]))

name, color, advice = categorize(fc["Forecast"][0])
st.markdown(f"<div style='padding:14px;border-radius:8px;background:{color};color:white'>"
            f"<b>Next day: {name}</b><br>{advice}</div>", unsafe_allow_html=True)

# ---- main pollutant driver (observed day) ----
r = row.iloc[0]
sub = pd.Series({p: sub_index(r[p], BP[p]) for p in BP}).dropna().sort_values(ascending=False)
st.subheader(f"What is driving AQI on {sel:%d %b %Y}")
if len(sub):
    main = sub.index[0]
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    order = list(sub.index)[::-1]
    ax2.barh([NAMES[p] for p in order], [sub[p] for p in order],
             color=["crimson" if p == main else "lightgray" for p in order])
    ax2.set_xlabel("Estimated sub-index (the highest one sets the AQI)")
    st.pyplot(fig2)
    others = ", ".join(NAMES[p] for p in sub.index[1:])
    st.markdown(f"**Main driver: {NAMES[main]}** (sub-index about {sub.iloc[0]:.0f}, "
                f"{categorize(sub.iloc[0])[0]}). Also considered: {others}.")
    missing = [NAMES[p] for p in BP if pd.isna(r[p])]
    if missing:
        st.caption("Not recorded on this day: " + ", ".join(missing))
    st.caption("Estimated from the day's concentrations with CPCB breakpoints. The forecast covers overall AQI; "
               "this chart explains the observed day, not the forecast days.")
else:
    st.info("No pollutant readings on this day.")

# ---- forecast chart ----
past = series[(series.index > sel - pd.Timedelta(days=90)) & (series.index <= sel)]
have = fc.dropna(subset=["Actual"])
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(past.index, past.values, label="Observed (previous 90 days)")
ax.plot(fc["Date"], fc["Forecast"], "o--", color="crimson", label="Forecast")
if not have.empty:
    ax.plot(have["Date"], have["Actual"], "s-", color="green", label="Actual")
for hi, _, col, _ in CATS:
    ax.axhline(hi, color=col, alpha=0.25, lw=1)
top = max(150, past.max() if len(past) else 0, fc["Forecast"].max(),
          have["Actual"].max() if not have.empty else 0)
ax.set_ylim(0, top * 1.15)
ax.set_ylabel("AQI"); ax.legend(); ax.grid(alpha=0.2)
st.pyplot(fig)

if not have.empty:
    mae = (have["Forecast"] - have["Actual"]).abs().mean()
    st.metric(f"Mean absolute error over {len(have)} day(s) with actual data", f"{mae:.1f} AQI points")

show = fc.assign(Date=fc["Date"].dt.strftime("%a %d %b %Y"))
if show["Actual"].notna().any():
    show["Actual"] = show["Actual"].round(0).astype("Int64")
    show["Error"] = (show["Forecast"] - show["Actual"]).astype("Int64")
else:
    show = show.drop(columns=["Actual"])
st.subheader("7-day forecast")
st.dataframe(show, hide_index=True, use_container_width=True)

# ---- backtest ----
st.subheader("Backtest: forecast vs actual (Jan to Jul 2020)")
hz = st.selectbox("Horizon (days ahead)", list(range(1, 8)), index=2)
t = valid[valid["Date"] >= "2020-01-01"].copy()
t["Pred"] = np.clip(t["AQI"] + bundle["holdout"][hz].predict(t[feats].astype(float)), 0, 500)
t["Target_date"] = t["Date"] + pd.Timedelta(days=hz)
t["Actual"] = t["Target_date"].map(series)
t = t.dropna(subset=["Actual"])
if len(t) > 5:
    fig3, ax3 = plt.subplots(figsize=(10, 3.5))
    ax3.plot(t["Target_date"], t["Actual"], label="Actual")
    ax3.plot(t["Target_date"], t["Pred"], color="crimson", label=f"Forecast ({hz} days ahead)")
    ax3.set_ylabel("AQI"); ax3.legend(); ax3.grid(alpha=0.2)
    st.pyplot(fig3)
    a, p = t["Actual"], t["Pred"]
    m1, m2 = st.columns(2)
    m1.metric("MAE", f"{(a - p).abs().mean():.1f}")
    m2.metric("R²", f"{1 - ((a - p) ** 2).sum() / ((a - a.mean()) ** 2).sum():.2f}")
else:
    st.info("Not enough 2020 data for this city.")

st.info("Forecast error grows with horizon (about R² 0.9 at 1 day, about 0.6 at 3 to 7 days, pre-lockdown). "
        "Jaipur and Kolkata have shorter training history.")