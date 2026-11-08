import os
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
    extra_path = "artifacts/actuals_extra.csv"
    if os.path.exists(extra_path):
        extra = pd.read_csv(extra_path, parse_dates=["Date"])[["City", "Date", "AQI"]]
        hist = pd.concat([hist, extra], ignore_index=True)
    hist = hist.drop_duplicates(subset=["City", "Date"], keep="last")
    return fe, hist

bundle = load_models()
feats = bundle["feats"]
holdout = bundle.get("holdout")
fe, hist = load_data()

st.title("India AQI Forecast")
st.caption("Pick a city and a start date. The table compares the 7-day forecast with what actually happened.")

city = st.sidebar.selectbox("City", sorted(fe["City"].unique()))
valid = fe[fe["City"] == city].dropna(subset=["AQI_lag14", "AQI_roll30"])
series = hist[hist["City"] == city].set_index("Date")["AQI"]
dmin, dmax = valid["Date"].min(), valid["Date"].max()

# default date: the latest date if actuals exist after it, otherwise 30 days earlier
if series.index.max() > dmax:
    default = dmax
else:
    default = valid["Date"].iloc[-30] if len(valid) > 30 else dmax
sel = st.sidebar.date_input("Forecast from (date)", value=default.date(),
                            min_value=dmin.date(), max_value=dmax.date())
sel = pd.Timestamp(sel)

row = valid[valid["Date"] == sel]
if row.empty:
    st.warning("No record for this date (gap in the data). Try a nearby date.")
    st.stop()

live = sel == dmax
use_main = live or holdout is None
models = bundle["models"] if use_main else holdout
X = row[feats].astype(float)
base = float(row["AQI"].iloc[0])

preds = [float(np.clip(base + models[h].predict(X)[0], 0, 500)) for h in range(1, 8)]
dates = pd.date_range(sel + pd.Timedelta(days=1), periods=7)
fc = pd.DataFrame({"Date": dates, "Forecast": np.round(preds).astype(int)})
fc["Actual"] = fc["Date"].map(series)

if live:
    st.info("This is the last date in the dataset. Actual values appear only if you added them in artifacts/actuals_extra.csv.")
elif holdout is None or sel <= TRAIN_END:
    st.warning("The model saw this period during training, so the forecast looks better than it would on new data.")
else:
    st.success("This date is after the training period (Dec 2019), so the forecast is a fair test on unseen data.")

c1, c2, c3 = st.columns(3)
c1.metric(f"Observed on {sel:%d %b %Y}", int(base))
c2.metric("Next day (forecast)", int(fc["Forecast"][0]))
c3.metric("Day 7 (forecast)", int(fc["Forecast"][6]))

name, color, advice = categorize(fc["Forecast"][0])
st.markdown(f"<div style='padding:14px;border-radius:8px;background:{color};color:white'>"
            f"<b>Next day: {name}</b><br>{advice}</div>", unsafe_allow_html=True)

# ---- 7-day table: forecast vs actual ----
st.subheader("7-day forecast vs actual")
have = fc.dropna(subset=["Actual"]).copy()
disp = pd.DataFrame({
    "Date": fc["Date"].dt.strftime("%a %d %b %Y"),
    "Forecast AQI": fc["Forecast"],
    "Forecast category": fc["Forecast"].apply(lambda a: categorize(a)[0]),
    "Actual AQI": [str(int(round(a))) if pd.notna(a) else "not available" for a in fc["Actual"]],
    "Actual category": [categorize(a)[0] if pd.notna(a) else "-" for a in fc["Actual"]],
    "Difference (forecast - actual)": [f"{int(f) - int(round(a)):+d}" if pd.notna(a) else "-"
                                       for f, a in zip(fc["Forecast"], fc["Actual"])],
})
st.dataframe(disp, hide_index=True, use_container_width=True)

# ---- summary ----
st.subheader("Summary")
if have.empty:
    st.info("No actual values exist for these 7 days in the data, so there is nothing to compare yet. "
            "Pick an earlier date to see the comparison.")
else:
    diff = have["Forecast"] - have["Actual"]
    n = len(have)
    mae = diff.abs().mean()
    worst = diff.abs().idxmax()
    match = sum(categorize(f)[0] == categorize(a)[0] for f, a in zip(have["Forecast"], have["Actual"]))
    bias = diff.mean()
    lean = "higher" if bias > 0 else "lower"
    st.markdown(
        f"- Over **{n} day(s)** with actual data, the model forecast an average AQI of **{have['Forecast'].mean():.0f}**, "
        f"while the actual average was **{have['Actual'].mean():.0f}**.\n"
        f"- On average the forecast was off by **{mae:.1f} AQI points**"
        f"{f' and leaned {lean} than actual by {abs(bias):.1f}' if abs(bias) >= 1 else ''}.\n"
        f"- The biggest miss was on **{have.loc[worst, 'Date']:%d %b %Y}**: forecast {int(have.loc[worst, 'Forecast'])} "
        f"vs actual {int(round(have.loc[worst, 'Actual']))}.\n"
        f"- The forecast category matched the actual category on **{match} of {n}** day(s)."
    )

# ---- forecast chart ----
past = series[(series.index > sel - pd.Timedelta(days=90)) & (series.index <= sel)]
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
    st.caption("Estimated from the day's concentrations with CPCB breakpoints. "
               "This explains the observed day, not the forecast days.")
else:
    st.info("No pollutant readings on this day.")

st.info("Forecast error grows with horizon (about R² 0.9 at 1 day, about 0.6 at 3 to 7 days, pre-lockdown). "
        "Jaipur and Kolkata have shorter training history.")