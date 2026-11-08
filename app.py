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

def sub_index(x, bands):
    if pd.isna(x):
        return np.nan
    for clo, chi, ilo, ihi in bands:
        if x <= chi:
            return ilo + (max(x, clo) - clo) * (ihi - ilo) / (chi - clo)
    return 500.0

r = row.iloc[0]
sub = pd.Series({p: sub_index(r[p], BP[p]) for p in BP}).dropna().sort_values(ascending=False)
main = sub.index[0]

st.subheader(f"What is driving AQI on {sel:%d %b %Y}")
fig2, ax2 = plt.subplots(figsize=(8, 3))
order = list(sub.index)[::-1]
ax2.barh([NAMES[p] for p in order], [sub[p] for p in order],
         color=["crimson" if p == main else "lightgray" for p in order])
ax2.set_xlabel("Estimated sub-index (highest one sets the AQI)")
st.pyplot(fig2)

others = ", ".join(NAMES[p] for p in sub.index[1:])
st.markdown(f"**Main driver: {NAMES[main]}** (sub-index about {sub.iloc[0]:.0f}, "
            f"{categorize(sub.iloc[0])[0]}). Also considered: {others}.")
missing = [NAMES[p] for p in BP if pd.isna(r[p])]
if missing:
    st.caption("Not recorded on this day: " + ", ".join(missing))
st.caption("Estimated from the day's concentrations with CPCB breakpoints. The forecast covers overall AQI; "
           "this chart explains the observed day, not the forecast days.")





def categorize(a):
    for hi, name, color, advice in CATS:
        if a <= hi:
            return name, color, advice
    return CATS[-1][1:]

@st.cache_resource
def load_models():
    return joblib.load("artifacts/aqi_models.joblib")

@st.cache_data
def load_data():
    latest = pd.read_csv("artifacts/latest.csv", parse_dates=["Date"])
    hist = pd.read_csv("artifacts/history.csv", parse_dates=["Date"])
    return latest, hist

bundle = load_models()
models, feats = bundle["models"], bundle["feats"]
latest, hist = load_data()

st.title("India AQI Forecast")
st.caption("7-day AQI forecast with health advice. Demo uses data up to the last available date (Jul 2020).")

city = st.sidebar.selectbox("City", sorted(latest["City"].unique()))
row = latest[latest["City"] == city].iloc[0]
X = row[feats].to_frame().T.astype(float)

preds = [float(np.clip(row["AQI"] + models[h].predict(X)[0], 0, 500)) for h in range(1, 8)]
dates = pd.date_range(row["Date"] + pd.Timedelta(days=1), periods=7)
fc = pd.DataFrame({"Date": dates, "AQI": np.round(preds).astype(int)})
fc["Category"] = fc["AQI"].apply(lambda a: categorize(a)[0])

c1, c2, c3 = st.columns(3)
c1.metric(f"Last observed ({row['Date']:%d %b %Y})", int(row["AQI"]))
c2.metric("Tomorrow", int(fc["AQI"][0]))
c3.metric("Day 7", int(fc["AQI"][6]))

name, color, advice = categorize(fc["AQI"][0])
st.markdown(
    f"<div style='padding:14px;border-radius:8px;background:{color};color:white'>"
    f"<b>Tomorrow: {name}</b><br>{advice}</div>", unsafe_allow_html=True)

h = hist[hist["City"] == city].tail(90)
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(h["Date"], h["AQI"], label="Observed (last 90 days)")
ax.plot(fc["Date"], fc["AQI"], "o--", color="crimson", label="Forecast")
for hi, _, col, _ in CATS:
    ax.axhline(hi, color=col, alpha=0.25, lw=1)
ax.set_ylabel("AQI"); ax.legend(); ax.grid(alpha=0.2)
ax.set_ylim(0, max(150, h["AQI"].max(), fc["AQI"].max()) * 1.15)
st.pyplot(fig)

st.subheader("7-day forecast")
st.dataframe(fc.assign(Date=fc["Date"].dt.strftime("%a %d %b")), hide_index=True, use_container_width=True)

st.info("Forecast error grows with horizon (about R² 0.9 at 1 day, about 0.6 at 3 to 7 days). "
        "Jaipur and Kolkata have shorter training history.")

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