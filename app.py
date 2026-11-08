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