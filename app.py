import streamlit as st
import pandas as pd
import joblib
from datetime import date

# Load artifacts
model  = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
le     = joblib.load("label_encoder.pkl")

FEATURES = list(scaler.feature_names_in_)

st.title("🛒 Sales Demand Spike Predictor")

col1, col2, col3 = st.columns(3)

with col1:
    category             = st.selectbox("Category", le.classes_)
    unit_price           = st.number_input("Unit Price ($)", value=50.0)
    discount_pct         = st.number_input("Discount %", value=0.0, min_value=0.0, max_value=100.0)
    marketing_spend      = st.number_input("Marketing Spend ($)", value=1000.0)
    cogs_per_unit        = st.number_input("COGS Per Unit ($)", value=20.0)

with col2:
    economic_index       = st.number_input("Economic Index", value=100.0)
    consumer_sentiment   = st.number_input("Consumer Sentiment", value=50.0)
    csat_score           = st.number_input("CSAT Score (1-5)", value=4.0)
    competitor_price     = st.number_input("Competitor Price ($)", value=50.0)
    price_vs_competitor  = st.number_input("Price vs Competitor", value=0.0)

with col3:
    pred_date            = st.date_input("Prediction Date", value=date.today())
    is_holiday           = st.checkbox("Is Holiday?")
    is_black_friday_week = st.checkbox("Is Black Friday Week?")
    is_cyber_monday      = st.checkbox("Is Cyber Monday?")
    is_prime_day         = st.checkbox("Is Prime Day?")

st.divider()
st.subheader("📈 Recent Revenue & Units (from your records)")
st.caption("Enter recent sales figures — used by the model to detect trend patterns.")

c1, c2 = st.columns(2)
with c1:
    revenue_lag_1d    = st.number_input("Yesterday's Revenue ($)",       value=5000.0)
    revenue_lag_7d    = st.number_input("Revenue 7 Days Ago ($)",        value=4800.0)
    revenue_lag_30d   = st.number_input("Revenue 30 Days Ago ($)",       value=4600.0)
    revenue_roll7_mean= st.number_input("Avg Revenue Last 7 Days ($)",   value=4900.0)
with c2:
    revenue_roll30_mean=st.number_input("Avg Revenue Last 30 Days ($)",  value=4700.0)
    revenue_roll7_std  =st.number_input("Std Dev Revenue Last 7 Days",   value=300.0)
    units_lag_1d      = st.number_input("Units Sold Yesterday",          value=100.0)
    units_roll7_mean  = st.number_input("Avg Units Sold Last 7 Days",    value=95.0)

if st.button("🔍 Predict", use_container_width=True):

    # Derive date features automatically
    year       = pred_date.year
    quarter    = (pred_date.month - 1) // 3 + 1
    month      = pred_date.month
    day_of_week= pred_date.weekday()          # 0=Mon, 6=Sun
    is_weekend = int(day_of_week >= 5)
    season_ord = {1:1, 2:1, 3:2, 4:2, 5:2, 6:3, 7:3, 8:3, 9:4, 10:4, 11:4, 12:1}[month]

    raw = {
        # Categorical
        "category"              : le.transform([category])[0],
        # Price / marketing
        "unit_price"            : unit_price,
        "discount_pct"          : discount_pct,
        "marketing_spend"       : marketing_spend,
        "cogs_per_unit"         : cogs_per_unit,
        "competitor_price"      : competitor_price,
        "price_vs_competitor"   : price_vs_competitor,
        # Sentiment / quality
        "economic_index"        : economic_index,
        "consumer_sentiment"    : consumer_sentiment,
        "csat_score"            : csat_score,
        # Date features (auto-derived)
        "year"                  : year,
        "month"                 : month,
        "quarter"               : quarter,
        "day_of_week"           : day_of_week,
        "season_ord"            : season_ord,
        "is_weekend"            : is_weekend,
        # Event flags
        "is_holiday"            : int(is_holiday),
        "is_black_friday_week"  : int(is_black_friday_week),
        "is_cyber_monday"       : int(is_cyber_monday),
        "is_prime_day"          : int(is_prime_day),
        # Lag / rolling features
        "revenue_lag_1d"        : revenue_lag_1d,
        "revenue_lag_7d"        : revenue_lag_7d,
        "revenue_lag_30d"       : revenue_lag_30d,
        "revenue_roll7_mean"    : revenue_roll7_mean,
        "revenue_roll30_mean"   : revenue_roll30_mean,
        "revenue_roll7_std"     : revenue_roll7_std,
        "units_lag_1d"          : units_lag_1d,
        "units_roll7_mean"      : units_roll7_mean,
    }

    # Reorder to exactly match scaler
    input_df     = pd.DataFrame([raw])[FEATURES]
    input_scaled = scaler.transform(input_df)
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"🔴 HIGH DEMAND SPIKE predicted!  —  {probability*100:.1f}% confidence")
    else:
        st.success(f"🟢 Normal demand expected  —  {(1-probability)*100:.1f}% confidence")