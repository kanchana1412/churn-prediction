import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import io

st.set_page_config(page_title="Churn Predictor", page_icon="🌸", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
* { font-family: 'Poppins', sans-serif; }
.stApp { background-color: #fff0f5; }
.hero { text-align: center; padding: 40px 0 10px 0; }
.hero h1 { font-size: 52px; font-weight: 700; color: #c2185b; }
.hero p { color: #f48fb1; font-size: 18px; }
.stat-card { background: white; border: 2px solid #f8bbd0; border-radius: 20px; padding: 20px; text-align: center; box-shadow: 0 4px 15px rgba(194,24,91,0.08); }
.stat-number { font-size: 36px; font-weight: 700; color: #e91e8c; }
.stat-label { font-size: 12px; color: #f48fb1; text-transform: uppercase; letter-spacing: 1px; }
.input-card { background: white; border: 2px solid #f8bbd0; border-radius: 20px; padding: 30px; box-shadow: 0 4px 15px rgba(194,24,91,0.08); }
.result-high { background: linear-gradient(135deg, #ff6b9d, #ff4081); border-radius: 20px; padding: 35px; text-align: center; color: white; }
.result-medium { background: linear-gradient(135deg, #ffb3c6, #ff80ab); border-radius: 20px; padding: 35px; text-align: center; color: white; }
.result-low { background: linear-gradient(135deg, #fce4ec, #f8bbd0); border-radius: 20px; padding: 35px; text-align: center; color: #c2185b; }
.result-number { font-size: 60px; font-weight: 700; }
.result-label { font-size: 22px; font-weight: 600; margin-top: 5px; }
.result-tip { font-size: 14px; margin-top: 12px; opacity: 0.9; }
div[data-testid="stButton"] button { background: linear-gradient(90deg, #f48fb1, #e91e8c); color: white !important; border: none !important; border-radius: 12px !important; font-size: 17px !important; font-weight: 600 !important; }
label, p { color: #880e4f !important; }
h3 { color: #c2185b !important; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_train():
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df.drop('customerID', axis=1, inplace=True)
    df_raw = df.copy()
    df = pd.get_dummies(df, drop_first=True)
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    return model, scaler, X.columns, df, df_raw

model, scaler, feature_cols, df, df_raw = load_and_train()

total = len(df)
churned = int(df['Churn'].sum())
stayed = total - churned
churn_rate = (churned / total) * 100

# Hero
st.markdown("""
<div class="hero">
    <h1>🌸 Customer Churn Predictor</h1>
    <p>AI-powered tool to predict if a telecom customer will leave</p>
</div>""", unsafe_allow_html=True)

# Stats
c1, c2, c3, c4 = st.columns(4)
c1.markdown(f'<div class="stat-card"><div class="stat-number">{total:,}</div><div class="stat-label">👥 Total Customers</div></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="stat-card"><div class="stat-number" style="color:#66bb6a">{stayed:,}</div><div class="stat-label">✅ Stayed</div></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="stat-card"><div class="stat-number" style="color:#e91e8c">{churned:,}</div><div class="stat-label">❌ Churned</div></div>', unsafe_allow_html=True)
c4.markdown(f'<div class="stat-card"><div class="stat-number" style="color:#ff80ab">{churn_rate:.1f}%</div><div class="stat-label">📉 Churn Rate</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Single Predict", "📊 Charts", "📁 Bulk Predict"])

# ── TAB 1: Single Predict ──
with tab1:
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown("### 🎀 Customer Profile")
        name = st.text_input("👤 Customer Name", "John Doe")
        tenure = st.slider("📅 Tenure (months)", 0, 72, 12)
        monthly = st.slider("💰 Monthly Charges ($)", 18.0, 119.0, 65.0)
        contract = st.selectbox("📋 Contract Type", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("🌐 Internet Service", ["Fiber optic", "DSL", "No"])
        predict_btn = st.button("🌸 Predict Churn Risk", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown("### 💝 Prediction Result")
        if predict_btn:
            input_data = pd.DataFrame(columns=feature_cols)
            input_data.loc[0] = 0
            input_data['tenure'] = tenure
            input_data['MonthlyCharges'] = monthly
            input_data['TotalCharges'] = tenure * monthly
            if contract == 'One year' and 'Contract_One year' in input_data.columns:
                input_data['Contract_One year'] = 1
            elif contract == 'Two year' and 'Contract_Two year' in input_data.columns:
                input_data['Contract_Two year'] = 1
            if internet == 'Fiber optic' and 'InternetService_Fiber optic' in input_data.columns:
                input_data['InternetService_Fiber optic'] = 1
            elif internet == 'No' and 'InternetService_No' in input_data.columns:
                input_data['InternetService_No'] = 1

            input_scaled = scaler.transform(input_data)
            prob = model.predict_proba(input_scaled)[0][1]
            risk = prob * 100

            if risk > 70:
                level, color = "HIGH RISK 🔴", "#ff4081"
                tip = "⚡ Act now! Offer a discount or upgrade plan immediately."
                st.markdown(f'<div class="result-high"><div class="result-number">{risk:.1f}%</div><div class="result-label">HIGH RISK</div><div class="result-tip">{tip}</div></div>', unsafe_allow_html=True)
            elif risk > 40:
                level, color = "MEDIUM RISK 🟡", "#ff80ab"
                tip = "💌 Send a loyalty reward or schedule a check-in call."
                st.markdown(f'<div class="result-medium"><div class="result-number">{risk:.1f}%</div><div class="result-label">MEDIUM RISK</div><div class="result-tip">{tip}</div></div>', unsafe_allow_html=True)
            else:
                level, color = "LOW RISK 🟢", "#f48fb1"
                tip = "✨ Customer is happy! Keep up the great service."
                st.markdown(f'<div class="result-low"><div class="result-number">{risk:.1f}%</div><div class="result-label">LOW RISK</div><div class="result-tip">{tip}</div></div>', unsafe_allow_html=True)

            # Download Report Card
            st.markdown("<br>", unsafe_allow_html=True)
            report = f"""
CUSTOMER CHURN RISK REPORT
============================
Customer Name   : {name}
Tenure          : {tenure} months
Monthly Charges : ${monthly:.2f}
Contract        : {contract}
Internet        : {internet}
----------------------------
Churn Risk      : {risk:.1f}%
Risk Level      : {level}
Suggestion      : {tip}
============================
Generated by 🌸 Churn Predictor AI
            """
            st.download_button(
                label="📥 Download Report Card",
                data=report,
                file_name=f"{name}_churn_report.txt",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.markdown("""
            <div style="background:white;border:2px dashed #f8bbd0;border-radius:20px;padding:60px;text-align:center;">
                <div style="font-size:60px">🌸</div>
                <div style="color:#f48fb1;margin-top:10px;font-size:16px">Fill the customer profile<br>and click Predict</div>
            </div>""", unsafe_allow_html=True)

# ── TAB 2: Charts ──
with tab2:
    st.markdown("### 📊 Data Insights")
    ch1, ch2 = st.columns(2)

    with ch1:
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#fff0f5')
        ax.set_facecolor('#fff0f5')
        churn_counts = df['Churn'].value_counts()
        ax.pie(churn_counts, labels=['Stayed', 'Churned'],
               colors=['#f8bbd0', '#e91e8c'], autopct='%1.1f%%',
               startangle=90, textprops={'color': '#880e4f'})
        ax.set_title('Overall Churn Distribution', color='#c2185b', fontweight='bold')
        st.pyplot(fig)

    with ch2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        fig2.patch.set_facecolor('#fff0f5')
        ax2.set_facecolor('#fff0f5')
        tenure_churn = df_raw.groupby('Churn')['tenure'].mean()
        bars = ax2.bar(['Stayed', 'Churned'], tenure_churn.values,
                       color=['#f8bbd0', '#e91e8c'], edgecolor='white', linewidth=2)
        ax2.set_title('Avg Tenure: Stayed vs Churned', color='#c2185b', fontweight='bold')
        ax2.set_ylabel('Months', color='#880e4f')
        ax2.tick_params(colors='#880e4f')
        for bar, val in zip(bars, tenure_churn.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}m', ha='center', color='#880e4f', fontweight='bold')
        st.pyplot(fig2)

    ch3, ch4 = st.columns(2)
    with ch3:
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        fig3.patch.set_facecolor('#fff0f5')
        ax3.set_facecolor('#fff0f5')
        monthly_churn = df_raw.groupby('Churn')['MonthlyCharges'].mean()
        bars3 = ax3.bar(['Stayed', 'Churned'], monthly_churn.values,
                        color=['#f8bbd0', '#e91e8c'], edgecolor='white', linewidth=2)
        ax3.set_title('Avg Monthly Charges', color='#c2185b', fontweight='bold')
        ax3.set_ylabel('$ Amount', color='#880e4f')
        ax3.tick_params(colors='#880e4f')
        for bar, val in zip(bars3, monthly_churn.values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'${val:.1f}', ha='center', color='#880e4f', fontweight='bold')
        st.pyplot(fig3)

    with ch4:
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        fig4.patch.set_facecolor('#fff0f5')
        ax4.set_facecolor('#fff0f5')
        contract_churn = df_raw.groupby('Contract')['Churn'].mean() * 100
        bars4 = ax4.bar(contract_churn.index, contract_churn.values,
                        color=['#e91e8c', '#f48fb1', '#f8bbd0'], edgecolor='white', linewidth=2)
        ax4.set_title('Churn Rate by Contract Type', color='#c2185b', fontweight='bold')
        ax4.set_ylabel('Churn %', color='#880e4f')
        ax4.tick_params(colors='#880e4f')
        for bar, val in zip(bars4, contract_churn.values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.1f}%', ha='center', color='#880e4f', fontweight='bold')
        st.pyplot(fig4)

# ── TAB 3: Bulk Predict ──
with tab3:
    st.markdown("### 📁 Bulk Customer Prediction")
    st.info("Upload a CSV file with columns: tenure, MonthlyCharges, Contract, InternetService")

    sample = pd.DataFrame({
        'tenure': [12, 45, 2],
        'MonthlyCharges': [65.0, 40.0, 95.0],
        'Contract': ['Month-to-month', 'Two year', 'Month-to-month'],
        'InternetService': ['Fiber optic', 'DSL', 'Fiber optic']
    })
    st.download_button("📥 Download Sample CSV", sample.to_csv(index=False),
                       "sample_customers.csv", "text/csv", use_container_width=True)

    uploaded = st.file_uploader("Upload your CSV", type=['csv'])
    if uploaded:
        bulk_df = pd.read_csv(uploaded)
        st.write("Preview:", bulk_df.head())

        results = []
        for _, row in bulk_df.iterrows():
            inp = pd.DataFrame(columns=feature_cols)
            inp.loc[0] = 0
            inp['tenure'] = row['tenure']
            inp['MonthlyCharges'] = row['MonthlyCharges']
            inp['TotalCharges'] = row['tenure'] * row['MonthlyCharges']
            if row['Contract'] == 'One year' and 'Contract_One year' in inp.columns:
                inp['Contract_One year'] = 1
            elif row['Contract'] == 'Two year' and 'Contract_Two year' in inp.columns:
                inp['Contract_Two year'] = 1
            if row['InternetService'] == 'Fiber optic' and 'InternetService_Fiber optic' in inp.columns:
                inp['InternetService_Fiber optic'] = 1
            elif row['InternetService'] == 'No' and 'InternetService_No' in inp.columns:
                inp['InternetService_No'] = 1
            prob = model.predict_proba(scaler.transform(inp))[0][1]
            risk = prob * 100
            level = "🔴 HIGH" if risk > 70 else "🟡 MEDIUM" if risk > 40 else "🟢 LOW"
            results.append({'Churn Risk %': f"{risk:.1f}%", 'Risk Level': level})

        result_df = pd.concat([bulk_df, pd.DataFrame(results)], axis=1)
        st.success("✅ Predictions complete!")
        st.dataframe(result_df)
        st.download_button("📥 Download Results", result_df.to_csv(index=False),
                          "churn_predictions.csv", "text/csv", use_container_width=True)

st.markdown("<br><center style='color:#f48fb1'>Made with 💗 using Python · Scikit-learn · Streamlit</center>", unsafe_allow_html=True)