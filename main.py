import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import pickle
import random
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="PlanWise",
    page_icon=":moneybag:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for models
@st.cache_resource
def load_or_train_models():
    try:
        with open('rf_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        with open('lr_model.pkl', 'rb') as f:
            lr_model = pickle.load(f)
        with open('kmeans_model.pkl', 'rb') as f:
            kmeans = pickle.load(f)
        return rf_model, lr_model, kmeans
    except FileNotFoundError:
        return train_models()

def generate_user_record():
    income = random.randint(30000, 150000)
    expense = random.randint(20000, int(income * 0.8))
    sip = random.randint(1000, int(income * 0.3))
    epf = random.randint(1000, 10000)
    other_inv = random.randint(500, 8000)
    savings = random.randint(50000, 800000)
    loans = random.randint(0, 1500000)
    home_loan = random.randint(0, 30000)
    personal_loan = random.randint(0, 15000)
    credit_score = random.randint(600, 850)
    goal_amount = random.randint(5000000, 15000000)
    current_year = 2025
    goal_year = random.randint(2030, 2045)

    years_to_goal = goal_year - current_year
    total_investment = sip + epf + other_inv
    net_income = income - expense - home_loan - personal_loan
    estimated_growth = savings + (total_investment * 12 * years_to_goal * random.uniform(1.05, 1.15))

    return {
        "monthly_income": income,
        "monthly_expense": expense,
        "sip_investment": sip,
        "epf_contribution": epf,
        "other_investments": other_inv,
        "credit_score": credit_score,
        "home_loan_emi": home_loan,
        "personal_loan_emi": personal_loan,
        "total_savings": savings,
        "total_loans": loans,
        "goal_amount": goal_amount,
        "goal_year": goal_year,
        "current_year": current_year,
        "years_to_goal": years_to_goal,
        "total_investment": total_investment,
        "net_income": net_income,
        "estimated_corpus": estimated_growth
    }

def train_models():
    with st.spinner("Training ML models... This may take a moment."):
        dataset = [generate_user_record() for _ in range(500)]
        df_train = pd.DataFrame(dataset)
        features = [
            'monthly_income', 'monthly_expense', 'sip_investment',
            'epf_contribution', 'other_investments', 'credit_score',
            'home_loan_emi', 'personal_loan_emi', 'total_savings',
            'total_loans', 'years_to_goal', 'total_investment', 'net_income'
        ]
        X = df_train[features]
        y = df_train['estimated_corpus']

        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X, y)

        lr_model = LinearRegression()
        lr_model.fit(X, y)

        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(X)

        try:
            with open('rf_model.pkl', 'wb') as f:
                pickle.dump(rf_model, f)
            with open('lr_model.pkl', 'wb') as f:
                pickle.dump(lr_model, f)
            with open('kmeans_model.pkl', 'wb') as f:
                pickle.dump(kmeans, f)
        except:
            pass

        return rf_model, lr_model, kmeans

def calculate_goal_gap(goal, current_year=2025, inflation_rate=0.06):
    years_to_goal = goal["goal_year"] - current_year
    inflated_goal = goal["goal_amount"] * ((1 + inflation_rate) ** years_to_goal)
    return round(inflated_goal, 2), years_to_goal

def configure_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("\u274c Gemini API key not found. Set it in your environment.")
        st.stop()
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("models/gemini-1.5-pro")

def generate_financial_advice(user_data, user_goals, rf_model, model):
    features = [
        'monthly_income', 'monthly_expense', 'sip_investment',
        'epf_contribution', 'other_investments', 'credit_score',
        'home_loan_emi', 'personal_loan_emi', 'total_savings',
        'total_loans', 'years_to_goal', 'total_investment', 'net_income'
    ]
    df_user = pd.DataFrame([user_data])
    df_user["net_income"] = df_user["monthly_income"] - df_user["monthly_expense"]
    df_user["total_investment"] = df_user["sip_investment"] + df_user["epf_contribution"] + df_user["other_investments"]

    results = []
    for goal in user_goals:
        inflated_target, years_left = calculate_goal_gap(goal)
        df_user["years_to_goal"] = years_left
        X_user = df_user[features]
        predicted_corpus = rf_model.predict(X_user)[0]

        prompt = f"""
        The user wants to achieve the goal: {goal['goal_name']} by {goal['goal_year']}.
        The target (inflation adjusted) is ₹{inflated_target:,.2f}, but the ML model predicts they will have ₹{predicted_corpus:,.2f}.
        Suggest personalized financial advice to help them meet this goal.
        Be practical and use bullet points.
        """

        try:
            response = model.generate_content(prompt)
            gemini_advice = response.text
        except Exception as e:
            gemini_advice = f"Error generating advice: {str(e)}"

        results.append({
            "goal": goal["goal_name"],
            "year": goal["goal_year"],
            "inflated_target": inflated_target,
            "predicted_corpus": predicted_corpus,
            "gemini_advice": gemini_advice,
            "gap": inflated_target - predicted_corpus
        })

    return results

# Load models and Gemini
rf_model, lr_model, kmeans = load_or_train_models()
gemini_model = configure_gemini()

# Main App
st.title("🤖 PlanWise- AI Powered Financial Planning Assistant")
st.markdown("---")

# Sidebar for user inputs
with st.sidebar:
    st.header("📊 Your Financial Profile")
    
    # Income and Expenses
    st.subheader("💰 Income & Expenses")
    monthly_income = st.number_input("Monthly Income (₹)", min_value=0, value=75000, step=1000)
    monthly_expense = st.number_input("Monthly Expenses (₹)", min_value=0, value=45000, step=1000)
    
    # Investments
    st.subheader("📈 Investments")
    sip_investment = st.number_input("SIP Investment (₹)", min_value=0, value=10000, step=500)
    epf_contribution = st.number_input("EPF Contribution (₹)", min_value=0, value=5000, step=500)
    other_investments = st.number_input("Other Investments (₹)", min_value=0, value=2000, step=500)
    
    # Loans
    st.subheader("🏠 Loans & EMIs")
    home_loan_emi = st.number_input("Home Loan EMI (₹)", min_value=0, value=12000, step=1000)
    personal_loan_emi = st.number_input("Personal Loan EMI (₹)", min_value=0, value=4000, step=500)
    
    # Assets and Liabilities
    st.subheader("💎 Assets & Liabilities")
    total_savings = st.number_input("Total Savings (₹)", min_value=0, value=350000, step=10000)
    total_loans = st.number_input("Total Loans Outstanding (₹)", min_value=0, value=950000, step=10000)
    credit_score = st.slider("Credit Score", min_value=300, max_value=900, value=780, step=10)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🎯 Financial Goals")
    
    # Dynamic goal creation
    if 'goals' not in st.session_state:
        st.session_state.goals = [
            {"goal_name": "Retirement", "goal_amount": 10000000, "goal_year": 2040},
            {"goal_name": "Child's Education", "goal_amount": 2500000, "goal_year": 2035},
            {"goal_name": "Buy a House", "goal_amount": 6000000, "goal_year": 2028}
        ]
    
    # Display and edit goals
    for i, goal in enumerate(st.session_state.goals):
        with st.expander(f"Goal {i+1}: {goal['goal_name']}", expanded=True):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                goal['goal_name'] = st.text_input(f"Goal Name {i+1}", value=goal['goal_name'])
            with col_b:
                goal['goal_amount'] = st.number_input(f"Target Amount (₹) {i+1}", min_value=0, value=goal['goal_amount'], step=100000)
            with col_c:
                goal['goal_year'] = st.number_input(f"Target Year {i+1}", min_value=2025, max_value=2060, value=goal['goal_year'])
    
    # Add/Remove goals
    col_add, col_remove = st.columns(2)
    with col_add:
        if st.button("➕ Add Goal"):
            st.session_state.goals.append({"goal_name": "New Goal", "goal_amount": 1000000, "goal_year": 2035})
            st.rerun()
    with col_remove:
        if st.button("➖ Remove Last Goal") and len(st.session_state.goals) > 1:
            st.session_state.goals.pop()
            st.rerun()

with col2:
    st.header("📊 Quick Stats")
    
    net_income = monthly_income - monthly_expense - home_loan_emi - personal_loan_emi
    total_investment = sip_investment + epf_contribution + other_investments
    
    st.metric("Net Monthly Income", f"₹{net_income:,}")
    st.metric("Total Monthly Investment", f"₹{total_investment:,}")
    st.metric("Investment Rate", f"{(total_investment/monthly_income*100):.1f}%")

# Generate Analysis
if st.button("🔮 Generate Financial Analysis", type="primary", use_container_width=True):
    # Prepare user data
    user_data = {
        "monthly_income": monthly_income,
        "monthly_expense": monthly_expense,
        "sip_investment": sip_investment,
        "epf_contribution": epf_contribution,
        "other_investments": other_investments,
        "credit_score": credit_score,
        "home_loan_emi": home_loan_emi,
        "personal_loan_emi": personal_loan_emi,
        "total_savings": total_savings,
        "total_loans": total_loans,
        "current_year": 2025
    }
    
    with st.spinner("🤖 AI is analyzing your financial profile..."):
        results = generate_financial_advice(user_data, st.session_state.goals, rf_model, gemini_model)
    
    # Display Results
    st.header("📋 Analysis Results")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        achievable_goals = sum(1 for r in results if r['gap'] <= 0)
        st.metric("Achievable Goals", f"{achievable_goals}/{len(results)}")
    with col2:
        total_gap = sum(max(0, r['gap']) for r in results)
        st.metric("Total Funding Gap", f"₹{total_gap:,.0f}")
    with col3:
        avg_prediction = np.mean([r['predicted_corpus'] for r in results])
        st.metric("Avg Predicted Corpus", f"₹{avg_prediction:,.0f}")
    
    # Visualization
    if results:
        goal_names = [r['goal'] for r in results]
        targets = [r['inflated_target'] for r in results]
        predictions = [r['predicted_corpus'] for r in results]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(goal_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, targets, width, label='Target (Inflation Adjusted)', color='orange', alpha=0.8)
        bars2 = ax.bar(x + width/2, predictions, width, label='Predicted Corpus', color='green', alpha=0.8)
        
        ax.set_xlabel('Goals')
        ax.set_ylabel('Amount (₹)')
        ax.set_title('Predicted vs Target Corpus Analysis')
        ax.set_xticks(x)
        ax.set_xticklabels(goal_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format y-axis to show values in lakhs/crores
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x/100000:.1f}L' if x < 10000000 else f'₹{x/10000000:.1f}Cr'))
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Detailed analysis for each goal
    for i, result in enumerate(results):
        with st.expander(f"📊 {result['goal']} Analysis", expanded=True):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric("Target Year", result['year'])
                st.metric("Inflation-Adjusted Target", f"₹{result['inflated_target']:,.0f}")
                st.metric("Predicted Corpus", f"₹{result['predicted_corpus']:,.0f}")
                
                if result['gap'] > 0:
                    st.error(f"Shortfall: ₹{result['gap']:,.0f}")
                else:
                    st.success(f"Surplus: ₹{abs(result['gap']):,.0f}")
            
            with col2:
                st.subheader("🤖 AI Recommendations")
                st.write(result['gemini_advice'])

    # User cluster analysis
    st.header("👤 Financial Profile Analysis")
    user_features = pd.DataFrame([{
        'monthly_income': monthly_income,
        'monthly_expense': monthly_expense,
        'sip_investment': sip_investment,
        'epf_contribution': epf_contribution,
        'other_investments': other_investments,
        'credit_score': credit_score,
        'home_loan_emi': home_loan_emi,
        'personal_loan_emi': personal_loan_emi,
        'total_savings': total_savings,
        'total_loans': total_loans,
        'years_to_goal': 15,  # Average
        'total_investment': total_investment,
        'net_income': net_income
    }])
    
    user_cluster = kmeans.predict(user_features)[0]
    cluster_names = {0: "Conservative Investor", 1: "Balanced Investor", 2: "Aggressive Investor"}
    
    st.info(f"🏷️ Your Financial Profile: **{cluster_names.get(user_cluster, f'Cluster {user_cluster}')}**")

# Footer
st.markdown("---")
st.markdown("💡 **Tip**: This analysis uses machine learning predictions and should be used for guidance only. Please consult with a certified financial advisor for personalized advice.")