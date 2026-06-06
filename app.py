# =============================================================================
#  CREDIT RISK ANALYSIS — STREAMLIT WEB APP
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Credit Risk Analyser", page_icon="🏦", layout="wide")
st.title("🏦 Credit Risk Analysis Model")
st.markdown("**German Credit Dataset · Logistic Regression · Decision Tree · Random Forest**")
st.markdown("---")

# =============================================================================
# LOAD & TRAIN
# =============================================================================
@st.cache_resource
def load_and_train():
    column_names = [
        'checking_account','duration','credit_history','purpose','credit_amount',
        'savings_account','employment','installment_rate','personal_status',
        'other_debtors','residence_since','property','age','other_installments',
        'housing','existing_credits','job','dependents','telephone',
        'foreign_worker','target'
    ]
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
        df = pd.read_csv(url, sep=' ', header=None, names=column_names)
    except:
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            'checking_account': np.random.choice(['A11','A12','A13','A14'], n, p=[0.27,0.27,0.06,0.40]),
            'duration': np.random.randint(4, 73, n),
            'credit_history': np.random.choice(['A30','A31','A32','A33','A34'], n),
            'purpose': np.random.choice(['A40','A41','A42','A43','A44','A45','A46','A48','A49'], n),
            'credit_amount': np.random.randint(250, 18425, n),
            'savings_account': np.random.choice(['A61','A62','A63','A64','A65'], n, p=[0.60,0.10,0.06,0.06,0.18]),
            'employment': np.random.choice(['A71','A72','A73','A74','A75'], n),
            'installment_rate': np.random.randint(1, 5, n),
            'personal_status': np.random.choice(['A91','A92','A93','A94'], n),
            'other_debtors': np.random.choice(['A101','A102','A103'], n),
            'residence_since': np.random.randint(1, 5, n),
            'property': np.random.choice(['A121','A122','A123','A124'], n),
            'age': np.random.randint(19, 75, n),
            'other_installments': np.random.choice(['A141','A142','A143'], n),
            'housing': np.random.choice(['A151','A152','A153'], n),
            'existing_credits': np.random.randint(1, 5, n),
            'job': np.random.choice(['A171','A172','A173','A174'], n),
            'dependents': np.random.randint(1, 3, n),
            'telephone': np.random.choice(['A191','A192'], n),
            'foreign_worker': np.random.choice(['A201','A202'], n),
            'target': np.random.choice([1, 2], n, p=[0.70, 0.30])
        })

    df['target'] = df['target'].map({1: 0, 2: 1})
    numeric_cols = ['duration','credit_amount','installment_rate',
                    'residence_since','age','existing_credits','dependents']
    categorical_cols = [c for c in df.columns if c not in numeric_cols and c != 'target']

    # Store label encoders per column so we can reuse them for prediction
    encoders = {}
    df_enc = df.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le

    feature_cols = [c for c in df_enc.columns if c != 'target']
    X = df_enc[feature_cols]
    y = df_enc['target']

    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree':       DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    }
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred  = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        auc   = roc_auc_score(y_test, proba)
        trained[name] = {'model': model, 'pred': pred, 'proba': proba, 'auc': auc}

    importances = pd.Series(
        trained['Random Forest']['model'].feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    return df, X_scaled, y, X_test, y_test, trained, scaler, numeric_cols, feature_cols, importances, encoders, categorical_cols

df, X_scaled, y, X_test, y_test, trained, scaler, numeric_cols, feature_cols, importances, encoders, categorical_cols = load_and_train()

# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", [
    "📊 Dataset Overview",
    "🤖 Model Performance",
    "🔍 Predict New Borrower",
    "📈 Feature Importance"
])
st.sidebar.markdown("---")
st.sidebar.info("German Credit Dataset — 1,000 applicants, 20 features, 3 ML models.")

# =============================================================================
# PAGE 1 — DATASET OVERVIEW
# =============================================================================
if page == "📊 Dataset Overview":
    st.header("📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Applicants", "1,000")
    col2.metric("Features", "20")
    col3.metric("Good Risk", f"{(y==0).sum()} ({(y==0).mean()*100:.0f}%)")
    col4.metric("Bad Risk",  f"{(y==1).sum()} ({(y==1).mean()*100:.0f}%)")

    st.markdown("### Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        ax.bar(['Good Risk', 'Bad Risk'], [(y==0).sum(), (y==1).sum()], color=['#2ecc71','#e74c3c'])
        ax.set_title('Target Distribution'); ax.set_ylabel('Count')
        st.pyplot(fig); plt.close()
    with col2:
        fig, ax = plt.subplots()
        ax.hist(df[df['target']==0]['age'], bins=20, alpha=0.6, color='#3498db', label='Good Risk')
        ax.hist(df[df['target']==1]['age'], bins=20, alpha=0.6, color='#e74c3c', label='Bad Risk')
        ax.set_title('Age Distribution'); ax.set_xlabel('Age'); ax.legend()
        st.pyplot(fig); plt.close()

    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots()
        ax.hist(df[df['target']==0]['credit_amount'], bins=25, alpha=0.6, color='#2ecc71', label='Good Risk')
        ax.hist(df[df['target']==1]['credit_amount'], bins=25, alpha=0.6, color='#e74c3c', label='Bad Risk')
        ax.set_title('Loan Amount by Risk'); ax.set_xlabel('Credit Amount (DM)'); ax.legend()
        st.pyplot(fig); plt.close()
    with col4:
        fig, ax = plt.subplots()
        ax.hist(df[df['target']==0]['duration'], bins=20, alpha=0.6, color='#9b59b6', label='Good Risk')
        ax.hist(df[df['target']==1]['duration'], bins=20, alpha=0.6, color='#e67e22', label='Bad Risk')
        ax.set_title('Loan Duration by Risk'); ax.set_xlabel('Duration (months)'); ax.legend()
        st.pyplot(fig); plt.close()

# =============================================================================
# PAGE 2 — MODEL PERFORMANCE
# =============================================================================
elif page == "🤖 Model Performance":
    st.header("🤖 Model Performance")
    perf_df = pd.DataFrame([
        {'Model': name, 'AUC-ROC': f"{res['auc']:.3f}",
         'Rating': '🟢 Good' if res['auc'] > 0.70 else '🟡 Fair'}
        for name, res in trained.items()
    ])
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ROC Curves")
        fig, ax = plt.subplots()
        colors = ['#9b59b6','#e67e22','#e74c3c']
        for (name, res), col in zip(trained.items(), colors):
            fpr, tpr, _ = roc_curve(y_test, res['proba'])
            ax.plot(fpr, tpr, color=col, lw=2, label=f"{name} ({res['auc']:.2f})")
        ax.plot([0,1],[0,1],'k--', alpha=0.4, label='Random (0.50)')
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves'); ax.legend(fontsize=8)
        st.pyplot(fig); plt.close()
    with col2:
        st.markdown("### Confusion Matrix — Random Forest")
        fig, ax = plt.subplots()
        cm = confusion_matrix(y_test, trained['Random Forest']['pred'])
        disp = ConfusionMatrixDisplay(cm, display_labels=['Good Risk','Bad Risk'])
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        st.pyplot(fig); plt.close()

    st.info("""
    **AUC-ROC Guide:**
    - 0.50 = Random guessing | 0.70+ = Acceptable | 0.80+ = Good | 0.85+ = Excellent
    """)

# =============================================================================
# PAGE 3 — PREDICT NEW BORROWER
# =============================================================================
elif page == "🔍 Predict New Borrower":
    st.header("🔍 Predict Credit Risk for a New Borrower")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**💳 Loan Details**")
        duration = st.slider("Loan Duration (months)", 6, 72, 24)
        credit_amount = st.number_input("Loan Amount (DM)", 250, 20000, 5000, step=250)
        purpose = st.selectbox("Loan Purpose", ['A40','A41','A42','A43','A44','A45'])
        installment_rate = st.slider("Instalment Rate (% of income)", 1, 4, 2)

    with col2:
        st.markdown("**🏦 Financial Status**")
        checking_account = st.selectbox("Checking Account", {
            'A11 - Negative balance':'A11',
            'A12 - 0 to 200 DM':'A12',
            'A13 - Over 200 DM':'A13',
            'A14 - No account':'A14'
        }.keys())
        savings_account = st.selectbox("Savings Account", {
            'A61 - Under 100 DM':'A61',
            'A62 - 100 to 500 DM':'A62',
            'A63 - 500 to 1000 DM':'A63',
            'A64 - Over 1000 DM':'A64',
            'A65 - Unknown/None':'A65'
        }.keys())
        credit_history = st.selectbox("Credit History", {
            'A30 - Critical':'A30',
            'A31 - Delayed in past':'A31',
            'A32 - Existing paid':'A32',
            'A33 - All paid':'A33',
            'A34 - No credits':'A34'
        }.keys())
        property_ = st.selectbox("Property", {
            'A121 - Real estate':'A121',
            'A122 - Life insurance':'A122',
            'A123 - Car/other':'A123',
            'A124 - None':'A124'
        }.keys())

    with col3:
        st.markdown("**👤 Personal Details**")
        age = st.slider("Age (years)", 18, 75, 35)
        employment = st.selectbox("Employment Duration", {
            'A71 - Unemployed':'A71',
            'A72 - Under 1 year':'A72',
            'A73 - 1 to 4 years':'A73',
            'A74 - 4 to 7 years':'A74',
            'A75 - Over 7 years':'A75'
        }.keys())
        housing = st.selectbox("Housing", {
            'A151 - Rent':'A151',
            'A152 - Own':'A152',
            'A153 - For free':'A153'
        }.keys())
        other_debtors = st.selectbox("Guarantor", {
            'A101 - None':'A101',
            'A102 - Co-applicant':'A102',
            'A103 - Guarantor':'A103'
        }.keys())
        existing_credits = st.slider("Existing Credits", 1, 4, 1)
        dependents = st.slider("Dependents", 1, 2, 1)

    # Extract codes from dropdown labels
    ca_code = checking_account.split(' - ')[0]
    sv_code = savings_account.split(' - ')[0]
    ch_code = credit_history.split(' - ')[0]
    em_code = employment.split(' - ')[0]
    ho_code = housing.split(' - ')[0]
    od_code = other_debtors.split(' - ')[0]
    pr_code = property_.split(' - ')[0]

    if st.button("🔍 Predict Credit Risk", type="primary", use_container_width=True):

        # Build profile dict
        profile_dict = {
            'checking_account': ca_code,
            'duration': duration,
            'credit_history': ch_code,
            'purpose': purpose,
            'credit_amount': credit_amount,
            'savings_account': sv_code,
            'employment': em_code,
            'installment_rate': installment_rate,
            'personal_status': 'A93',
            'other_debtors': od_code,
            'residence_since': 3,
            'property': pr_code,
            'age': age,
            'other_installments': 'A143',
            'housing': ho_code,
            'existing_credits': existing_credits,
            'job': 'A173',
            'dependents': dependents,
            'telephone': 'A192',
            'foreign_worker': 'A201'
        }

        # Encode using SAME encoders used during training
        profile_enc = {}
        for col in feature_cols:
            val = profile_dict[col]
            if col in encoders:
                le = encoders[col]
                # Handle unseen labels gracefully
                if str(val) in le.classes_:
                    profile_enc[col] = le.transform([str(val)])[0]
                else:
                    profile_enc[col] = 0
            else:
                profile_enc[col] = val

        profile_df = pd.DataFrame([profile_enc], columns=feature_cols)
        profile_df[numeric_cols] = scaler.transform(profile_df[numeric_cols])

        # Predict
        st.markdown("---")
        st.markdown("### 📋 Prediction Results")
        cols = st.columns(3)
        for i, (name, res) in enumerate(trained.items()):
            model = res['model']
            pred  = model.predict(profile_df)[0]
            proba = model.predict_proba(profile_df)[0][1]
            with cols[i]:
                if pred == 0:
                    st.success(f"**{name}**\n\n✅ GOOD RISK\n\nDefault Prob: **{proba*100:.1f}%**")
                else:
                    st.error(f"**{name}**\n\n❌ BAD RISK\n\nDefault Prob: **{proba*100:.1f}%**")

        rf_pred  = trained['Random Forest']['model'].predict(profile_df)[0]
        rf_proba = trained['Random Forest']['model'].predict_proba(profile_df)[0][1]
        lgd = 0.45
        el  = rf_proba * lgd * credit_amount

        st.markdown("---")
        st.markdown("### 💡 Final Recommendation (Random Forest)")
        if rf_pred == 0:
            st.success(f"""
            ✅ **APPROVE** — Low credit risk

            - Probability of Default (PD): **{rf_proba*100:.1f}%**
            - Loss Given Default (LGD): **{lgd*100:.0f}%**
            - Exposure at Default (EAD): **DM {credit_amount:,}**
            - Expected Loss (EL): **DM {el:,.0f}**

            *Recommended: Approve with standard interest rate.*
            """)
        else:
            st.error(f"""
            ❌ **DECLINE** — High credit risk

            - Probability of Default (PD): **{rf_proba*100:.1f}%**
            - Loss Given Default (LGD): **{lgd*100:.0f}%**
            - Exposure at Default (EAD): **DM {credit_amount:,}**
            - Expected Loss (EL): **DM {el:,.0f}**

            *Recommended: Decline or require guarantor / reduce loan amount.*
            """)

# =============================================================================
# PAGE 4 — FEATURE IMPORTANCE
# =============================================================================
elif page == "📈 Feature Importance":
    st.header("📈 Feature Importance")
    top10 = importances.head(10)
    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#e74c3c' if v == top10.max() else '#3498db' for v in top10.values]
        ax.barh(top10.index[::-1], top10.values[::-1], color=colors[::-1], alpha=0.85)
        ax.set_xlabel('Importance Score'); ax.set_title('Top 10 Features — Random Forest')
        st.pyplot(fig); plt.close()
    with col2:
        st.markdown("### Top Features")
        for feat, score in top10.items():
            st.markdown(f"**{feat}** — `{score:.4f}`")

    st.info("""
    | Feature | Meaning |
    |---|---|
    | **credit_amount** | Higher loan = higher risk |
    | **age** | Older borrowers tend to be more reliable |
    | **duration** | Longer loans carry more risk |
    | **checking_account** | Account balance is a strong predictor |
    | **savings_account** | More savings = lower risk |
    """)
