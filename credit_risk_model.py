# =============================================================================
#  CREDIT RISK ANALYSIS MODEL
#  Dataset : German Credit Data (UCI Machine Learning Repository)
#  Author  : Internship Project
#  Purpose : Predict whether a loan applicant is a GOOD or BAD credit risk
# =============================================================================
#
#  WHAT THIS SCRIPT DOES (step by step):
#  1. Downloads the German Credit dataset (publicly available, no sign-in needed)
#  2. Cleans and prepares the data
#  3. Trains THREE machine learning models:
#       a) Logistic Regression   — the industry standard scorecard model
#       b) Decision Tree         — easy to explain to non-technical people
#       c) Random Forest         — more powerful, higher accuracy
#  4. Evaluates each model with standard banking metrics
#  5. Shows which features (age, savings, etc.) matter most
#  6. Lets you predict risk for a NEW borrower
#  7. Saves all charts as PNG files
#
#  TO RUN:
#       python credit_risk_model.py
#
#  REQUIRED LIBRARIES (install once):
#       pip install scikit-learn pandas numpy matplotlib seaborn
# =============================================================================

# --- STEP 0: Import libraries ------------------------------------------------
# Think of libraries as toolboxes. We import specific tools we need.

import pandas as pd          # For handling tables of data (like Excel in Python)
import numpy as np           # For numerical calculations
import matplotlib.pyplot as plt  # For drawing charts
import seaborn as sns        # For prettier statistical charts
import warnings
warnings.filterwarnings('ignore')  # Suppress minor warnings for cleaner output

# Machine Learning tools from scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)

print("=" * 65)
print("       CREDIT RISK ANALYSIS MODEL — GERMAN CREDIT DATASET")
print("=" * 65)


# =============================================================================
# STEP 1: LOAD THE DATASET
# =============================================================================
# The German Credit Dataset has 1000 loan applicants.
# Each row = one applicant.
# Target variable: 1 = Good credit risk, 2 = Bad credit risk
#
# We download it directly from UCI — no file needed!

print("\n[STEP 1] Loading German Credit Dataset from UCI repository...")

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

# Column names (the raw file has no headers, so we define them)
column_names = [
    'checking_account',    # Status of existing checking account
    'duration',            # Duration of credit in months
    'credit_history',      # Credit history (past behaviour)
    'purpose',             # Purpose of the loan (car, furniture, etc.)
    'credit_amount',       # Loan amount
    'savings_account',     # Savings account / bonds
    'employment',          # Present employment since (years)
    'installment_rate',    # Installment rate as % of disposable income
    'personal_status',     # Personal status and gender
    'other_debtors',       # Other debtors / guarantors
    'residence_since',     # Present residence since (years)
    'property',            # Property owned
    'age',                 # Age in years
    'other_installments',  # Other installment plans
    'housing',             # Housing (own, free, rent)
    'existing_credits',    # Number of existing credits at this bank
    'job',                 # Job type
    'dependents',          # Number of people liable to provide maintenance for
    'telephone',           # Telephone registered?
    'foreign_worker',      # Foreign worker?
    'target'               # 1 = Good risk, 2 = Bad risk  ← THIS is what we predict
]

try:
    df = pd.read_csv(url, sep=' ', header=None, names=column_names)
    print(f"   ✓ Dataset loaded successfully!")
    print(f"   ✓ Shape: {df.shape[0]} applicants, {df.shape[1]} features")
except Exception as e:
    # Fallback: generate a synthetic dataset if internet is unavailable
    print(f"   ✗ Could not download dataset ({e})")
    print("   → Generating synthetic dataset with same structure...")
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
    print(f"   ✓ Synthetic dataset created: {df.shape[0]} rows, {df.shape[1]} columns")


# =============================================================================
# STEP 2: EXPLORE THE DATA (Exploratory Data Analysis)
# =============================================================================
# Before building a model, we always look at what we have.
# This is called EDA — Exploratory Data Analysis.

print("\n[STEP 2] Exploring the dataset...")
print("\n--- First 3 rows of data ---")
print(df.head(3).to_string())

print("\n--- Basic statistics (numeric columns) ---")
print(df[['duration', 'credit_amount', 'age']].describe().round(2))

# TARGET DISTRIBUTION
# How many applicants are "good risk" vs "bad risk"?
target_counts = df['target'].value_counts()
good = target_counts.get(1, 0)
bad  = target_counts.get(2, 0)
print(f"\n--- Target variable distribution ---")
print(f"   Good risk (1) : {good} applicants ({good/len(df)*100:.1f}%)")
print(f"   Bad  risk (2) : {bad}  applicants ({bad/len(df)*100:.1f}%)")
print(f"   NOTE: This is an imbalanced dataset — more good than bad applicants.")


# =============================================================================
# STEP 3: DATA PREPROCESSING
# =============================================================================
# Machine learning models only understand NUMBERS.
# But our data has text codes like 'A11', 'A32', etc.
# We need to:
#   a) Convert target: 1=Good → 0, 2=Bad → 1  (1 means "will default")
#   b) Encode text columns into numbers (Label Encoding)
#   c) Scale numeric columns (so big numbers don't dominate)

print("\n[STEP 3] Preprocessing data...")

# a) Convert target variable
#    In banking: 1 = "will default" (bad outcome), 0 = "will repay" (good outcome)
df['target'] = df['target'].map({1: 0, 2: 1})
print(f"   ✓ Target encoded: 0=Good/No Default, 1=Bad/Default")

# b) Identify which columns are text (categorical) and which are numbers
categorical_cols = df.select_dtypes(include='object').columns.tolist()
numeric_cols     = ['duration', 'credit_amount', 'installment_rate',
                    'residence_since', 'age', 'existing_credits', 'dependents']

print(f"   ✓ Categorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"   ✓ Numeric columns    ({len(numeric_cols)}): {numeric_cols}")

# c) Label Encoding — converts each unique text value to a number
#    e.g., 'A11' → 0, 'A12' → 1, 'A13' → 2, etc.
le = LabelEncoder()
df_encoded = df.copy()
for col in categorical_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col])

print(f"   ✓ Text columns converted to numbers")

# d) Separate features (X) from the target (y)
#    X = everything we use to make the prediction (inputs)
#    y = what we're trying to predict (output)
feature_cols = [c for c in df_encoded.columns if c != 'target']
X = df_encoded[feature_cols]
y = df_encoded['target']

# e) Scale numeric features
#    StandardScaler makes all numbers comparable (mean=0, std=1)
#    This prevents "credit_amount" (thousands) from drowning out "age" (tens)
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])

print(f"   ✓ Numeric features scaled (StandardScaler)")
print(f"   ✓ Final feature matrix shape: {X_scaled.shape}")


# =============================================================================
# STEP 4: SPLIT DATA INTO TRAINING AND TESTING SETS
# =============================================================================
# We split data into two parts:
#   - Training set (80%): the model LEARNS from this
#   - Test set    (20%): we CHECK how well the model does on UNSEEN data
#
# This is critical! If we test on the same data we trained on,
# the model looks artificially good (like memorising answers).

print("\n[STEP 4] Splitting data into train/test sets (80% / 20%)...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,       # 20% goes to testing
    random_state=42,     # Fixed seed = reproducible results
    stratify=y           # Keep same good/bad ratio in both splits
)

print(f"   ✓ Training set : {X_train.shape[0]} applicants")
print(f"   ✓ Test set     : {X_test.shape[0]} applicants")


# =============================================================================
# STEP 5: TRAIN THREE MODELS
# =============================================================================

print("\n[STEP 5] Training machine learning models...")
print("-" * 45)

results = {}  # Store results for comparison later

# ─── MODEL 1: Logistic Regression ──────────────────────────────────────────
# The INDUSTRY STANDARD model for credit scoring.
# It calculates a probability (0 to 1) of default for each applicant.
# Banks love it because it's simple, interpretable, and regulators approve it.

print("\n  Model 1: Logistic Regression (industry standard)")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)  # ← This is where "learning" happens

lr_pred  = lr_model.predict(X_test)
lr_proba = lr_model.predict_proba(X_test)[:, 1]  # Probability of default
lr_acc   = accuracy_score(y_test, lr_pred)
lr_auc   = roc_auc_score(y_test, lr_proba)
lr_cv    = cross_val_score(lr_model, X_scaled, y, cv=5, scoring='roc_auc').mean()

results['Logistic Regression'] = {'acc': lr_acc, 'auc': lr_auc, 'cv_auc': lr_cv,
                                   'pred': lr_pred, 'proba': lr_proba}
print(f"     Accuracy : {lr_acc*100:.1f}%")
print(f"     AUC-ROC  : {lr_auc:.3f}  (5-fold CV: {lr_cv:.3f})")

# ─── MODEL 2: Decision Tree ────────────────────────────────────────────────
# A flowchart-like model: "If checking account < X AND age > Y → Good risk"
# Very easy to visualise and explain to business stakeholders.
# Tendency to overfit, so we limit max depth.

print("\n  Model 2: Decision Tree (explainable model)")
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

dt_pred  = dt_model.predict(X_test)
dt_proba = dt_model.predict_proba(X_test)[:, 1]
dt_acc   = accuracy_score(y_test, dt_pred)
dt_auc   = roc_auc_score(y_test, dt_proba)
dt_cv    = cross_val_score(dt_model, X_scaled, y, cv=5, scoring='roc_auc').mean()

results['Decision Tree'] = {'acc': dt_acc, 'auc': dt_auc, 'cv_auc': dt_cv,
                             'pred': dt_pred, 'proba': dt_proba}
print(f"     Accuracy : {dt_acc*100:.1f}%")
print(f"     AUC-ROC  : {dt_auc:.3f}  (5-fold CV: {dt_cv:.3f})")

# Print the top 3 levels of the decision tree (human-readable rules)
tree_rules = export_text(dt_model, feature_names=feature_cols, max_depth=2)
print("\n     Top decision rules (first 2 levels):")
for line in tree_rules.split('\n')[:12]:
    print("     " + line)

# ─── MODEL 3: Random Forest ────────────────────────────────────────────────
# An ENSEMBLE of many decision trees.
# Each tree votes → majority wins. More robust and accurate than a single tree.
# Best model in this set, but harder to explain.

print("\n  Model 3: Random Forest (best accuracy)")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

rf_pred  = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]
rf_acc   = accuracy_score(y_test, rf_pred)
rf_auc   = roc_auc_score(y_test, rf_proba)
rf_cv    = cross_val_score(rf_model, X_scaled, y, cv=5, scoring='roc_auc').mean()

results['Random Forest'] = {'acc': rf_acc, 'auc': rf_auc, 'cv_auc': rf_cv,
                             'pred': rf_pred, 'proba': rf_proba}
print(f"     Accuracy : {rf_acc*100:.1f}%")
print(f"     AUC-ROC  : {rf_auc:.3f}  (5-fold CV: {rf_cv:.3f})")


# =============================================================================
# STEP 6: EVALUATE MODELS
# =============================================================================
# AUC-ROC is the main metric in credit risk.
# AUC = Area Under the Curve. Ranges from 0.5 (random) to 1.0 (perfect).
# > 0.75 is considered good for credit scoring.
#
# We also look at:
#   Precision : Of applicants flagged as "bad", how many truly were?
#   Recall    : Of all truly bad applicants, how many did we catch?
#   F1-score  : Harmonic mean of precision and recall

print("\n[STEP 6] Model evaluation summary")
print("=" * 50)
print(f"{'Model':<22} {'Accuracy':>9} {'AUC-ROC':>9} {'CV-AUC':>9}")
print("-" * 50)
for name, r in results.items():
    print(f"{name:<22} {r['acc']*100:>8.1f}% {r['auc']:>9.3f} {r['cv_auc']:>9.3f}")
print("=" * 50)

# Detailed classification report for best model (Random Forest)
print("\n--- Detailed report: Random Forest ---")
print(classification_report(y_test, rf_pred, target_names=['Good Risk', 'Bad Risk']))


# =============================================================================
# STEP 7: FEATURE IMPORTANCE
# =============================================================================
# Which features matter most for predicting default?
# Random Forest gives us an "importance score" for each feature.
# This is critical for banks — they need to justify model decisions.

print("\n[STEP 7] Feature importance (Random Forest)")
importances = pd.Series(rf_model.feature_importances_, index=feature_cols)
top_features = importances.sort_values(ascending=False).head(10)
print("\n  Top 10 most important features:")
for feat, score in top_features.items():
    bar = '█' * int(score * 200)
    print(f"  {feat:<22} {score:.4f}  {bar}")


# =============================================================================
# STEP 8: PREDICT FOR A NEW BORROWER
# =============================================================================
# Now let's use our trained model to assess a hypothetical new loan applicant.
# We'll create two profiles — one likely Good risk, one likely Bad risk.

print("\n[STEP 8] Predicting risk for new borrowers")
print("-" * 50)

def predict_risk(profile_values, profile_name):
    """
    Given a borrower's feature values, predict their credit risk.
    profile_values: list of 20 values matching feature_cols order
    """
    # Create a DataFrame from the input (same structure as training data)
    profile_df = pd.DataFrame([profile_values], columns=feature_cols)

    # Encode categorical columns using the same encoding as training
    df_temp = pd.concat([X, profile_df], ignore_index=True)
    for col in categorical_cols:
        df_temp[col] = le.fit_transform(df_temp[col].astype(str))
    profile_encoded = df_temp.tail(1)[feature_cols]

    # Scale numeric columns
    profile_scaled = profile_encoded.copy()
    profile_scaled[numeric_cols] = scaler.transform(profile_encoded[numeric_cols])

    # Get prediction and probability
    pred  = rf_model.predict(profile_scaled)[0]
    proba = rf_model.predict_proba(profile_scaled)[0][1]

    label = "BAD RISK  ✗ (Likely to Default)" if pred == 1 else "GOOD RISK ✓ (Likely to Repay)"
    print(f"\n  Borrower: {profile_name}")
    print(f"  Prediction       : {label}")
    print(f"  Default Prob (PD): {proba*100:.1f}%")
    if pred == 1:
        print(f"  Recommendation   : DECLINE or require guarantor / collateral")
    else:
        print(f"  Recommendation   : APPROVE with standard interest rate")
    return pred, proba

# Profile 1 — Strong applicant (likely Good risk)
# Interpretation: Has a good checking account, short loan, excellent credit history,
# high savings, long employment, owns property, middle-aged
predict_risk(
    profile_values=['A14', 12, 'A34', 'A42', 2000, 'A65', 'A75',
                    2, 'A93', 'A101', 3, 'A121', 45, 'A143', 'A152',
                    1, 'A173', 1, 'A192', 'A201'],
    profile_name="Profile A — Strong applicant (good savings, long employment)"
)

# Profile 2 — Weak applicant (likely Bad risk)
# Interpretation: Negative checking account, long expensive loan, prior delays,
# no savings, recently employed, young, renting
predict_risk(
    profile_values=['A11', 60, 'A30', 'A40', 15000, 'A61', 'A71',
                    4, 'A91', 'A101', 1, 'A124', 22, 'A141', 'A153',
                    3, 'A171', 2, 'A191', 'A201'],
    profile_name="Profile B — Weak applicant (no savings, short employment, young)"
)


# =============================================================================
# STEP 9: GENERATE CHARTS AND SAVE AS PNG FILES
# =============================================================================

print("\n[STEP 9] Generating charts...")

plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(20, 16))
fig.suptitle('Credit Risk Analysis — German Credit Dataset\n', fontsize=16, fontweight='bold')

# ─── Chart 1: Target Distribution ──────────────────────────────────────────
ax1 = fig.add_subplot(3, 3, 1)
counts = y.value_counts()
bars = ax1.bar(['Good Risk (0)', 'Bad Risk (1)'], [counts[0], counts[1]],
               color=['#2ecc71', '#e74c3c'], edgecolor='white', linewidth=1.5)
for bar, count in zip(bars, [counts[0], counts[1]]):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f'{count}\n({count/len(y)*100:.0f}%)', ha='center', va='bottom', fontsize=10)
ax1.set_title('Target Distribution', fontweight='bold')
ax1.set_ylabel('Number of Applicants')

# ─── Chart 2: Credit Amount Distribution ───────────────────────────────────
ax2 = fig.add_subplot(3, 3, 2)
df_plot = df.copy()
good_amounts = df_plot[df_plot['target'] == 0]['credit_amount']
bad_amounts  = df_plot[df_plot['target'] == 1]['credit_amount']
ax2.hist(good_amounts, bins=30, alpha=0.6, color='#2ecc71', label='Good Risk', edgecolor='white')
ax2.hist(bad_amounts,  bins=30, alpha=0.6, color='#e74c3c', label='Bad Risk',  edgecolor='white')
ax2.set_title('Loan Amount Distribution', fontweight='bold')
ax2.set_xlabel('Credit Amount (DM)')
ax2.set_ylabel('Count')
ax2.legend()

# ─── Chart 3: Age Distribution ─────────────────────────────────────────────
ax3 = fig.add_subplot(3, 3, 3)
ax3.hist(df_plot[df_plot['target'] == 0]['age'], bins=25, alpha=0.6,
         color='#3498db', label='Good Risk', edgecolor='white')
ax3.hist(df_plot[df_plot['target'] == 1]['age'], bins=25, alpha=0.6,
         color='#e67e22', label='Bad Risk',  edgecolor='white')
ax3.set_title('Age Distribution by Risk', fontweight='bold')
ax3.set_xlabel('Age (years)')
ax3.set_ylabel('Count')
ax3.legend()

# ─── Chart 4: Model Comparison ─────────────────────────────────────────────
ax4 = fig.add_subplot(3, 3, 4)
model_names = list(results.keys())
aucs = [results[m]['auc'] for m in model_names]
accs = [results[m]['acc'] for m in model_names]
x = np.arange(len(model_names))
width = 0.35
ax4.bar(x - width/2, aucs, width, label='AUC-ROC', color='#9b59b6', alpha=0.85)
ax4.bar(x + width/2, accs, width, label='Accuracy', color='#1abc9c', alpha=0.85)
ax4.set_xticks(x)
ax4.set_xticklabels(model_names, rotation=10, ha='right')
ax4.set_ylim(0.5, 1.0)
ax4.axhline(0.75, color='red', linestyle='--', linewidth=1, alpha=0.6, label='AUC=0.75 benchmark')
ax4.set_title('Model Performance Comparison', fontweight='bold')
ax4.set_ylabel('Score')
ax4.legend(fontsize=9)

# ─── Chart 5: ROC Curves ───────────────────────────────────────────────────
ax5 = fig.add_subplot(3, 3, 5)
colors = ['#9b59b6', '#e67e22', '#e74c3c']
for (name, res), col in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res['proba'])
    ax5.plot(fpr, tpr, color=col, lw=2, label=f"{name} (AUC={res['auc']:.2f})")
ax5.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUC=0.50)')
ax5.set_xlabel('False Positive Rate')
ax5.set_ylabel('True Positive Rate')
ax5.set_title('ROC Curves — All Models', fontweight='bold')
ax5.legend(fontsize=8)

# ─── Chart 6: Confusion Matrix (Random Forest) ─────────────────────────────
ax6 = fig.add_subplot(3, 3, 6)
cm = confusion_matrix(y_test, rf_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Good Risk', 'Bad Risk'])
disp.plot(ax=ax6, colorbar=False, cmap='Blues')
ax6.set_title('Confusion Matrix — Random Forest', fontweight='bold')

# ─── Chart 7: Top 10 Feature Importances ───────────────────────────────────
ax7 = fig.add_subplot(3, 3, 7)
top10 = importances.sort_values(ascending=True).tail(10)
colors_bar = ['#e74c3c' if v > 0.07 else '#3498db' for v in top10.values]
ax7.barh(top10.index, top10.values, color=colors_bar, alpha=0.85)
ax7.set_title('Top 10 Feature Importances\n(Random Forest)', fontweight='bold')
ax7.set_xlabel('Importance Score')

# ─── Chart 8: Default Probability Distribution ─────────────────────────────
ax8 = fig.add_subplot(3, 3, 8)
ax8.hist(rf_proba[y_test == 0], bins=25, alpha=0.6, color='#2ecc71',
         label='Actual Good Risk', edgecolor='white')
ax8.hist(rf_proba[y_test == 1], bins=25, alpha=0.6, color='#e74c3c',
         label='Actual Bad Risk',  edgecolor='white')
ax8.axvline(0.5, color='black', linestyle='--', linewidth=1.5, label='Decision threshold 0.5')
ax8.set_title('PD Score Distribution\n(Random Forest)', fontweight='bold')
ax8.set_xlabel('Predicted Probability of Default')
ax8.set_ylabel('Count')
ax8.legend(fontsize=9)

# ─── Chart 9: Duration vs Credit Amount scatter ────────────────────────────
ax9 = fig.add_subplot(3, 3, 9)
good_df = df_plot[df_plot['target'] == 0]
bad_df  = df_plot[df_plot['target'] == 1]
ax9.scatter(good_df['duration'], good_df['credit_amount'],
            alpha=0.3, s=15, color='#2ecc71', label='Good Risk')
ax9.scatter(bad_df['duration'],  bad_df['credit_amount'],
            alpha=0.4, s=15, color='#e74c3c', label='Bad Risk')
ax9.set_title('Loan Duration vs Credit Amount', fontweight='bold')
ax9.set_xlabel('Duration (months)')
ax9.set_ylabel('Credit Amount (DM)')
ax9.legend()

plt.tight_layout()
output_path = 'credit_risk_analysis_charts.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"   ✓ Charts saved → {output_path}")
plt.close()


# =============================================================================
# STEP 10: FINAL SUMMARY
# =============================================================================

best_model = max(results, key=lambda m: results[m]['auc'])
best_auc   = results[best_model]['auc']

print("\n" + "=" * 65)
print("  FINAL SUMMARY")
print("=" * 65)
print(f"  Dataset          : German Credit Data (1,000 applicants)")
print(f"  Features used    : {len(feature_cols)}")
print(f"  Models trained   : Logistic Regression, Decision Tree, Random Forest")
print(f"  Best model       : {best_model}")
print(f"  Best AUC-ROC     : {best_auc:.3f}  (benchmark: 0.75+)")
print(f"  Chart saved to   : credit_risk_analysis_charts.png")
print("=" * 65)
print("""
  KEY METRICS EXPLAINED (for your report):
  ─────────────────────────────────────────
  PD  (Probability of Default) — How likely is this borrower to default?
  LGD (Loss Given Default)     — What % of the loan do we lose if they default?
  EAD (Exposure at Default)    — How much is outstanding when they default?
  EL  (Expected Loss)          — EL = PD × LGD × EAD  ← The key banking formula

  AUC-ROC > 0.75  → Model is useful for production credit decisions
  AUC-ROC > 0.80  → Good model
  AUC-ROC > 0.85  → Excellent model (rarely achieved on real data)
""")
print("  Script completed successfully! ✓")
print("=" * 65)
