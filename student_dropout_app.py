import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import scale
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import RidgeClassifier, LogisticRegression, ElasticNet, Lasso
from sklearn.metrics import (r2_score, mean_squared_error, confusion_matrix,
                             classification_report, precision_score,
                             recall_score, f1_score)


st.set_page_config(page_title="Student Dropout Analysis", layout="wide")
st.title("🎓 Student Dropout Analysis")
st.markdown("---")

@st.cache_data
def load_data():
    # ✅ Relative path — place the CSV in the same folder as this script
    df = pd.read_csv("student_dropout_dataset.csv")
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("❌ 'student_dropout_dataset.csv' not found. Place it in the same folder as this script.")
    st.stop()

st.header("1. Data Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Total Students", df.shape[0])
col2.metric("Features", df.shape[1] - 1)
col3.metric("Missing Values", int(df.isnull().sum().sum()))

st.subheader("First 5 Rows")
st.dataframe(df.head(), use_container_width=True)

st.subheader("Basic Statistics")
st.dataframe(df.describe(), use_container_width=True)

st.header("2. Target Variable Distribution")

target_counts = df["Target"].value_counts()

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].bar(target_counts.index, target_counts.values,
          color=["#e74c3c", "#2ecc71", "#3498db"])
ax[0].set_title("Student Outcome Counts")
ax[0].set_xlabel("Outcome")
ax[0].set_ylabel("Count")
for i, v in enumerate(target_counts.values):
    ax[0].text(i, v + 20, str(v), ha="center", fontweight="bold")

ax[1].pie(target_counts.values, labels=target_counts.index,
          autopct="%1.1f%%", colors=["#e74c3c", "#2ecc71", "#3498db"])
ax[1].set_title("Student Outcome Proportions")

plt.tight_layout()
st.pyplot(fig)
plt.close()

st.header("3. Binary Classification Setup (Dropout vs Graduate)")
st.info("Enrolled students are removed. Only Dropout (1) vs Graduate (0) are kept for binary classification.")

df_binary = df[df["Target"] != "Enrolled"].copy()
df_binary["Target_binary"] = (df_binary["Target"] == "Dropout").astype(int)

n_grad  = int((df_binary["Target_binary"] == 0).sum())
n_drop  = int((df_binary["Target_binary"] == 1).sum())
n_total = int(df_binary.shape[0])

breakdown = pd.DataFrame({
    "Class":      ["Graduate (0)", "Dropout (1)", "Total"],
    "Count":      [n_grad, n_drop, n_total],
    "Percentage": [f"{n_grad/n_total*100:.1f}%", f"{n_drop/n_total*100:.1f}%", "100%"]
})
st.table(breakdown)

fig2, ax2 = plt.subplots(figsize=(5, 3))
ax2.bar(["Graduate (0)", "Dropout (1)"], [n_grad, n_drop],
        color=["#2ecc71", "#e74c3c"])
ax2.set_title("Binary Target Distribution")
ax2.set_ylabel("Count")
for i, v in enumerate([n_grad, n_drop]):
    ax2.text(i, v + 10, str(v), ha="center", fontweight="bold")
plt.tight_layout()
st.pyplot(fig2)
plt.close()

st.header("4. Correlation Heatmap")

key_features = [
    "Age at enrollment", "Admission grade", "Previous qualification (grade)",
    "Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (approved)", "Curricular units 2nd sem (grade)",
    "Tuition fees up to date", "Scholarship holder", "Debtor",
    "Unemployment rate", "GDP", "Target_binary"
]
# Keep only features that exist in the dataframe
key_features = [f for f in key_features if f in df_binary.columns]

corr_df = df_binary[key_features].corr()

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, ax=ax)
ax.set_title("Correlation Heatmap of Key Features", fontsize=14)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.header("5. Exploratory Data Analysis")

plot_features = [
    ("Age at enrollment",                   "Age at Enrollment"),
    ("Admission grade",                     "Admission Grade"),
    ("Curricular units 1st sem (approved)", "1st Sem Units Approved"),
    ("Curricular units 2nd sem (approved)", "2nd Sem Units Approved"),
    ("Curricular units 1st sem (grade)",    "1st Sem Grade"),
    ("Curricular units 2nd sem (grade)",    "2nd Sem Grade"),
]
plot_features = [(c, t) for c, t in plot_features if c in df_binary.columns]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
colors_eda = {"Dropout": "#e74c3c", "Graduate": "#2ecc71"}

for i, (col, title) in enumerate(plot_features):
    for label in ["Dropout", "Graduate"]:
        subset = df_binary[df_binary["Target"] == label][col]
        axes[i].hist(subset, bins=20, alpha=0.6, label=label, color=colors_eda[label])
    axes[i].set_title(title)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")
    axes[i].legend()

for j in range(len(plot_features), len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Feature Distributions: Dropout vs Graduate", fontsize=14)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.header("6. Variance Inflation Factor (VIF) – Multicollinearity Check")
st.markdown("VIF > 10 indicates high multicollinearity. These features are dropped before modelling.")

vif_features = [
    "Age at enrollment", "Admission grade", "Previous qualification (grade)",
    "Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (approved)", "Curricular units 2nd sem (grade)",
    "Tuition fees up to date", "Scholarship holder", "Debtor",
    "Unemployment rate", "GDP"
]
vif_features = [f for f in vif_features if f in df_binary.columns]

X_vif        = df_binary[vif_features].dropna()
X_vif_scaled = pd.DataFrame(scale(X_vif), columns=vif_features)

vif_data = pd.DataFrame({
    "Feature": vif_features,
    "VIF":     [variance_inflation_factor(X_vif_scaled.values, i)
                for i in range(X_vif_scaled.shape[1])]
})
vif_data = vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)

col1, col2 = st.columns(2)
with col1:
    st.dataframe(
        vif_data.style.background_gradient(cmap="RdYlGn_r", subset=["VIF"]),
        use_container_width=True
    )
with col2:
    fig, ax = plt.subplots(figsize=(6, 5))
    colors_vif = ["#e74c3c" if v > 10 else "#2ecc71" for v in vif_data["VIF"]]
    ax.barh(vif_data["Feature"], vif_data["VIF"], color=colors_vif)
    ax.axvline(x=10, color="black", linestyle="--", label="VIF = 10 threshold")
    ax.set_title("VIF Values")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

high_vif       = vif_data[vif_data["VIF"] > 10]["Feature"].tolist()
final_features = [f for f in vif_features if f not in high_vif]

if high_vif:
    st.warning(f"High VIF features (dropped for modelling): {high_vif}")
else:
    st.success("No features with VIF > 10. All features kept.")

X_model  = df_binary[final_features].copy()
y_model  = df_binary["Target_binary"].copy()
mask     = X_model.notna().all(axis=1)
X_model  = X_model[mask]
y_model  = y_model[mask]
X_scaled = pd.DataFrame(scale(X_model), columns=final_features)

# Reset index so iloc-based K-Fold splits work correctly
X_scaled = X_scaled.reset_index(drop=True)
y_model  = y_model.reset_index(drop=True)

k     = 5
kfold = KFold(n_splits=k, random_state=0, shuffle=True)

st.header("7. Logistic Regression – Statsmodels Summary")

X_sm        = sm.add_constant(X_scaled.values)   # numpy array avoids index mismatches
logit_model = sm.Logit(y_model.values, X_sm)
result      = logit_model.fit(disp=0)

st.subheader("Model Fit Metrics")
fit_col1, fit_col2, fit_col3 = st.columns(3)
fit_col1.metric("Log-Likelihood",        f"{result.llf:.2f}")
fit_col2.metric("AIC",                   f"{result.aic:.2f}")
fit_col3.metric("Pseudo R² (McFadden)", f"{result.prsquared:.4f}")

conf_int = result.conf_int()  # shape (n_params, 2) numpy array
summary_df = pd.DataFrame({
    "Feature":        ["const"] + final_features,
    "Coefficient":    np.round(result.params, 4),
    "Std Error":      np.round(result.bse, 4),
    "z-value":        np.round(result.tvalues, 4),
    "P-value":        np.round(result.pvalues, 4),
    "95% CI Lower":   np.round(conf_int[:, 0], 4),
    "95% CI Upper":   np.round(conf_int[:, 1], 4),
})
summary_df["Significant (p<0.05)"] = summary_df["P-value"] < 0.05

st.subheader("Coefficients Table")
st.dataframe(
    summary_df.style.map(
        lambda v: "background-color: #d4efdf" if v is True
                  else ("background-color: #fadbd8" if v is False else ""),
        subset=["Significant (p<0.05)"]
    ).format({
        "Coefficient":  "{:.4f}",
        "Std Error":    "{:.4f}",
        "z-value":      "{:.4f}",
        "P-value":      "{:.4f}",
        "95% CI Lower": "{:.4f}",
        "95% CI Upper": "{:.4f}",
    }),
    use_container_width=True
)

fig, ax = plt.subplots(figsize=(8, 5))
colors_coef = ["#2ecc71" if p < 0.05 else "#95a5a6" for p in summary_df["P-value"]]
ax.barh(summary_df["Feature"], summary_df["Coefficient"], color=colors_coef)
ax.axvline(x=0, color="black", linestyle="--")
ax.set_title("Logistic Regression Coefficients\n(Green = Significant at p<0.05)")
ax.set_xlabel("Coefficient Value")
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.header("8. Logistic Regression – K-Fold Cross Validation")

log_model  = LogisticRegression(max_iter=1000)
log_scores = cross_val_score(log_model, X_scaled, y_model, cv=kfold, scoring="accuracy")

log_df = pd.DataFrame({"Fold": [f"Fold {i+1}" for i in range(k)], "Accuracy": log_scores})
st.dataframe(log_df, use_container_width=True)

c1, c2 = st.columns(2)
c1.metric("Mean CV Accuracy", f"{log_scores.mean():.4f}")
c2.metric("Std Dev",          f"{log_scores.std():.4f}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(log_df["Fold"], log_df["Accuracy"], color="#3498db", alpha=0.8)
ax.axhline(y=log_scores.mean(), color="red", linestyle="--",
           label=f"Mean = {log_scores.mean():.3f}")
ax.set_ylim(0, 1)
ax.set_title("Logistic Regression – K-Fold Accuracy")
ax.set_ylabel("Accuracy")
ax.legend()
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.header("9. Ridge Classifier – K-Fold Cross Validation")

ridge_model  = RidgeClassifier()
ridge_scores = cross_val_score(ridge_model, X_scaled, y_model, cv=kfold, scoring="accuracy")

ridge_df = pd.DataFrame({"Fold": [f"Fold {i+1}" for i in range(k)], "Accuracy": ridge_scores})
st.dataframe(ridge_df, use_container_width=True)

c1, c2 = st.columns(2)
c1.metric("Ridge Mean CV Accuracy", f"{ridge_scores.mean():.4f}")
c2.metric("Ridge Std Dev",          f"{ridge_scores.std():.4f}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(ridge_df["Fold"], ridge_df["Accuracy"], color="#e67e22", alpha=0.8)
ax.axhline(y=ridge_scores.mean(), color="red", linestyle="--",
           label=f"Mean = {ridge_scores.mean():.3f}")
ax.set_ylim(0, 1)
ax.set_title("Ridge Classifier – K-Fold Accuracy")
ax.set_ylabel("Accuracy")
ax.legend()
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.header("10. Lasso – K-Fold Cross Validation")
st.info("Lasso is a regression model. Predictions are thresholded at 0.5 to get classification accuracy.")

lasso_model      = Lasso(alpha=0.01, max_iter=10000)
lasso_acc_scores = []

for train_idx, test_idx in kfold.split(X_scaled):
    X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
    y_train, y_test = y_model.iloc[train_idx],  y_model.iloc[test_idx]
    lasso_model.fit(X_train, y_train)
    y_pred = (lasso_model.predict(X_test) >= 0.5).astype(int)
    lasso_acc_scores.append((y_pred == y_test.values).mean())

lasso_acc_scores = np.array(lasso_acc_scores)
lasso_df = pd.DataFrame({"Fold": [f"Fold {i+1}" for i in range(k)], "Accuracy": lasso_acc_scores})
st.dataframe(lasso_df, use_container_width=True)

c1, c2 = st.columns(2)
c1.metric("Lasso Mean CV Accuracy", f"{lasso_acc_scores.mean():.4f}")
c2.metric("Lasso Std Dev",          f"{lasso_acc_scores.std():.4f}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(lasso_df["Fold"], lasso_df["Accuracy"], color="#9b59b6", alpha=0.8)
ax.axhline(y=lasso_acc_scores.mean(), color="red", linestyle="--",
           label=f"Mean = {lasso_acc_scores.mean():.3f}")
ax.set_ylim(0, 1)
ax.set_title("Lasso – K-Fold Accuracy")
ax.set_ylabel("Accuracy")
ax.legend()
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.header("11. Elastic Net – K-Fold Cross Validation")
st.info("Elastic Net combines Ridge (L2) + Lasso (L1) penalties. l1_ratio=0.5 means equal mix of both.")

enet_model      = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)
enet_acc_scores = []

for train_idx, test_idx in kfold.split(X_scaled):
    X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
    y_train, y_test = y_model.iloc[train_idx],  y_model.iloc[test_idx]
    enet_model.fit(X_train, y_train)
    y_pred = (enet_model.predict(X_test) >= 0.5).astype(int)
    enet_acc_scores.append((y_pred == y_test.values).mean())

enet_acc_scores = np.array(enet_acc_scores)
enet_df = pd.DataFrame({"Fold": [f"Fold {i+1}" for i in range(k)], "Accuracy": enet_acc_scores})
st.dataframe(enet_df, use_container_width=True)

c1, c2 = st.columns(2)
c1.metric("Elastic Net Mean CV Accuracy", f"{enet_acc_scores.mean():.4f}")
c2.metric("Elastic Net Std Dev",          f"{enet_acc_scores.std():.4f}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(enet_df["Fold"], enet_df["Accuracy"], color="#1abc9c", alpha=0.8)
ax.axhline(y=enet_acc_scores.mean(), color="red", linestyle="--",
           label=f"Mean = {enet_acc_scores.mean():.3f}")
ax.set_ylim(0, 1)
ax.set_title("Elastic Net – K-Fold Accuracy")
ax.set_ylabel("Accuracy")
ax.legend()
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.header("12. R² and MSE – All Models")
st.markdown("Each model is fitted on the full scaled dataset and scored against the binary target.")

# Fresh model instances to avoid state leakage
log_m   = LogisticRegression(max_iter=1000)
ridge_m = RidgeClassifier()
lasso_m = Lasso(alpha=0.01, max_iter=10000)
enet_m  = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)

log_m.fit(X_scaled,   y_model)
ridge_m.fit(X_scaled, y_model)
lasso_m.fit(X_scaled, y_model)
enet_m.fit(X_scaled,  y_model)

log_pred   = log_m.predict(X_scaled).astype(int)
ridge_pred = ridge_m.predict(X_scaled).astype(int)
lasso_pred = lasso_m.predict(X_scaled)
enet_pred  = enet_m.predict(X_scaled)

log_r2    = r2_score(y_model, log_pred)
ridge_r2  = r2_score(y_model, ridge_pred)
lasso_r2  = r2_score(y_model, lasso_pred)
enet_r2   = r2_score(y_model, enet_pred)

log_mse   = mean_squared_error(y_model, log_pred)
ridge_mse = mean_squared_error(y_model, ridge_pred)
lasso_mse = mean_squared_error(y_model, lasso_pred)
enet_mse  = mean_squared_error(y_model, enet_pred)

metrics_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Ridge Classifier", "Lasso", "Elastic Net"],
    "R²":    [log_r2,  ridge_r2,  lasso_r2,  enet_r2],
    "MSE":   [log_mse, ridge_mse, lasso_mse, enet_mse]
})

st.dataframe(
    metrics_df.style
        .format({"R²": "{:.4f}", "MSE": "{:.4f}"})
        .highlight_max(subset=["R²"],  color="#d4efdf")
        .highlight_min(subset=["MSE"], color="#d4efdf"),
    use_container_width=True
)

bar_colors = ["#3498db", "#e67e22", "#9b59b6", "#1abc9c"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(metrics_df["Model"], metrics_df["R²"], color=bar_colors, alpha=0.85)
axes[0].set_title("R² Comparison (higher = better)")
axes[0].set_ylabel("R²")
axes[0].set_ylim(0, 1)
plt.setp(axes[0].get_xticklabels(), rotation=15, ha="right")
for i, v in enumerate(metrics_df["R²"]):
    axes[0].text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")

axes[1].bar(metrics_df["Model"], metrics_df["MSE"], color=bar_colors, alpha=0.85)
axes[1].set_title("MSE Comparison (lower = better)")
axes[1].set_ylabel("MSE")
plt.setp(axes[1].get_xticklabels(), rotation=15, ha="right")
for i, v in enumerate(metrics_df["MSE"]):
    axes[1].text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()
st.pyplot(fig)
plt.close()

st.header("13. Final Model Comparison – Which is Best?")

comparison_df = pd.DataFrame({
    "Model":            ["Logistic Regression", "Ridge Classifier", "Lasso", "Elastic Net"],
    "Mean CV Accuracy": [log_scores.mean(), ridge_scores.mean(),
                         lasso_acc_scores.mean(), enet_acc_scores.mean()],
    "Std Dev":          [log_scores.std(),  ridge_scores.std(),
                         lasso_acc_scores.std(), enet_acc_scores.std()],
    "R²":               [log_r2,  ridge_r2,  lasso_r2,  enet_r2],
    "MSE":              [log_mse, ridge_mse, lasso_mse, enet_mse],
})

best_idx  = comparison_df["Mean CV Accuracy"].idxmax()
best_name = comparison_df.loc[best_idx, "Model"]
best_acc  = comparison_df.loc[best_idx, "Mean CV Accuracy"]

st.dataframe(
    comparison_df.style
        .format({"Mean CV Accuracy": "{:.4f}", "Std Dev": "{:.4f}",
                 "R²": "{:.4f}", "MSE": "{:.4f}"})
        .highlight_max(subset=["Mean CV Accuracy", "R²"], color="#d4efdf")
        .highlight_min(subset=["MSE", "Std Dev"],          color="#d4efdf"),
    use_container_width=True
)

st.success(f"🏆 Best Model: **{best_name}** — CV Accuracy = {best_acc:.4f}")

fig, ax = plt.subplots(figsize=(10, 5))
x    = np.arange(len(comparison_df))
bars = ax.bar(comparison_df["Model"], comparison_df["Mean CV Accuracy"],
              color=bar_colors, alpha=0.85, width=0.5)
ax.errorbar(x, comparison_df["Mean CV Accuracy"], yerr=comparison_df["Std Dev"],
            fmt="none", color="black", capsize=5)
ax.set_ylim(0.5, 1.0)
ax.set_title("Final Model Accuracy Comparison (with Std Dev error bars)")
ax.set_ylabel("Mean CV Accuracy")
for bar, val in zip(bars, comparison_df["Mean CV Accuracy"]):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
            f"{val:.4f}", ha="center", fontweight="bold")
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.header("14. Confusion Matrix – All 4 Models")
st.markdown("Each model is fitted on the full dataset and predictions are compared against actual labels.")
st.info("Rows = Actual class | Columns = Predicted class | 0 = Graduate | 1 = Dropout")

# Reuse predictions already computed above; threshold regressors at 0.5
all_preds = {
    "Logistic Regression": log_pred,
    "Ridge Classifier":    ridge_pred,
    "Lasso":               (lasso_pred >= 0.5).astype(int),
    "Elastic Net":         (enet_pred  >= 0.5).astype(int),
}

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, (name, preds) in zip(axes, all_preds.items()):
    cm = confusion_matrix(y_model, preds)
    sns.heatmap(
        cm, annot=True, fmt="d", ax=ax,
        cmap="Blues", linewidths=0.5,
        xticklabels=["Graduate", "Dropout"],
        yticklabels=["Graduate", "Dropout"],
        annot_kws={"size": 13, "weight": "bold"}
    )
    ax.set_title(name, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual",    fontsize=10)

plt.suptitle("Confusion Matrices – All Models", fontsize=14)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("""
**How to read the confusion matrix:**
- **Top-left (TN):** Graduates correctly predicted as Graduate  
- **Top-right (FP):** Graduates wrongly predicted as Dropout  
- **Bottom-left (FN):** Dropouts wrongly predicted as Graduate ← *costly error*  
- **Bottom-right (TP):** Dropouts correctly predicted as Dropout ✅
""")

st.subheader("Precision, Recall & F1-Score – All Models")
st.markdown("Scores shown for the **Dropout class (1)** — the minority and more important class.")

prf_rows = []
for name, preds in all_preds.items():
    prf_rows.append({
        "Model":     name,
        "Precision": round(precision_score(y_model, preds, zero_division=0), 4),
        "Recall":    round(recall_score(y_model, preds,    zero_division=0), 4),
        "F1-Score":  round(f1_score(y_model, preds,        zero_division=0), 4),
        "Accuracy":  round((preds == y_model.values).mean(), 4),
    })

prf_df = pd.DataFrame(prf_rows)
st.dataframe(
    prf_df.style
        .format({"Precision": "{:.4f}", "Recall": "{:.4f}",
                 "F1-Score":  "{:.4f}", "Accuracy": "{:.4f}"})
        .highlight_max(subset=["Precision", "Recall", "F1-Score", "Accuracy"], color="#d4efdf")
        .highlight_min(subset=["Precision", "Recall", "F1-Score", "Accuracy"], color="#fadbd8"),
    use_container_width=True
)

x     = np.arange(len(prf_df))
width = 0.22
fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(x - width, prf_df["Precision"], width, label="Precision", color="#3498db", alpha=0.85)
ax.bar(x,         prf_df["Recall"],    width, label="Recall",    color="#e74c3c", alpha=0.85)
ax.bar(x + width, prf_df["F1-Score"],  width, label="F1-Score",  color="#2ecc71", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(prf_df["Model"], rotation=10, ha="right")
ax.set_ylim(0, 1.1)
ax.set_title("Precision, Recall & F1-Score per Model (Dropout class)")
ax.set_ylabel("Score")
ax.legend()
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.subheader("Full Classification Report – All Models")

all_reports = {
    name: classification_report(y_model, preds,
                                target_names=["Graduate", "Dropout"],
                                output_dict=True)
    for name, preds in all_preds.items()
}

for name, report_dict in all_reports.items():
    with st.expander(f"📋 {name} — Classification Report"):
        report_df = pd.DataFrame(report_dict).transpose()
        report_df = report_df.loc[["Graduate", "Dropout", "macro avg", "weighted avg"]]
        report_df = report_df[["precision", "recall", "f1-score", "support"]].round(4)
        report_df.columns = ["Precision", "Recall", "F1-Score", "Support"]
        report_df["Support"] = report_df["Support"].fillna(0).astype(int)
        st.dataframe(
            report_df.style
                .format({"Precision": "{:.4f}", "Recall": "{:.4f}", "F1-Score": "{:.4f}"})
                .highlight_max(subset=["Precision", "Recall", "F1-Score"], color="#d4efdf"),
            use_container_width=True
        )

st.markdown("---")
st.caption("Student Dropout Analysis | pandas · numpy · matplotlib · seaborn · sklearn · statsmodels")