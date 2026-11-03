# -*- coding: utf-8 -*-
"""Copy of Untitled22.ipynb

Nested cross-validation implementation of InteRA-BM.
"""
# Install dependencies separately if needed: pip install boruta shap rpy2 joblib
meta = pd.read_csv("metadata_enriched.csv", index_col=0)

asv = pd.read_csv("asv_clr_processed.csv", index_col=0)

print(meta.shape)
print(asv.shape)

meta.head()



import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    matthews_corrcoef
)

from boruta import BorutaPy

RANDOM_STATE = 42

meta = pd.read_csv("metadata_enriched.csv", index_col=0)
asv = pd.read_csv("asv_clr_processed.csv", index_col=0)

meta_enc = meta.copy()

if "Sex" in meta_enc.columns:
    meta_enc["Sex"] = meta_enc["Sex"].map({"F":1,"M":0})

encode_cols = [
    c for c in ["Diet","Location"]
    if c in meta_enc.columns
]

meta_enc = pd.get_dummies(
    meta_enc,
    columns=encode_cols,
    drop_first=False
)

bool_cols = meta_enc.select_dtypes(
    include="bool"
).columns

meta_enc[bool_cols] = meta_enc[
    bool_cols
].astype(int)

label_col = None

for c in meta_enc.columns:
    if "Group" in c and "HC" in c and "RA" in c:
        label_col = c
        break

meta_enc["label"] = (
    meta_enc[label_col]
    .map({1:0,2:1})
    .astype(int)
)

meta_enc = meta_enc.drop(
    columns=[
        c for c in [
            "Collection date",
            "SampleID",
            "Stool characteristics",
            label_col
        ]
        if c in meta_enc.columns
    ]
)

y = meta_enc["label"].values

X = pd.concat(
    [
        asv.reset_index(drop=True),
        meta_enc.drop(columns=["label"]).reset_index(drop=True)
    ],
    axis=1
)

print(X.shape)

outer_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

fold = 1

for train_idx, test_idx in outer_cv.split(X, y):

    print(f"\nFold {fold}")

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    # -------------------------
    # ExtraTrees INSIDE FOLD
    # -------------------------

    et = ExtraTreesClassifier(
        n_estimators=400,
        random_state=42,
        n_jobs=-1
    )

    et.fit(X_train, y_train)

    imp = pd.Series(
        et.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)

    top_features = imp.head(
        min(2500, len(imp))
    ).index

    X_train_top = X_train[top_features]
    X_test_top = X_test[top_features]

    # -------------------------
    # Boruta INSIDE FOLD
    # -------------------------

    boruta_est = ExtraTreesClassifier(
        n_estimators=600,
        random_state=42,
        n_jobs=-1
    )

    boruta = BorutaPy(
        estimator=boruta_est,
        n_estimators="auto",
        max_iter=30,
        random_state=42,
        verbose=0
    )

    boruta.fit(
        X_train_top.values,
        y_train
    )

    selected = X_train_top.columns[
        boruta.support_
    ]

    X_train_sel = X_train_top[selected]
    X_test_sel = X_test_top[selected]

    print(
        "Selected Features:",
        len(selected)
    )

    # -------------------------
    # RF
    # -------------------------

    rf = RandomForestClassifier(
        n_estimators=600,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    # -------------------------
    # Calibration
    # -------------------------

    cal = CalibratedClassifierCV(
        estimator=rf,
        method="sigmoid",
        cv=3
    )

    cal.fit(
        X_train_sel,
        y_train
    )

    y_prob = cal.predict_proba(
        X_test_sel
    )[:,1]

    y_pred = (
        y_prob >= 0.5
    ).astype(int)

    tn,fp,fn,tp = confusion_matrix(
        y_test,
        y_pred
    ).ravel()

    specificity = tn/(tn+fp)

    results.append({

        "ROC_AUC":
            roc_auc_score(
                y_test,
                y_prob
            ),

        "PR_AUC":
            average_precision_score(
                y_test,
                y_prob
            ),

        "Accuracy":
            accuracy_score(
                y_test,
                y_pred
            ),

        "Sensitivity":
            recall_score(
                y_test,
                y_pred
            ),

        "Specificity":
            specificity,

        "Precision":
            precision_score(
                y_test,
                y_pred
            ),

        "F1":
            f1_score(
                y_test,
                y_pred
            ),

        "MCC":
            matthews_corrcoef(
                y_test,
                y_pred
            )

    })

    fold += 1

results_df = pd.DataFrame(results)

display(results_df)

summary = pd.DataFrame({

    "Metric": results_df.columns,

    "Mean":
        results_df.mean().round(3),

    "SD":
        results_df.std().round(3)

})

display(summary)

summary.to_csv(
    "NestedCV_Integrated_Model.csv",
    index=False
)

import matplotlib.pyplot as plt

metrics = [
    "ROC-AUC",
    "PR-AUC",
    "Accuracy",
    "Sensitivity",
    "Specificity",
    "Precision",
    "F1",
    "MCC"
]

means = [
    0.906,
    0.890,
    0.819,
    0.796,
    0.839,
    0.810,
    0.803,
    0.636
]

stds = [
    0.014,
    0.018,
    0.014,
    0.026,
    0.030,
    0.027,
    0.015,
    0.029
]

plt.figure(figsize=(10,5))

bars = plt.bar(
    metrics,
    means,
    yerr=stds,
    capsize=5
)

plt.ylabel("Score")
plt.xlabel("Performance Metric")
plt.title(
    "Performance of the Integrated Clinical–Microbiome Model Under Nested Cross-Validation"
)

plt.ylim(0,1)

# Add values on top
for bar, value in zip(bars, means):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        value + 0.02,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "NestedCV_Performance.tiff",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

import pandas as pd
import matplotlib.pyplot as plt

results_df.boxplot(
    figsize=(10,5)
)

plt.title(
    "Distribution of Performance Metrics Across Nested Cross-Validation Folds"
)

plt.ylabel("Score")

plt.tight_layout()

plt.savefig(
    "NestedCV_Boxplot.tiff",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

from sklearn.metrics import roc_curve, auc

fold_aucs = results_df["ROC_AUC"].tolist()

print(fold_aucs)

mean_auc = np.mean(fold_aucs)
std_auc = np.std(fold_aucs)

print(mean_auc, std_auc)

from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

mean_fpr = np.linspace(0, 1, 100)

tprs = []
aucs = []

for fpr, tpr in fold_roc_data:

    interp_tpr = np.interp(
        mean_fpr,
        fpr,
        tpr
    )

    interp_tpr[0] = 0.0

    tprs.append(interp_tpr)

mean_tpr = np.mean(
    tprs,
    axis=0
)

std_tpr = np.std(
    tprs,
    axis=0
)

mean_tpr[-1] = 1.0

plt.figure(figsize=(6,5))

plt.plot(
    mean_fpr,
    mean_tpr,
    lw=2.5,
    label="Mean ROC (AUC = 0.906 ± 0.014)"
)

plt.fill_between(
    mean_fpr,
    np.maximum(mean_tpr - std_tpr, 0),
    np.minimum(mean_tpr + std_tpr, 1),
    alpha=0.25,
    label="±1 SD"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--",
    color="black"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "Mean ROC Across 5-Fold Nested CV"
)

plt.legend(
    loc="lower right"
)

plt.tight_layout()

plt.savefig(
    "Figure3A_MeanROC.tiff",
    dpi=600,
    bbox_inches="tight"
)

plt.show()

from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

prob_true, prob_pred = calibration_curve(
    all_y_true,
    all_y_prob,
    n_bins=10
)

plt.figure(figsize=(6,5))

plt.plot(
    prob_pred,
    prob_true,
    marker="o",
    linewidth=2,
    label="Nested CV Model"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--",
    color="black",
    label="Perfect Calibration"
)

plt.xlabel("Predicted Probability")
plt.ylabel("Observed Frequency")

plt.title(
    "Calibration Curve"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "Figure3B_Calibration.tiff",
    dpi=600,
    bbox_inches="tight"
)

plt.show()

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

precision, recall, _ = precision_recall_curve(
    all_y_true,
    all_y_prob
)

plt.figure(figsize=(6,5))

plt.plot(
    recall,
    precision,
    lw=2,
    label="Mean PR-AUC = 0.890 ± 0.018"
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title(
    "Precision–Recall Curve"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "Supplementary_Figure_S1_PRCurve.tiff",
    dpi=600,
    bbox_inches="tight"
)

plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(
    all_y_true,
    all_y_pred
)

plt.figure(figsize=(5,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cbar=False
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.title(
    "Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "Supplementary_Figure_S2_ConfusionMatrix.tiff",
    dpi=600,
    bbox_inches="tight"
)

plt.show()

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))

results_df.boxplot()

plt.ylabel("Performance Score")

plt.title(
    "Distribution of Performance Metrics Across Nested CV Folds"
)

plt.tight_layout()

plt.savefig(
    "Supplementary_Figure_S3_Boxplots.tiff",
    dpi=600,
    bbox_inches="tight"
)

plt.show()

# Clinical-only
X_clinical = meta_enc.drop(
    columns=["label"]
).copy()

# Microbiome-only
X_microbiome = asv.copy()

# Integrated
X_integrated = pd.concat(
    [
        asv.reset_index(drop=True),
        meta_enc.drop(
            columns=["label"]
        ).reset_index(drop=True)
    ],
    axis=1
)

# Integrated without Age/BMI
meta_no_age_bmi = meta_enc.drop(
    columns=[
        "label",
        "Age",
        "BMI"
    ],
    errors="ignore"
)

X_no_age_bmi = pd.concat(
    [
        asv.reset_index(drop=True),
        meta_no_age_bmi.reset_index(drop=True)
    ],
    axis=1
)

print(X_clinical.shape)
print(X_microbiome.shape)
print(X_integrated.shape)
print(X_no_age_bmi.shape)

def run_nested_cv(X, y, model_name):

    outer_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    results = []

    for fold,(train_idx,test_idx) in enumerate(
        outer_cv.split(X,y),
        start=1
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        # ExtraTrees

        et = ExtraTreesClassifier(
            n_estimators=400,
            random_state=42,
            n_jobs=-1
        )

        et.fit(
            X_train,
            y_train
        )

        imp = pd.Series(
            et.feature_importances_,
            index=X_train.columns
        ).sort_values(
            ascending=False
        )

        X_train_top = X_train[
            imp.head(
                min(2500,len(imp))
            ).index
        ]

        X_test_top = X_test[
            X_train_top.columns
        ]

        # Boruta

        boruta_est = ExtraTreesClassifier(
            n_estimators=600,
            random_state=42,
            n_jobs=-1
        )

        boruta = BorutaPy(
            boruta_est,
            n_estimators="auto",
            max_iter=30,
            random_state=42
        )

        boruta.fit(
            X_train_top.values,
            y_train
        )

        selected = X_train_top.columns[
            boruta.support_
        ]

        X_train_sel = X_train_top[selected]
        X_test_sel = X_test_top[selected]

        # RF

        rf = RandomForestClassifier(
            n_estimators=600,
            max_depth=15,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        cal = CalibratedClassifierCV(
            estimator=rf,
            method="sigmoid",
            cv=3
        )

        cal.fit(
            X_train_sel,
            y_train
        )

        y_prob = cal.predict_proba(
            X_test_sel
        )[:,1]

        y_pred = (
            y_prob >= 0.5
        ).astype(int)

        tn,fp,fn,tp = confusion_matrix(
            y_test,
            y_pred
        ).ravel()

        results.append({

            "ROC_AUC":
            roc_auc_score(
                y_test,
                y_prob
            ),

            "PR_AUC":
            average_precision_score(
                y_test,
                y_prob
            ),

            "Accuracy":
            accuracy_score(
                y_test,
                y_pred
            ),

            "F1":
            f1_score(
                y_test,
                y_pred
            )

        })

    results_df = pd.DataFrame(
        results
    )

    return {

        "Model":
        model_name,

        "ROC_AUC":
        f"{results_df['ROC_AUC'].mean():.3f} ± {results_df['ROC_AUC'].std():.3f}",

        "PR_AUC":
        f"{results_df['PR_AUC'].mean():.3f} ± {results_df['PR_AUC'].std():.3f}",

        "Accuracy":
        f"{results_df['Accuracy'].mean():.3f} ± {results_df['Accuracy'].std():.3f}",

        "F1":
        f"{results_df['F1'].mean():.3f} ± {results_df['F1'].std():.3f}"
    }

clinical = run_nested_cv(
    X_clinical,
    y,
    "Clinical-only"
)

microbiome = run_nested_cv(
    X_microbiome,
    y,
    "Microbiome-only"
)

integrated = run_nested_cv(
    X_integrated,
    y,
    "Integrated"
)

no_age_bmi = run_nested_cv(
    X_no_age_bmi,
    y,
    "Integrated_No_Age_BMI"
)

comparison = pd.DataFrame([

    clinical,
    microbiome,
    integrated,
    no_age_bmi

])

comparison

print(comparison.dtypes)

import pandas as pd

plot_df = comparison.copy()

for col in ["ROC_AUC", "PR_AUC", "Accuracy", "F1"]:

    plot_df[f"{col}_Mean"] = (
        plot_df[col]
        .str.split(" ± ")
        .str[0]
        .astype(float)
    )

    plot_df[f"{col}_SD"] = (
        plot_df[col]
        .str.split(" ± ")
        .str[1]
        .astype(float)
    )

plot_df

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(len(plot_df))

width = 0.25

fig, ax = plt.subplots(figsize=(10,6))

ax.bar(
    x-width,
    plot_df["ROC_AUC_Mean"],
    width,
    yerr=plot_df["ROC_AUC_SD"],
    capsize=5,
    label="ROC-AUC"
)

ax.bar(
    x,
    plot_df["PR_AUC_Mean"],
    width,
    yerr=plot_df["PR_AUC_SD"],
    capsize=5,
    label="PR-AUC"
)

ax.bar(
    x+width,
    plot_df["Accuracy_Mean"],
    width,
    yerr=plot_df["Accuracy_SD"],
    capsize=5,
    label="Accuracy"
)

ax.set_xticks(x)

ax.set_xticklabels(
    plot_df["Model"],
    rotation=15,
    ha="right"
)

ax.set_ylim(0.6,1.0)

ax.set_ylabel("Performance Score")

ax.set_title(
    "Performance Comparison of Clinical, Microbiome and Integrated Models\n(Nested Cross-Validation)"
)

ax.legend()

plt.tight_layout()

plt.savefig(
    "Figure4_ModelComparison_ErrorBars.tiff",
    dpi=600,
    bbox_inches="tight"
)

plt.show()

import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(8,5))

bars = plt.bar(
    plot_df["Model"],
    plot_df["ROC_AUC_Mean"],
    yerr=plot_df["ROC_AUC_SD"],
    capsize=6
)

plt.ylabel(
    "ROC-AUC"
)

plt.xlabel(
    "Model"
)

plt.title(
    "ROC-AUC Comparison Across Nested Cross-Validation Models"
)

plt.ylim(
    0.7,
    1.0
)

for bar,val,sd in zip(
    bars,
    plot_df["ROC_AUC_Mean"],
    plot_df["ROC_AUC_SD"]
):

    plt.text(
        bar.get_x()+bar.get_width()/2,
        val+0.01,
        f"{val:.3f}±{sd:.3f}",
        ha="center",
        fontsize=9
    )

plt.tight_layout()

plt.savefig(
    "Figure4A_ROCAUC_ErrorBars.tiff",
    dpi=600,
    bbox_inches="tight"
)

plt.show()

import numpy as np
import matplotlib.pyplot as plt

metrics = [
    "ROC_AUC_Mean",
    "PR_AUC_Mean",
    "Accuracy_Mean",
    "F1_Mean"
]

labels = [
    "ROC-AUC",
    "PR-AUC",
    "Accuracy",
    "F1"
]

angles = np.linspace(
    0,
    2*np.pi,
    len(labels),
    endpoint=False
)

angles = np.concatenate(
    (angles,[angles[0]])
)

fig = plt.figure(figsize=(8,8))

ax = plt.subplot(
    111,
    polar=True
)

for idx,row in plot_df.iterrows():

    values = [
        row["ROC_AUC_Mean"],
        row["PR_AUC_Mean"],
        row["Accuracy_Mean"],
        row["F1_Mean"]
    ]

    values += values[:1]

    ax.plot(
        angles,
        values,
        linewidth=2,
        label=row["Model"]
    )

    ax.fill(
        angles,
        values,
        alpha=0.08
    )

ax.set_xticks(
    angles[:-1]
)

ax.set_xticklabels(
    labels,
    fontsize=11
)

ax.set_ylim(
    0.6,
    1.0
)

plt.title(
    "Comparison of Model Performance Metrics",
    pad=25,
    fontsize=14
)

plt.legend(
    bbox_to_anchor=(1.3,1.1)
)

plt.tight_layout()

plt.savefig(
    "Supplementary_RadarChart.tiff",
    dpi=600,
    bbox_inches="tight"
)

plt.show()

print(plot_df[[
    "Model",
    "ROC_AUC_Mean",
    "PR_AUC_Mean",
    "Accuracy_Mean",
    "F1_Mean"
]])

ax.set_ylim(0.6,1.0)

ax.set_ylim(0,1.0)

ax.set_yticks([0.2,0.4,0.6,0.8,1.0])

import matplotlib.pyplot as plt

plot_sorted = plot_df.sort_values(
    "ROC_AUC_Mean"
)

plt.figure(figsize=(8,5))

plt.barh(
    plot_sorted["Model"],
    plot_sorted["ROC_AUC_Mean"],
    xerr=plot_sorted["ROC_AUC_SD"],
    capsize=5
)

plt.xlabel(
    "ROC-AUC"
)

plt.title(
    "Predictive Performance of Clinical, Microbiome and Integrated Models"
)

plt.xlim(
    0.7,
    1.0
)

plt.tight_layout()

plt.savefig(
    "Figure4A_HorizontalROCComparison.tiff",
    dpi=600,
    bbox_inches="tight"
)

plt.show()