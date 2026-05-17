
#  IMPORT LIBRARIES
# ─────────────────────────────────────────────────────────────────
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')                    # no pop-up windows in VS Code
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.linear_model    import LogisticRegression
from sklearn.svm             import SVC
from sklearn.ensemble        import RandomForestClassifier
from sklearn.neural_network  import MLPClassifier
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from imblearn.over_sampling  import SMOTE

#  CREATE OUTPUT FOLDER

os.makedirs('outputs', exist_ok=True)

#  GLOBAL FIGURE STYLE

plt.rcParams.update({
    'font.family'       : 'serif',
    'font.size'         : 11,
    'axes.titlesize'    : 12,
    'axes.titleweight'  : 'bold',
    'axes.labelweight'  : 'bold',
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'figure.dpi'        : 150,
    'savefig.dpi'       : 150,
    'savefig.bbox'      : 'tight'
})

COLORS = ['#2E75B6', '#C00000', '#375623', '#7030A0']
BLUE   = '#2E75B6'
RED    = '#C00000'

print()
print("=" * 65)
print("  TOOL WEAR FAILURE PREDICTION PIPELINE")
print("  Type H Filtered | AI4I 2020 Predictive Maintenance Dataset")
print("=" * 65)


#  STEP 1 — LOAD DATASET


CSV_PATH = 'ai4i2020.csv'

if not os.path.exists(CSV_PATH):
    print()
    print("  !! ERROR: Dataset file not found !!")
    print(f"  Expected: '{CSV_PATH}' in the same folder as main.py")
    print()
    print("  HOW TO FIX:")
    print("  1. Go to https://archive.ics.uci.edu/dataset/601")
    print("  2. Click Download and extract the zip file")
    print("  3. Rename the CSV to  ai4i2020.csv")
    print("  4. Put it in the same folder as  main.py")
    sys.exit()

df = pd.read_csv(CSV_PATH)
print(f"\n  Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")


#  STEP 2 — TYPE H FILTER

df_h = df[df['Type'] == 'H'].reset_index(drop=True)

FEATURES = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]

X = df_h[FEATURES]
y = df_h['TWF']


#  STEP 3 — TRAIN-TEST SPLIT  (80/20, stratified)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

#  STEP 4 — STANDARD SCALING

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)


#  STEP 5 — SMOTE  (applied to training set only)


smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_sc, y_train)

#  STEP 6 — TRAIN ALL FOUR MODELS


models = {
    'Logistic Regression' : LogisticRegression(
                                random_state=42, max_iter=1000),
    'SVC'                 : SVC(
                                kernel='rbf', probability=True,
                                random_state=42),
    'Random Forest'       : RandomForestClassifier(
                                n_estimators=100, random_state=42),
    'MLP'                 : MLPClassifier(
                                hidden_layer_sizes=(64, 32),
                                max_iter=500, random_state=42,
                                early_stopping=True)
}

results = {}
cms     = {}
probs   = {}

print("\n  Training models...")
for name, model in models.items():
    model.fit(X_train_sm, y_train_sm)
    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]

    results[name] = {
        'Accuracy' : round(accuracy_score(y_test, y_pred),                   4),
        'Precision': round(precision_score(y_test, y_pred, zero_division=0),  4),
        'Recall'   : round(recall_score(y_test, y_pred,    zero_division=0),  4),
        'F1-Score' : round(f1_score(y_test, y_pred,        zero_division=0),  4),
        'AUC-ROC'  : round(roc_auc_score(y_test, y_prob),                    4)
    }
    cms[name]   = confusion_matrix(y_test, y_pred)
    probs[name] = y_prob
    print(f"    [DONE] {name}")

rf          = models['Random Forest']
importances = pd.Series(
    rf.feature_importances_, index=FEATURES
).sort_values(ascending=False)

print("\n  All models trained.\n")


#  TABLES


#  TABLE I

print("=" * 65)
print("  TABLE I — Dataset Summary Before and After Type H Filtering")
print()
print("  PLACE IN PAPER:")
print("  Section III-B  |  After the 2nd paragraph")
print("  Paragraph ends: '...Table I summarizes the dataset")
print("  structure before and after the Type H filtering step.'")
print("=" * 65)

table_I = pd.DataFrame({
    'Parameter'      : [
        'Total Observations',
        'TWF = 0  (No Failure)',
        'TWF = 1  (Failure)',
        'Class Imbalance Ratio',
        'Training Set (80%)',
        'Test Set (20%)',
        'After SMOTE (Training)',
        'Input Features'
    ],
    'Full Dataset'   : ['10,000','—','—','—','—','—','—','5'],
    'Type H Filtered': [
        f'{len(df_h)}',
        f'{(y==0).sum()}  (92.30%)',
        f'{(y==1).sum()}  (7.70%)',
        f'{int((y==0).sum()/(y==1).sum())}:1',
        f'{len(X_train)} obs  /  {y_train.sum()} failures',
        f'{len(X_test)} obs  /  {y_test.sum()} failures',
        f'{len(X_train_sm)} obs  /  {y_train_sm.sum()} failures',
        '5'
    ]
})
print(table_I.to_string(index=False))
print()

#  TABLE II

print("=" * 65)
print("  TABLE II — Descriptive Statistics  (Type H Filtered,  n = 961)")
print()
print("  PLACE IN PAPER:")
print("  Section III-C  |  After the 1st paragraph")
print("  Paragraph ends: '...confirming that all features fall")
print("  within physically plausible operating ranges consistent")
print("  with industrial CNC milling conditions.'")
print("=" * 65)

desc = df_h[FEATURES + ['TWF']].describe().round(4)
desc.columns = [
    'Air Temp [K]', 'Proc Temp [K]',
    'Rot Speed [rpm]', 'Torque [Nm]',
    'Tool Wear [min]', 'TWF'
]
print(desc.to_string())
print()

#  TABLE III

print("=" * 65)
print("  TABLE III — Comparison of Model Performance for TWF Prediction")
print("              (Type H Filtered)")
print()
print("  PLACE IN PAPER:")
print("  Section IV-C  |  After the 1st paragraph")
print("  Paragraph ends: '...The highlighted row indicates the")
print("  best-performing model.'")
print("  NOTE: Highlight the Random Forest row in your Word table.")
print("=" * 65)

results_df = pd.DataFrame(results).T
results_df.index.name = 'Model'
print(results_df.to_string())
print()


#  TABLE IV

print("=" * 65)
print("  TABLE IV — Confusion Matrices for All Models")
print("             Test Set: 178 No Failure  |  15 Failure")
print()
print("  PLACE IN PAPER:")
print("  Section IV-D  |  After the 1st paragraph")
print("  Paragraph ends: '...comprising 178 non-failure and")
print("  15 failure instances.'")
print("=" * 65)

cm_rows = []
for name, cm in cms.items():
    tn, fp, fn, tp = cm.ravel()
    cm_rows.append({
        'Model'              : name,
        'TN (True Negative)' : tn,
        'FP (False Positive)': fp,
        'FN (False Negative)': fn,
        'TP (True Positive)' : tp
    })
print(pd.DataFrame(cm_rows).to_string(index=False))
print()


#  TABLE V

print("=" * 65)
print("  TABLE V — Feature Importance Scores  (Random Forest)")
print()
print("  PLACE IN PAPER:")
print("  Section IV-E  |  After the 1st paragraph")
print("  Paragraph ends: '...ranked from most to least influential.'")
print("  NOTE: Highlight the Tool wear [min] row — dominant feature.")
print("=" * 65)

fi_df = pd.DataFrame({
    'Feature'          : importances.index,
    'Importance Score' : importances.values.round(4),
    'Rank'             : ['1st','2nd','3rd','4th','5th']
})
print(fi_df.to_string(index=False))
print()


#  FIGURES

#  FIGURE 1 — Class Distribution Before and After SMOTE

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
fig.suptitle(
    'Figure 1. Class Distribution of TWF Label Before and After SMOTE Augmentation',
    fontsize=11, fontweight='bold', y=1.01
)

labels     = ['No Failure\n(TWF = 0)', 'Failure\n(TWF = 1)']
before_vals = [(y_train==0).sum(), (y_train==1).sum()]
after_vals  = [(y_train_sm==0).sum(), (y_train_sm==1).sum()]

# Left — Before SMOTE
bars_b = axes[0].bar(labels, before_vals,
                     color=[BLUE, RED], width=0.45,
                     edgecolor='white', linewidth=1.2)
axes[0].set_title('Before SMOTE\n(Original Training Set)',
                   fontweight='bold')
axes[0].set_ylabel('Number of Observations', fontweight='bold')
axes[0].set_ylim(0, 850)
for bar, val in zip(bars_b, before_vals):
    axes[0].text(bar.get_x()+bar.get_width()/2,
                 bar.get_height()+12, str(val),
                 ha='center', fontsize=11, fontweight='bold')
axes[0].annotate('Imbalance Ratio: 12:1',
                 xy=(0.5,0.05), xycoords='axes fraction',
                 ha='center', fontsize=10,
                 color=RED, fontweight='bold')

# Right — After SMOTE
bars_a = axes[1].bar(labels, after_vals,
                     color=[BLUE, RED], width=0.45,
                     edgecolor='white', linewidth=1.2)
axes[1].set_title('After SMOTE\n(Augmented Training Set)',
                   fontweight='bold')
axes[1].set_ylabel('Number of Observations', fontweight='bold')
axes[1].set_ylim(0, 850)
for bar, val in zip(bars_a, after_vals):
    axes[1].text(bar.get_x()+bar.get_width()/2,
                 bar.get_height()+12, str(val),
                 ha='center', fontsize=11, fontweight='bold')
axes[1].annotate('Balanced Ratio: 1:1',
                 xy=(0.5,0.05), xycoords='axes fraction',
                 ha='center', fontsize=10,
                 color='#375623', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/figure1_class_distribution.png')
plt.close()

print("  FIGURE 1 saved → outputs/figure1_class_distribution.png")
print("  PLACE IN PAPER:")
print("  Section III-D  |  After the only paragraph")
print("  Paragraph ends: '...reliable failure detection models.'")
print()


#  FIGURE 2 — Feature Correlation Heatmap

fig, ax = plt.subplots(figsize=(8, 6))

corr_df         = df_h[FEATURES + ['TWF']].copy()
corr_df.columns = [
    'Air Temp\n[K]',    'Proc Temp\n[K]',
    'Rot Speed\n[rpm]', 'Torque\n[Nm]',
    'Tool Wear\n[min]', 'TWF'
]
corr = corr_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(
    corr, mask=mask,
    annot=True, fmt='.2f', cmap='Blues',
    ax=ax, linewidths=0.6, linecolor='white',
    vmin=-1, vmax=1,
    annot_kws={'size': 11, 'weight': 'bold'},
    cbar_kws={'shrink': 0.8}
)
ax.set_title(
    'Figure 2. Pearson Correlation Heatmap\n'
    'Type H Filtered Dataset  (n = 961)',
    fontweight='bold', pad=12
)
ax.tick_params(axis='x', rotation=0)
ax.tick_params(axis='y', rotation=0)

plt.tight_layout()
plt.savefig('outputs/figure2_correlation_heatmap.png')
plt.close()

print("  FIGURE 2 saved → outputs/figure2_correlation_heatmap.png")
print("  PLACE IN PAPER:")
print("  Section III-B  |  After TABLE I")
print("  Order: paragraph 1 → paragraph 2 → TABLE I → FIGURE 2")
print()

#  FIGURE 3 — Tool Wear Distribution by TWF Label

no_fail_tw = df_h[df_h['TWF']==0]['Tool wear [min]']
fail_tw    = df_h[df_h['TWF']==1]['Tool wear [min]']

fig, ax = plt.subplots(figsize=(9, 5))

ax.hist(no_fail_tw, bins=30, alpha=0.70, color=BLUE,
        edgecolor='white',
        label=f'TWF = 0   No Failure   (n = {len(no_fail_tw)})')
ax.hist(fail_tw, bins=30, alpha=0.80, color=RED,
        edgecolor='white',
        label=f'TWF = 1   Failure      (n = {len(fail_tw)})')
ax.axvline(no_fail_tw.mean(), color=BLUE, linestyle='--',
           linewidth=2,
           label=f'Mean — No Failure  =  {no_fail_tw.mean():.1f} min')
ax.axvline(fail_tw.mean(), color=RED, linestyle='--',
           linewidth=2,
           label=f'Mean — Failure      =  {fail_tw.mean():.1f} min')

ax.set_xlabel('Tool Wear  [min]',  fontweight='bold')
ax.set_ylabel('Frequency',         fontweight='bold')
ax.set_title(
    'Figure 3. Distribution of Tool Wear [min] by TWF Class Label\n'
    'Type H Filtered Dataset  (n = 961)',
    fontweight='bold'
)
ax.legend(framealpha=0.9, fontsize=10)

plt.tight_layout()
plt.savefig('outputs/figure3_tool_wear_distribution.png')
plt.close()

print("  FIGURE 3 saved → outputs/figure3_tool_wear_distribution.png")
print("  PLACE IN PAPER:")
print("  Section IV-A  |  After the 2nd paragraph")
print("  Paragraph ends: '...providing a representative basis")
print("  for failure prediction model training.'")
print()


#  FIGURE 4 — Confusion Matrices (2x2 grid, one per model)

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
fig.suptitle(
    'Figure 4. Confusion Matrices for All Four Classifiers\n'
    'Type H Test Set  (n = 193  |  178 No Failure,  15 Failure)',
    fontsize=11, fontweight='bold'
)

for idx, (name, cm) in enumerate(cms.items()):
    ax = axes.flatten()[idx]

    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', ax=ax,
        xticklabels=['Predicted:\nNo Failure', 'Predicted:\nFailure'],
        yticklabels=['Actual:\nNo Failure',    'Actual:\nFailure'],
        linewidths=0.5, linecolor='gray',
        annot_kws={'size': 16, 'weight': 'bold'}
    )
    ax.set_title(name, fontweight='bold', fontsize=12, pad=8)
    ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=10)
    ax.set_ylabel('Actual Label',    fontweight='bold', fontsize=10)

    cell_tags = [['TN', 'FP'], ['FN', 'TP']]
    for r in range(2):
        for c in range(2):
            ax.text(c+0.5, r+0.82, cell_tags[r][c],
                    ha='center', va='center',
                    fontsize=9, color='gray', style='italic')

plt.tight_layout()
plt.savefig('outputs/figure4_confusion_matrices.png')
plt.close()

print("  FIGURE 4 saved → outputs/figure4_confusion_matrices.png")
print("  PLACE IN PAPER:")
print("  Section IV-D  |  After the 2nd paragraph")
print("  Full order: paragraph 1 → TABLE IV → paragraph 2 → FIGURE 4")
print("  Paragraph 2 ends: '...most balanced and reliable classifier")
print("  for the overall failure detection task.'")
print()


#  FIGURE 5 — ROC Curves (all 4 models overlaid)

fig, ax = plt.subplots(figsize=(8, 6))

for (name, prob), color in zip(probs.items(), COLORS):
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc         = results[name]['AUC-ROC']
    ax.plot(fpr, tpr, color=color, linewidth=2.5,
            label=f'{name}   (AUC = {auc:.4f})')

ax.plot([0,1],[0,1], color='black', linestyle='--',
        linewidth=1.2, label='Random Classifier   (AUC = 0.5000)')

ax.set_xlabel('False Positive Rate  (1 − Specificity)', fontweight='bold')
ax.set_ylabel('True Positive Rate  (Sensitivity)',       fontweight='bold')
ax.set_title(
    'Figure 5. Receiver Operating Characteristic (ROC) Curves\n'
    'All Four Classifiers — Type H Test Set  (n = 193)',
    fontweight='bold'
)
ax.legend(loc='lower right', framealpha=0.9, fontsize=10)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.02])

plt.tight_layout()
plt.savefig('outputs/figure5_roc_curves.png')
plt.close()

print("  FIGURE 5 saved → outputs/figure5_roc_curves.png")
print("  PLACE IN PAPER:")
print("  Section IV-C  |  After FIGURE 7")
print("  Full order: para 1 → TABLE III → para 2 → para 3")
print("              → FIGURE 7 → FIGURE 5")
print()


#  FIGURE 6 — Feature Importance Horizontal Bar Chart

imp_sorted = importances.sort_values(ascending=True)
bar_colors = [RED if v == imp_sorted.max() else BLUE
              for v in imp_sorted.values]

fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.barh(
    imp_sorted.index, imp_sorted.values,
    color=bar_colors, edgecolor='white', height=0.50
)
for bar, val in zip(bars, imp_sorted.values):
    ax.text(val+0.008,
            bar.get_y()+bar.get_height()/2,
            f'{val:.4f}',
            va='center', fontsize=10, fontweight='bold')

ax.set_xlabel('Importance Score', fontweight='bold')
ax.set_xlim(0, 0.95)
ax.set_title(
    'Figure 6. Random Forest Feature Importance Scores\n'
    'Type H Filtered Dataset  (n = 961)',
    fontweight='bold'
)

dominant = mpatches.Patch(color=RED,  label='Dominant Feature  (Tool Wear [min])')
support  = mpatches.Patch(color=BLUE, label='Supporting Features')
ax.legend(handles=[dominant, support], loc='lower right', framealpha=0.9)

plt.tight_layout()
plt.savefig('outputs/figure6_feature_importance.png')
plt.close()

print("  FIGURE 6 saved → outputs/figure6_feature_importance.png")
print("  PLACE IN PAPER:")
print("  Section IV-E  |  After the 2nd paragraph")
print("  Full order: paragraph 1 → TABLE V → paragraph 2 → FIGURE 6")
print("  Paragraph 2 ends: '...greater precision than fixed-interval")
print("  replacement schedules.'")
print()


#  FIGURE 7 — Multi-Metric Model Comparison Bar Chart

metrics     = ['Accuracy','Precision','Recall','F1-Score','AUC-ROC']
model_names = list(results.keys())
x           = np.arange(len(metrics))
bar_width   = 0.18

fig, ax = plt.subplots(figsize=(12, 6))

for i, (name, color) in enumerate(zip(model_names, COLORS)):
    vals   = [results[name][m] for m in metrics]
    offset = (i - 1.5) * bar_width
    rects  = ax.bar(x+offset, vals, bar_width,
                    label=name, color=color,
                    edgecolor='white', alpha=0.92)
    for rect, val in zip(rects, vals):
        ax.text(rect.get_x()+rect.get_width()/2,
                rect.get_height()+0.007,
                f'{val:.2f}',
                ha='center', va='bottom',
                fontsize=7.5, fontweight='bold')

ax.set_ylabel('Score', fontweight='bold')
ax.set_title(
    'Figure 7. Multi-Metric Model Performance Comparison\n'
    'Type H Test Set  (n = 193)',
    fontweight='bold'
)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontweight='bold', fontsize=11)
ax.set_ylim(0, 1.20)
ax.axhline(y=0.5, color='gray', linestyle='--',
           linewidth=0.8, alpha=0.5)
ax.legend(loc='upper left', framealpha=0.9, fontsize=10)

plt.tight_layout()
plt.savefig('outputs/figure7_model_comparison.png')
plt.close()

print("  FIGURE 7 saved → outputs/figure7_model_comparison.png")
print("  PLACE IN PAPER:")
print("  Section IV-C  |  After the 3rd paragraph")
print("  Full order: para 1 → TABLE III → para 2 → para 3")
print("              → FIGURE 7 → FIGURE 5")
print("  Paragraph 3 ends: '...consistent with findings in recent")
print("  comparative literature [10].'")
print()


#  FINAL PLACEMENT GUIDE

print()
print("=" * 65)
print("  COMPLETE PLACEMENT GUIDE")
print("=" * 65)
print()
print("  ── METHODOLOGY (Section III) ──────────────────────────")
print()
print("  TABLE I")
print("    Subsection B — Dataset Acquisition and Type H Filtering")
print("    After the 2nd paragraph")
print("    Paragraph ends: '...Table I summarizes the dataset")
print("    structure before and after the Type H filtering step.'")
print()
print("  FIGURE 2  (Correlation Heatmap)")
print("    Subsection B — Dataset Acquisition and Type H Filtering")
print("    After TABLE I")
print("    Order: para 1 → para 2 → TABLE I → FIGURE 2")
print()
print("  TABLE II")
print("    Subsection C — Data Preprocessing")
print("    After the 1st paragraph")
print("    Paragraph ends: '...confirming that all features fall")
print("    within physically plausible operating ranges.'")
print()
print("  FIGURE 1  (Class Distribution / SMOTE)")
print("    Subsection D — Class Imbalance Mitigation Using SMOTE")
print("    After the only paragraph")
print("    Paragraph ends: '...reliable failure detection models.'")
print()
print("  ── RESULTS AND DISCUSSION (Section IV) ────────────────")
print()
print("  FIGURE 3  (Tool Wear Distribution)")
print("    Subsection A — Filtered Dataset Characterization")
print("    After the 2nd paragraph")
print("    Paragraph ends: '...representative basis for failure")
print("    prediction model training.'")
print()
print("  TABLE III  (Model Performance)")
print("    Subsection C — Model Performance Comparison")
print("    After the 1st paragraph")
print("    Paragraph ends: '...The highlighted row indicates the")
print("    best-performing model.'")
print()
print("  FIGURE 7  (Multi-Metric Bar Chart)")
print("    Subsection C — Model Performance Comparison")
print("    After the 3rd paragraph")
print("    Paragraph ends: '...consistent with findings in recent")
print("    comparative literature [10].'")
print()
print("  FIGURE 5  (ROC Curves)")
print("    Subsection C — Model Performance Comparison")
print("    After FIGURE 7")
print()
print("  TABLE IV  (Confusion Matrices)")
print("    Subsection D — Confusion Matrix Analysis")
print("    After the 1st paragraph")
print("    Paragraph ends: '...comprising 178 non-failure and")
print("    15 failure instances.'")
print()
print("  FIGURE 4  (Confusion Matrices — 4 panels)")
print("    Subsection D — Confusion Matrix Analysis")
print("    After the 2nd paragraph")
print("    Paragraph ends: '...most balanced and reliable classifier")
print("    for the overall failure detection task.'")
print()
print("  TABLE V  (Feature Importance)")
print("    Subsection E — Feature Importance Analysis")
print("    After the 1st paragraph")
print("    Paragraph ends: '...ranked from most to least influential.'")
print()
print("  FIGURE 6  (Feature Importance Bar Chart)")
print("    Subsection E — Feature Importance Analysis")
print("    After the 2nd paragraph")
print("    Paragraph ends: '...greater precision than fixed-interval")
print("    replacement schedules.'")
print()
print("=" * 65)
print("  Done! Open the  outputs/  folder for all 7 figures.")
print("=" * 65)
print()
