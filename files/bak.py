import pandas as pd
import numpy as np
import zlib
from collections import Counter
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import classification_report

def parse_fasta_header(header):
    chrom, coords = header.split(":")
    start, end = map(int, coords.split("-"))
    return chrom, int(start), int(end)

def load_fasta_to_dataframe(fasta_file, label, max_records=None):
    sequences = []
    for i, record in enumerate(SeqIO.parse(fasta_file, "fasta")):
        if max_records is not None and i >= max_records:
            break
        chrom, start, end = parse_fasta_header(record.id)
        sequences.append([str(record.seq), label, chrom, start, end])
    return sequences

recombined = load_fasta_to_dataframe("recombined.fasta", 1)
non_recombined = load_fasta_to_dataframe("non_recombined_sequences_without_per.fasta", 0, max_records=419)

data = recombined + non_recombined
df = pd.DataFrame(data, columns=["sequence", "label", "chrom", "start", "end"])

df_gene_dist_recombined = pd.read_csv("recombined_with_gene_distance.tsv", sep="\t", header=None)
df_gene_dist_recombined.columns = ["chrom", "start", "end", "gene_chrom", "gene_start", "gene_end", "distance"]

df_gene_dist_non = pd.read_csv("non_recombined_with_gene_distance.tsv", sep="\t", header=None)
df_gene_dist_non.columns = ["chrom", "start", "end", "dummy1", "dummy2", "strand", "gene_chrom", "gene_start", "gene_end", "distance"]

gene_dist_recombined_min = df_gene_dist_recombined.groupby(["chrom", "start", "end"])["distance"].min().reset_index()
gene_dist_non_min = df_gene_dist_non.groupby(["chrom", "start", "end"])["distance"].min().reset_index()

df = df.merge(gene_dist_recombined_min, on=["chrom", "start", "end"], how="left")
df = df.merge(gene_dist_non_min, on=["chrom", "start", "end"], how="left", suffixes=('_recombined', '_non'))

df["min_gene_distance"] = df["distance_recombined"].combine_first(df["distance_non"])
df.drop(columns=["distance_recombined", "distance_non"], inplace=True)

df_methyl_recombined = pd.read_csv("recombined_methylation_percent.bed", sep="\t")
df_methyl_recombined.columns = ["chrom", "start", "end", "percent_methylation"]

df_methyl_non = pd.read_csv("non_recombined_methylation_percent.bed", sep="\t")
df_methyl_non.columns = ["chrom", "start", "end", "percent_methylation"]

df_methyl_all = pd.concat([df_methyl_recombined, df_methyl_non], ignore_index=True)
df = df.merge(df_methyl_all, on=["chrom", "start", "end"], how="left")

def kmer_frequency(sequence, k=3):
    return dict(Counter(sequence[i:i+k] for i in range(len(sequence) - k + 1)))

df["kmer_freq"] = df["sequence"].apply(lambda x: kmer_frequency(x, k=3))

all_kmers = sorted(set(k for km in df["kmer_freq"] for k in km))

def kmer_to_dataframe(kmer_dict, all_kmers):
    return pd.Series({kmer: kmer_dict.get(kmer, 0) for kmer in all_kmers})

df_kmer = df["kmer_freq"].apply(lambda x: kmer_to_dataframe(x, all_kmers))
df_kmer["label"] = df["label"]

X_kmer = df_kmer.drop(columns=["label"])
y_kmer = df_kmer["label"]

scaler = MinMaxScaler()
X_kmer_normalized = scaler.fit_transform(X_kmer)

df_kmer_normalized = pd.DataFrame(X_kmer_normalized, columns=X_kmer.columns)
df_kmer_normalized["label"] = y_kmer.values

def calculate_gc_content(sequence):
    gc_count = sum(1 for base in sequence if base in ('G', 'C'))
    return gc_count / len(sequence) if sequence else 0.0

def calculate_gc_skew(sequence):
    g = sequence.count('G')
    c = sequence.count('C')
    return (g - c) / (g + c) if (g + c) > 0 else 0.0

def calculate_at_skew(sequence):
    a = sequence.count('A')
    t = sequence.count('T')
    return (a - t) / (a + t) if (a + t) > 0 else 0.0

df["gc_content"] = df["sequence"].apply(calculate_gc_content)
df["gc_skew"] = df["sequence"].apply(calculate_gc_skew)
df["at_skew"] = df["sequence"].apply(calculate_at_skew)

df_kmer_normalized["gc_content"] = df["gc_content"]
df_kmer_normalized["gc_skew"] = df["gc_skew"]
df_kmer_normalized["at_skew"] = df["at_skew"]

def average_kmer_distance_general(sequence, kmer):
    positions = [i for i in range(len(sequence) - len(kmer) + 1) if sequence[i:i+len(kmer)] == kmer]
    return np.mean(np.diff(positions)) if len(positions) >= 2 else 0

df["ATA_avg_distance"] = df["sequence"].apply(lambda x: average_kmer_distance_general(x, "ATA"))
df["GAG_avg_distance"] = df["sequence"].apply(lambda x: average_kmer_distance_general(x, "GAG"))

df_kmer_normalized["ATA_avg_distance"] = df["ATA_avg_distance"]
df_kmer_normalized["GAG_avg_distance"] = df["GAG_avg_distance"]

from itertools import product
all_2mers = [''.join(p) for p in product("ATGC", repeat=2)]

for kmer in all_2mers:
    df[f"{kmer}_avg_distance"] = df["sequence"].apply(lambda seq: average_kmer_distance_general(seq, kmer))
    df_kmer_normalized[f"{kmer}_avg_distance"] = df[f"{kmer}_avg_distance"]

def lz_compression_ratio(sequence):
    return len(zlib.compress(sequence.encode())) / len(sequence)

df["lz77_compression"] = df["sequence"].apply(lz_compression_ratio)
df_kmer_normalized["lz77_compression"] = df["lz77_compression"]

df_kmer_normalized["min_gene_distance"] = df["min_gene_distance"]
df_kmer_normalized["percent_methylation"] = df["percent_methylation"]

X_train, X_test, y_train, y_test = train_test_split(
    df_kmer_normalized.drop(columns=["label"]), df_kmer_normalized["label"], test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_leaf=5, oob_score=True, class_weight="balanced", random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Dokładność modelu: {accuracy:.4f}")
print(f"OOB Error: {1 - rf.oob_score_:.4f}")
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Macierz pomyłek")
plt.show()

feature_names = X_kmer.columns.tolist() + [
    "gc_content", "gc_skew", "at_skew",
    "ATA_avg_distance", "GAG_avg_distance"
] + [f"{kmer}_avg_distance" for kmer in all_2mers] + [
    "lz77_compression", "min_gene_distance", "percent_methylation"
]

feature_importances = rf.feature_importances_

feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importances})
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df.head(33))
plt.title("Ważność cech (TOP 33)")
plt.tight_layout()
plt.show()

df_kmer_normalized.to_csv("cechy_sekwencji.csv", index=False)

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)
shap_values_selected = shap_values[:, :, 1]

shap.summary_plot(shap_values_selected, X_test, plot_type='bar')
shap.summary_plot(shap_values_selected, X_test)
shap.dependence_plot("at_skew", shap_values_selected, X_test)

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, RocCurveDisplay, PrecisionRecallDisplay

roc_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
print(f"ROC AUC: {roc_auc:.4f}")
RocCurveDisplay.from_estimator(rf, X_test, y_test)
plt.title("ROC Curve")
plt.show()

precision, recall, _ = precision_recall_curve(y_test, rf.predict_proba(X_test)[:, 1])
pr_auc = auc(recall, precision)
print(f"PR AUC: {pr_auc:.4f}")
PrecisionRecallDisplay(precision=precision, recall=recall).plot()
plt.title("Precision-Recall Curve")
plt.show()

from sklearn.model_selection import cross_val_score, StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, df_kmer_normalized.drop(columns=["label"]), df_kmer_normalized["label"], cv=cv, scoring='accuracy')
print(f"Cross-val Accuracy (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

corr = df_kmer_normalized.corr().abs()
top_corr = corr["label"].sort_values(ascending=False).head(35).index
plt.figure(figsize=(12, 10))
sns.heatmap(df_kmer_normalized[top_corr].corr(), annot=True, cmap="coolwarm")
plt.title("Top 35 Korelacje zmiennych")
plt.show()

# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_kmer_normalized)
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_kmer, palette="Set1", alpha=0.7)
# plt.xlabel("PCA 1")
# plt.ylabel("PCA 2")
# plt.title("PCA 2D Visualisation")
# plt.legend(title="Label")
# plt.tight_layout()
# plt.show()

from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(score_func=f_classif, k=30)
selector.fit(X_kmer_normalized, y_kmer)
selected_features = pd.Series(selector.scores_, index=X_kmer.columns).sort_values(ascending=False).head(30)
plt.figure(figsize=(12, 6))
sns.barplot(x=selected_features.values, y=selected_features.index)
plt.title("ANOVA F-score")
plt.xlabel("F-score")
plt.tight_layout()
plt.show()
