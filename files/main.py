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

def load_fasta_to_dataframe(fasta_file, label):
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append([str(record.seq), label])
    return sequences

recombined = load_fasta_to_dataframe("recombined.fasta", 1)  
non_recombined = load_fasta_to_dataframe("non_recombined.fasta", 0)  


print(f"Liczba sekwencji rekombinowanych: {len(recombined)}")
print(f"Liczba sekwencji nierekombinowanych: {len(non_recombined)}")

data = recombined + non_recombined
df = pd.DataFrame(data, columns=["sequence", "label"])
print(df.head())

def kmer_frequency(sequence, k=3):
    kmers = []
    for i in range(len(sequence) - k + 1):
        kmers.append(sequence[i:i+k])
    return dict(Counter(kmers))

df["kmer_freq"] = df["sequence"].apply(lambda x: kmer_frequency(x, k=3))
print(df["kmer_freq"].head())

all_kmers = set()
for kmer_dict in df["kmer_freq"]:
    all_kmers.update(kmer_dict.keys())
all_kmers = sorted(all_kmers)

def kmer_to_dataframe(kmer_dict, all_kmers):
    kmer_features = {}
    for kmer in all_kmers:
        kmer_features[kmer] = kmer_dict.get(kmer, 0)
    return pd.Series(kmer_features)

df_kmer = df["kmer_freq"].apply(lambda x: kmer_to_dataframe(x, all_kmers))
df_kmer["label"] = df["label"]
print(df_kmer.head())


X_kmer = df_kmer.drop(columns=["label"])
y_kmer = df_kmer["label"]

scaler = MinMaxScaler()
X_kmer_normalized = scaler.fit_transform(X_kmer)

df_kmer_normalized = pd.DataFrame(X_kmer_normalized, columns=X_kmer.columns)
df_kmer_normalized["label"] = y_kmer.values
print(df_kmer_normalized.head())

def calculate_gc_content(sequence):
    gc_count = 0
    for base in sequence:
        if base in ('G', 'C'):
            gc_count += 1
    return gc_count / len(sequence) if len(sequence) > 0 else 0.0

df["gc_content"] = df["sequence"].apply(calculate_gc_content)
df_kmer_normalized["gc_content"] = df["gc_content"]


def average_kmer_distance(sequence, kmer="ATA"):
    positions = []
    for i in range(len(sequence) - len(kmer) + 1):
        if sequence[i:i + len(kmer)] == kmer:
            positions.append(i)

    if len(positions) < 2:
        return 0 

    distances = []
    for i in range(1, len(positions)):
        distances.append(positions[i] - positions[i - 1])

    return sum(distances) / len(distances)

df["ATA_avg_distance"] = df["sequence"].apply(average_kmer_distance)
df_kmer_normalized["ATA_avg_distance"] = df["ATA_avg_distance"]

df["GAG_avg_distance"] = df["sequence"].apply(lambda seq: average_kmer_distance(seq, kmer="GAG"))
df_kmer_normalized["GAG_avg_distance"] = df["GAG_avg_distance"]

def lz_compression_ratio(sequence):
    compressed = zlib.compress(sequence.encode()) 
    return len(compressed) / len(sequence) 

df["lz77_compression"] = df["sequence"].apply(lz_compression_ratio)
df_kmer_normalized["lz77_compression"] = df["lz77_compression"]

print(df_kmer_normalized.head())


X_train, X_test, y_train, y_test = train_test_split(
    df_kmer_normalized.drop(columns=["label"]), df_kmer_normalized["label"], test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_leaf=5, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Dokładność modelu: {accuracy}")

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.show()

feature_importances = rf.feature_importances_
feature_names = X_kmer.columns.tolist() + ["gc_content", "ATA_avg_distance", "GAG_avg_distance", "lz77_compression"]

feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importances})
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
plt.title("Ważność cech w modelu Random Forest")
plt.show()

df_kmer_normalized.to_csv("cechy_sekwencji.csv", index=False)
