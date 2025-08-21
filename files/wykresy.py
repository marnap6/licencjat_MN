import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
from scipy.interpolate import make_interp_spline
from sklearn.preprocessing import minmax_scale

# Konfiguracja
THRESHOLDS = [0.70, 0.90]  # Testowane progi
MAX_GAP = 10000  # Maksymalna przerwa dla grupowania CO
BIN_SIZE = 500_000
CENTROMERES = {
    "A01": (13429467, 19461730), "A02": (18014870, 18395614),
    "A03": (31291529, 37706983), "A04": (6225738, 7697430),
    "A05": (12268671, 29192680), "A06": (12724469, 32881125),
    "A07": (3355288, 8176400), "A08": (4482485, 12820813),
    "A09": (19540071, 32490614), "A10": (3407210, 9585822),
    "C01": (28093399, 29398725), "C02": (43273523, 43841688),
    "C03": (47337016, 49832447), "C04": (22413162, 28374342),
    "C05": (21487494, 27923789), "C06": (13283684, 15872456),
    "C07": (11413954, 15549438), "C08": (7105951, 9813269),
    "C09": (30854070, 33634728)
}


# Funkcje pomocnicze
def create_directories(base_path, combinations):
    """Tworzy strukturę katalogów dla wyników"""
    for combo in combinations:
        path = os.path.join(base_path, combo)
        os.makedirs(os.path.join(path, "prediction"), exist_ok=True)
        os.makedirs(os.path.join(path, "probability"), exist_ok=True)
        os.makedirs(os.path.join(path, "prediction_vs_real_line"), exist_ok=True)
        os.makedirs(os.path.join(path, "bar_plots"), exist_ok=True)


def load_crossovers(chrom_lengths):
    """Wczytuje i przetwarza rzeczywiste dane CO"""
    crossovers = pd.read_csv("crossovers_1.bed", sep="\t",
                             header=None, names=["chrom", "start", "end", "id"])

    # Przygotowanie zbinowanych danych CO
    crossover_bins = {}
    for chrom, length in chrom_lengths.items():
        bins = list(range(0, length + BIN_SIZE, BIN_SIZE))
        bin_intervals = pd.IntervalIndex.from_breaks(bins, closed="left")
        bin_counts = pd.Series(0, index=bin_intervals)

        chrom_co = crossovers[crossovers["chrom"] == chrom]
        for _, row in chrom_co.iterrows():
            bin_idx = bin_intervals.get_indexer([row["start"]])[0]
            if bin_idx != -1:
                bin_counts.iloc[bin_idx] += 1

        crossover_bins[chrom] = bin_counts

    return crossover_bins, crossovers


def group_predictions(df, max_gap=10000):
    """Grupuje sąsiednie predykcje CO w pojedyncze zdarzenia"""
    pred_df = df[df["prediction_custom"] == 1].copy()
    if pred_df.empty:
        return 0

    pred_df = pred_df.sort_values("start").reset_index(drop=True)
    group_id = 0
    group_ids = [group_id]

    for i in range(1, len(pred_df)):
        if pred_df.loc[i, "start"] - pred_df.loc[i - 1, "end"] <= max_gap:
            group_ids.append(group_id)
        else:
            group_id += 1
            group_ids.append(group_id)

    pred_df["group"] = group_ids
    return pred_df["group"].nunique()


def generate_chrom_plots(df, chrom, crossover_bins, chrom_length, output_path, apply_scaling):
    """Generuje wykresy dla pojedynczego chromosomu"""
    # Wykres 1: Liczba predykcji
    pred_df = df[df["prediction_custom"] == 1]
    if not pred_df.empty:
        plt.figure(figsize=(12, 4))
        bins = range(0, chrom_length + BIN_SIZE, BIN_SIZE)
        pred_counts, _ = np.histogram(pred_df["mid"], bins=bins)
        bin_centers = [bins[i] + BIN_SIZE / 2 for i in range(len(pred_counts))]

        plt.bar(bin_centers, pred_counts, width=BIN_SIZE * 0.9,
                color="steelblue", edgecolor="black")

        if chrom in CENTROMERES:
            cent_start, cent_end = CENTROMERES[chrom]
            plt.axvspan(cent_start, cent_end, color="gray", alpha=0.3, label="Centromer")

        plt.title(f"Predykcje crossing-overów: {chrom}")
        plt.xlabel("Pozycja genomowa (bp)")
        plt.ylabel("Liczba predykcji")
        plt.xlim(0, chrom_length)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "prediction", f"{chrom}.png"))
        plt.close()

    # Wykres 2: Suma prawdopodobieństw
    plt.figure(figsize=(12, 4))
    prob_sum = df.groupby(pd.cut(df["mid"], bins=bins))["recomb_prob"].sum()
    bin_centers = [interval.mid for interval in prob_sum.index]

    plt.plot(bin_centers, prob_sum.values, linestyle="-", color="darkgreen")

    if chrom in CENTROMERES:
        cent_start, cent_end = CENTROMERES[chrom]
        plt.axvspan(cent_start, cent_end, color="gray", alpha=0.3, label="Centromer")

    plt.title(f"Suma prawdopodobieństw rekombinacji: {chrom}")
    plt.xlabel("Pozycja genomowa (bp)")
    plt.ylabel("Suma prawdopodobieństw")
    plt.xlim(0, chrom_length)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "probability", f"{chrom}.png"))
    plt.close()

    # Wykres 3: Porównanie z rzeczywistymi CO (z interpolacją)
    if chrom in crossover_bins:
        co_counts = crossover_bins[chrom]
        pred_counts = df.groupby(pd.cut(df["mid"], bins=bins))["prediction_custom"].sum()

        bin_centers = np.array([interval.mid for interval in co_counts.index])
        real_values = np.array(co_counts.values, dtype=float)
        pred_values = np.array(pred_counts.values, dtype=float)

        # Skalowanie Min-Max
        if apply_scaling:
            try:
                real_values = minmax_scale(real_values)
                pred_values = minmax_scale(pred_values)
            except ValueError:
                pass  # Pomijaj jeśli nie można przeskalować

        # Interpolacja
        xnew = np.linspace(bin_centers.min(), bin_centers.max(), 500)
        real_spline = make_interp_spline(bin_centers, real_values, k=3)
        pred_spline = make_interp_spline(bin_centers, pred_values, k=3)

        plt.figure(figsize=(12, 4))
        plt.plot(xnew, real_spline(xnew), label="Rzeczywiste CO", color="orange")
        plt.plot(xnew, pred_spline(xnew), label="Predykcje CO", color="steelblue")

        if chrom in CENTROMERES:
            cent_start, cent_end = CENTROMERES[chrom]
            plt.axvspan(cent_start, cent_end, color="gray", alpha=0.3, label="Centromer")

        plt.title(f"Porównanie CO: {chrom}")
        plt.xlabel("Pozycja genomowa (bp)")
        plt.ylabel("Liczba CO (znormalizowana)" if apply_scaling else "Liczba CO")
        plt.legend()
        plt.xlim(0, chrom_length)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "prediction_vs_real_line", f"{chrom}.png"))
        plt.close()


def generate_bar_plots(real_co_norm, pred_co_norm, chrom_lengths, output_path, title_suffix):
    """Generuje wykresy słupkowe porównawcze"""
    common_chroms = sorted(set(real_co_norm.index) & set(pred_co_norm.index))
    real_vals = real_co_norm.loc[common_chroms]
    pred_vals = pred_co_norm.loc[common_chroms]

    # Wykres porównawczy
    plt.figure(figsize=(14, 6))
    bar_width = 0.4
    x = range(len(common_chroms))

    plt.bar([i - bar_width / 2 for i in x], real_vals, width=bar_width,
            label="Rzeczywiste CO/Mb", color="orange")
    plt.bar([i + bar_width / 2 for i in x], pred_vals, width=bar_width,
            label="Predykcje CO/Mb", color="steelblue")

    plt.xticks(x, common_chroms, rotation=45)
    plt.xlabel("Chromosom")
    plt.ylabel("CO na 1 Mb")
    plt.title(f"Porównanie CO na Mb: {title_suffix}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "bar_plots", f"porownanie_co_{title_suffix}.png"))
    plt.close()


def main():
    # Wczytaj pliki i ustal długości chromosomów
    files = glob.glob("windows_*_predictions.tsv")
    chrom_lengths = {}
    all_data = {}

    for file in files:
        df = pd.read_csv(file, sep="\t")
        chrom = df["chrom"].iloc[0]
        chrom_lengths[chrom] = df["end"].max()
        all_data[chrom] = df

    # Wczytaj rzeczywiste CO
    crossover_bins, crossovers = load_crossovers(chrom_lengths)

    # Kombinacje parametrów do przetestowania
    combinations = [
        "threshold_0.70/no_grouping/no_scaling",
        "threshold_0.70/with_grouping/no_scaling",
        "threshold_0.70/no_grouping/with_scaling",
        "threshold_0.70/with_grouping/with_scaling",
        "threshold_0.90/no_grouping/no_scaling",
        "threshold_0.90/with_grouping/no_scaling",
        "threshold_0.90/no_grouping/with_scaling",
        "threshold_0.90/with_grouping/with_scaling",
    ]

    # Przygotuj struktury katalogów
    create_directories("plots/nowe", combinations)

    # Przetwarzanie dla każdej kombinacji
    for combo in combinations:
        parts = combo.split("/")
        threshold = float(parts[0].split("_")[1])
        grouping = "with_grouping" in parts[1]
        scaling = "with_scaling" in parts[2]

        output_path = os.path.join("plots/nowe", combo)
        print(f"\nProcessing: {combo}")

        # Przetwarzaj każdy chromosom
        predicted_counts = {}
        for chrom, df in all_data.items():
            df = df.copy()
            df["prediction_custom"] = (df["recomb_prob"] >= threshold).astype(int)
            df["mid"] = (df["start"] + df["end"]) // 2

            # Generuj wykresy chromosomowe
            generate_chrom_plots(
                df, chrom, crossover_bins, chrom_lengths[chrom],
                output_path, scaling
            )

            # Zliczanie CO (z grupowaniem lub bez)
            if grouping:
                predicted_counts[chrom] = group_predictions(df, MAX_GAP)
            else:
                predicted_counts[chrom] = df["prediction_custom"].sum()

        # Przygotuj dane do wykresów słupkowych
        real_co_counts = crossovers["chrom"].value_counts()
        real_co_norm = (real_co_counts / pd.Series(chrom_lengths)) * 1_000_000

        pred_co_counts = pd.Series(predicted_counts)
        pred_co_norm = (pred_co_counts / pd.Series(chrom_lengths)) * 1_000_000

        # Wygeneruj wykresy słupkowe
        generate_bar_plots(
            real_co_norm,
            pred_co_norm,
            chrom_lengths,
            output_path,
            f"thr{threshold}_grp{grouping}_scl{scaling}"
        )

    print("\nWszystkie kombinacje zostały przetworzone!")


if __name__ == "__main__":
    main()