import os
import json
import pandas as pd

# === CONFIGURATION ===
base_dir = "results_algos"
algorithms = [
    "Fed_avg_10_1",
    "Fed_per_10_1",
    "Fed_prox_10_1",
    "FedGA_10_1",
    "Fed_MAML_10_1",
    "Fed_GA_Meta_10_1"
]
metrics = ["Test Loss", "Test Accuracy", "Test F1", "Test ECE", "Worst 10% Accuracy"]
num_fogs = 5
target_round = 51

# === CHARGEMENT DES DONNÉES ===
data = []

for fog_id in range(1, num_fogs + 1):
    fog_name = f"fog{fog_id}"

    for algo in algorithms:
        file_path = os.path.join(base_dir, algo, fog_name, f"round_{target_round}.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                result = json.load(f)
                row = {
                    "Fog": fog_name,
                    "Algorithm": algo
                }
                for metric in metrics:
                    row[metric] = result.get(metric, "—")
                data.append(row)
        else:
            row = {"Fog": fog_name, "Algorithm": algo}
            for metric in metrics:
                row[metric] = "—"
            data.append(row)

# === CONVERSION EN DATAFRAME ===
df = pd.DataFrame(data)

# === MISE EN GRAS DES MEILLEURES VALEURS ===
def format_value(val):
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)

def bold_best_per_metric(df, metric, higher_is_better=True):
    result = df.copy()
    for fog in df["Fog"].unique():
        sub = df[df["Fog"] == fog]
        try:
            valid_vals = sub[metric].apply(pd.to_numeric, errors="coerce")
            if higher_is_better:
                best_val = valid_vals.max()
            else:
                best_val = valid_vals.min()
            mask = (df["Fog"] == fog) & (valid_vals == best_val)
            result.loc[mask, metric] = result.loc[mask, metric].apply(lambda x: f"\\textbf{{{format_value(x)}}}")
        except:
            pass
    return result

# Appliquer pour chaque métrique
df_formatted = df.copy()
for m in metrics:
    higher_is_better = not ("Loss" in m or "ECE" in m)
    df_formatted = bold_best_per_metric(df_formatted, m, higher_is_better=higher_is_better)

# === GÉNÉRATION DU TABLEAU LATEX ===
latex = "\\begin{table}[H]\n\\centering\n"
latex += "\\caption{Résultats finaux par algorithme et par fog (meilleures valeurs en gras)}\n"
latex += "\\begin{tabular}{ll" + "c" * len(metrics) + "}\n"
latex += "\\toprule\n"
latex += "Fog & Algorithm & " + " & ".join(metrics) + " \\\\\n"
latex += "\\midrule\n"

for idx, row in df_formatted.iterrows():
    values = [row["Fog"], row["Algorithm"].replace("_", "\\_")] + [format_value(row[m]) for m in metrics]
    latex += " & ".join(values) + " \\\\\n"

latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

# === SAUVEGARDE ===
# Créer le dossier results_algos s'il n'existe pas
if not os.path.exists("results_algos"):
    print("Le dossier results_algos n'existe pas. Création du dossier...")
    os.makedirs("results_algos", exist_ok=True)
else :
    print("Le dossier results_algos existe déjà.")

with open("results_algos/tableau_global_round50.tex", "w", encoding="utf-8") as f:
    f.write(latex)

print("✅ Tableau global LaTeX avec mise en gras généré : tableau_global_round50.tex")
