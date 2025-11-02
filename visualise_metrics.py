"""import os
import json
import numpy as np
import matplotlib.pyplot as plt

# === CONFIGURATION ===
base_dir = "Algorithms"
configs_dict = {
    "1": "1", #(1,50)
    "5": "5", #(5,20)
    "20": "20" #(20,10)
}
round_files = {
    "1": "round_51.json",
    "5": "round_21.json",
    "20": "round_11.json"
}
config_labels = list(configs_dict.keys())
metrics = ["Test F1", "Test ECE", "Worst 10% Accuracy"]

# Liste d'algorithmes (sans Fed_avg, remplacement de FedMeta par Fed_MAML)
base_algos = [
    "Fed_per",
    "Fed_prox",
    "Fed_MAML",
    "FedGA",
    "Fed_GA_Meta"
]

# Noms latex pour les algorithmes
latex_algo_names = {
    "Fed_per": "FedPer",
    "Fed_prox": "FedProx",
    "Fed_MAML": "FedMeta",
    "FedGA": "FedGA",
    "Fed_GA_Meta": "FedGA_Meta"
}

# Couleurs féministes douces
soft_feminist_colors = [
    "#EC407A",  # Rose framboise vif
    "#42A5F5",  # Bleu doux mais visible
    "#FFD54F",  # Jaune doré
    "#66BB6A",  # Vert doux mais contrasté
    "#AB47BC"   # Lavande soutenu
]

# === GÉNÉRATION POUR CHAQUE FOG ===
for fog_id in range(1, 6):
    # Extraction des données
    all_metrics_data = {metric: {label: [] for label in config_labels} for metric in metrics}
    for label, epoch in configs_dict.items():
        round_file = round_files[epoch]
        for algo in base_algos:
            algo_full = f"{algo}_10_{epoch}"
            file_path = os.path.join(base_dir, algo_full, f"fog{fog_id}", round_file)
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    content = json.load(f)
                    for metric in metrics:
                        value = content.get(metric, np.nan)
                        all_metrics_data[metric][label].append(value)
            else:
                for metric in metrics:
                    all_metrics_data[metric][label].append(np.nan)

    # Création de la figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    #fig.suptitle(f"Fog {fog_id} - Comparaison des configurations pour chaque métrique", fontsize=16)

    x = np.arange(len(config_labels))
    bar_width = 0.12

    for ax, metric in zip(axes, metrics):
        for i, algo in enumerate(base_algos):
            algo_values = [all_metrics_data[metric][label][i] for label in config_labels]
            ax.bar(x + i * bar_width, algo_values, width=bar_width,
                   label=latex_algo_names[algo],
                   color=soft_feminist_colors[i % len(soft_feminist_colors)])
        ax.set_title(metric)
        ax.set_xticks(x + bar_width * (len(base_algos) - 1) / 2)
        ax.set_xticklabels(config_labels)
        if metric == "Test F1":
            ax.set_ylabel('F1 Score')
        else:
         ax.set_ylabel(metric)
        ax.set_xlabel("Local Epochs")
        ax.grid(True)

    # Légende en bas, une seule ligne
    fig.legend(
        [latex_algo_names[a] for a in base_algos],
        loc='lower center',
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(base_algos),
        title="Frameworks"
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    plt.savefig(f"fog{fog_id}_feminist_metrics_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

# Re-import after kernel reset
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# === CONFIGURATION ===
base_dir = "Algorithms"
configs_dict = {
    "(1,50)": "1",
    "(5,20)": "5",
    "(20,10)": "20"
}
round_files = {
    "1": "round_51.json",
    "5": "round_21.json",
    "20": "round_11.json"
}
config_labels = list(configs_dict.keys())
metrics = ["Test F1", "Test ECE", "Worst 10% Accuracy"]

# Liste d'algorithmes (sans Fed_avg, remplacement de FedMeta par Fed_MAML)
base_algos = [
    "Fed_per",
    "Fed_prox",
    "Fed_MAML",
    "FedGA",
    "Fed_GA_Meta"
]

# Algorithmes noms en gras
latex_algo_names = {
    "Fed_per": "FedPer",
    "Fed_prox": "FedProx",
    "Fed_MAML": "FedMeta",
    "FedGA": "FedGA",
    "Fed_GA_Meta": "FedGA_Meta"
}

# Couleurs expressives et lisibles
final_colors = [
    "#EC407A",  # Rose framboise vif
    "#42A5F5",  # Bleu doux
    "#FFD54F",  # Jaune doré
    "#66BB6A",  # Vert doux
    "#AB47BC"   # Lavande soutenu
]

# === Extraction des données une seule fois pour chaque fog/metric
fog_data_per_metric = {
    metric: {f"fog{i}": {label: [] for label in config_labels} for i in range(1, 6)}
    for metric in metrics
}

for fog_id in range(1, 6):
    for label, epoch in configs_dict.items():
        round_file = round_files[epoch]
        for algo in base_algos:
            algo_full = f"{algo}_10_{epoch}"
            file_path = os.path.join(base_dir, algo_full, f"fog{fog_id}", round_file)
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    content = json.load(f)
                    for metric in metrics:
                        value = content.get(metric, np.nan)
                        fog_data_per_metric[metric][f"fog{fog_id}"][label].append(value)
            else:
                for metric in metrics:
                    fog_data_per_metric[metric][f"fog{fog_id}"][label].append(np.nan)

from matplotlib.gridspec import GridSpec

# === Créer une figure par métrique, affichant tous les fogs : 3 en haut, 2 en bas ===
for metric in metrics:
    fig = plt.figure(figsize=(22, 10))
    #fig.suptitle(f"{metric} - Framework comparison per local epoch across fogs", fontsize=16)

    gs = GridSpec(2, 3, figure=fig)
    fog_axes = []

    # Création des 5 sous-graphes
    fog_axes.append(fig.add_subplot(gs[0, 0]))  # fog1
    fog_axes.append(fig.add_subplot(gs[0, 1]))  # fog2
    fog_axes.append(fig.add_subplot(gs[0, 2]))  # fog3
    fog_axes.append(fig.add_subplot(gs[1, 0]))  # fog4
    fog_axes.append(fig.add_subplot(gs[1, 1]))  # fog5

    x = np.arange(len(config_labels))
    bar_width = 0.12

    for ax, (i, fog_name) in zip(fog_axes, enumerate(fog_data_per_metric[metric].keys())):
        for j, algo in enumerate(base_algos):
            values = [fog_data_per_metric[metric][fog_name][label][j] for label in config_labels]
            ax.bar(x + j * bar_width, values, width=bar_width,
                   label=latex_algo_names[algo],
                   color=final_colors[j % len(final_colors)])
        print(f"Fog {fog_name} - {metric}: {values}")
        if fog_name == "fog1":
            ax.set_title("MNIST")
        elif fog_name == "fog2":
            ax.set_title("USPS")
        elif fog_name == "fog3":
                    ax.set_title("SVHN")
        elif fog_name == "fog4":
                    ax.set_title("EMNIST")
        elif fog_name == "fog5":
                    ax.set_title("MNISTM")
        else : ax.set_title(fog_name.capitalize())
        ax.set_xticks(x + bar_width * (len(base_algos) - 1) / 2)
        ax.set_xticklabels([label.split(",")[0].replace("(", "") for label in config_labels])
        ax.set_xlabel("Local Epochs")
        if i == 0 or i == 3:
            ax.set_ylabel(metric)
        ax.grid(True)

    # Supprimer la 6ème cellule vide (gs[1,2]) si elle existe
    fig.delaxes(fig.add_subplot(gs[1, 2]))

    # Légende centrée en bas
    legend = fig.legend(
        [latex_algo_names[a] for a in base_algos],
        loc='lower center',
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(base_algos),
        title="Frameworks"
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")

    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    fig.savefig(f"{metric.replace(' ', '_')}_fogs_comparison_grid.png", dpi=300, bbox_inches='tight')
    plt.close()
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# === CONFIGURATION ===
base_dir = "results_algos"
configs_dict = {
    "(1,50)": "1",
    "(5,20)": "5",
    "(20,10)": "20"
}
round_files = {
    "1": "round_51.json",
    "5": "round_21.json",
    "20": "round_11.json"
}
config_labels = list(configs_dict.keys())
metrics = ["Test F1", "Test ECE", "Worst 10% Accuracy"]

base_algos = [
    "Fed_per",
    "Fed_prox",
    "Fed_MAML",
    "FedGA",
    "Fed_GA_Meta"
]

latex_algo_names = {
    "Fed_per": "FedPer",
    "Fed_prox": "FedProx",
    "Fed_MAML": "FedMeta",
    "FedGA": "FedGA",
    "Fed_GA_Meta": "FedGA-Meta"
}

final_colors = [
    "#EC407A",  # Rose framboise vif
    "#42A5F5",  # Bleu doux
    "#FFD54F",  # Jaune doré
    "#66BB6A",  # Vert doux
    "#AB47BC"   # Lavande soutenu
]

fog_titles = {
    "fog1": "MNIST",
    "fog2": "USPS",
    "fog3": "SVHN",
    "fog4": "EMNIST",
    "fog5": "MNISTM"
}

# === Extraction des données ===
fog_data_per_metric = {
    metric: {f"fog{i}": {label: [] for label in config_labels} for i in range(1, 6)}
    for metric in metrics
}

for fog_id in range(1, 6):
    for label, epoch in configs_dict.items():
        round_file = round_files[epoch]
        for algo in base_algos:
            algo_full = f"{algo}_10_{epoch}"
            file_path = os.path.join(base_dir, algo_full, f"fog{fog_id}", round_file)
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    content = json.load(f)
                    for metric in metrics:
                        value = content.get(metric, np.nan)
                        fog_data_per_metric[metric][f"fog{fog_id}"][label].append(value)
            else:
                for metric in metrics:
                    fog_data_per_metric[metric][f"fog{fog_id}"][label].append(np.nan)

# === Création des 15 figures ===
output_dir = "results_algos"
os.makedirs(output_dir, exist_ok=True)

x = np.arange(len(config_labels))
bar_width = 0.12

for metric in metrics:
    for fog_id in range(1, 6):
        fog_name = f"fog{fog_id}"
        fig, ax = plt.subplots(figsize=(8, 6))

        for j, algo in enumerate(base_algos):
            values = [fog_data_per_metric[metric][fog_name][label][j] for label in config_labels]
            ax.bar(x + j * bar_width, values, width=bar_width,
                   label=latex_algo_names[algo],
                   color=final_colors[j % len(final_colors)])

        ax.set_title(f"{fog_titles.get(fog_name, fog_name)}")
        ax.set_xticks(x + bar_width * (len(base_algos) - 1) / 2)
        ax.set_xticklabels([label.split(",")[0].replace("(", "") for label in config_labels])
        ax.set_xlabel("Local Epochs")
        if metric == "Test F1":
            ax.set_ylabel('F1-Score')
        else: ax.set_ylabel(metric)
        ax.grid(True)
        ax.legend(title="Frameworks", loc='upper center', bbox_to_anchor=(0.5, -0.12),
                  ncol=len(base_algos), fontsize=9, title_fontsize=10)
        fig.tight_layout()

        # Sauvegarde
        safe_metric = metric.replace(" ", "_").replace("%", "percent")
        filename = f"{fog_name}_{safe_metric}.png"
        fig.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        #plt.show()
        plt.close(fig)
