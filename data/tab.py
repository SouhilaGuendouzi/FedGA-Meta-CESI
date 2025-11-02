import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# Dossier contenant les résultats pour un seul algorithme
base_dir = "../Algorithms/Fed_per_10_1/"
#check  if the directory exists
if not os.path.exists(base_dir):
          print("Directory not found")
          exit()
# Rounds à extraire
target_rounds = {0, 20, 50}
# Initialisation des résultats
results = []

# Parcours des fog_1 à fog_5
for fog in range(1, 6):
    fog_path = os.path.join(base_dir, f"fog{fog}")
    if not os.path.exists(fog_path):
        continue
    
    # Lecture des fichiers JSON et extraction des métriques
    for filename in os.listdir(fog_path):
        if filename.endswith(".json"):
            file_path = os.path.join(fog_path, filename)
            with open(file_path, "r") as file:
                data = json.load(file)
            
            if data["Round"] in target_rounds:
                results.append({
                    "Fog": fog,
                    "Round": data["Round"],
                    "Train Loss": data["Train Loss"],
                    "Train Accuracy": data["Train Accuracy"],
                    "Train F1": data["Train F1"],
                    "Train ECE": data["Train ECE"],
                    "Test Loss": data["Test Loss"],
                    "Test Accuracy": data["Test Accuracy"],
                    "Test F1": data["Test F1"],
                    "Test ECE": data["Test ECE"],
                    "Worst 10% Accuracy": data["Worst 10% Accuracy"]
                })

# Création d'un DataFrame
if results:
    df_results = pd.DataFrame(results)
    print(df_results)
    
    # Sauvegarde des résultats en fichier JSON
    output_json_path = f"{base_dir}/tableau_0_20_50.json"
    df_results.to_json(output_json_path, orient="records", indent=4)
    print(f"Résultats sauvegardés dans {output_json_path}")
    
    # Génération des graphes
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle("Performance Metrics Over Rounds - Fed_MAML_10_1", fontsize=16)
    
    for (metric, ax, color) in zip([
        "Train Loss", "Train Accuracy", "Train F1", "Train ECE", "Test Loss", 
        "Test Accuracy", "Test F1", "Test ECE", "Worst 10% Accuracy"],
        axes.flatten(), ["b", "g", "r", "c", "b", "g", "r", "c", "m"]):
        
        for fog in df_results["Fog"].unique():
            df_fog = df_results[df_results["Fog"] == fog]
            ax.plot(df_fog["Round"], df_fog[metric], label=f"Fog {fog}", color=color)
            
        ax.set_title(metric)
        ax.set_xlabel("Round")
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid()
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
else:
    print("Aucune donnée trouvée.")