import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Dossier contenant les fichiers JSON
json_folder = "../Algorithms/FedGA_10_1/fog4"

# Initialisation des listes pour stocker les résultats
rounds = []
train_accuracies = []
test_accuracies = []
train_losses = []
test_losses = []
train_f1s = []
train_ece = []
test_losses = []
test_accuracies = []
test_f1s = []
test_ece = []
worst_10_acc = []

# Parcourir tous les fichiers JSON dans le dossier
for filename in os.listdir(json_folder):
  if ('fog' not in filename):
    if filename.endswith(".json"):
        filepath = os.path.join(json_folder, filename)
        with open(filepath, "r") as file:
            data = json.load(file)
            rounds.append(data["Round"])
            train_accuracies.append(data["Train Accuracy"])
            test_accuracies.append(data["Test Accuracy"])
            train_losses.append(data["Train Loss"])
            test_losses.append(data["Test Loss"])
            train_f1s.append(data["Train F1"])
            test_f1s.append(data["Test F1"])
            train_ece.append(data["Train ECE"])
            test_ece.append(data["Test ECE"])
            worst_10_acc.append(data["Worst 10% Accuracy"])


# Tri des données par rounds pour un affichage cohérent
sorted_indices = sorted(range(len(rounds)), key=lambda i: rounds[i])
rounds = [rounds[i] for i in sorted_indices]
train_accuracies = [train_accuracies[i] for i in sorted_indices]
test_accuracies = [test_accuracies[i] for i in sorted_indices]
train_losses = [train_losses[i] for i in sorted_indices]
test_losses = [test_losses[i] for i in sorted_indices]
train_f1s = [train_f1s[i] for i in sorted_indices]
test_f1s = [test_f1s[i] for i in sorted_indices]
train_ece = [train_ece[i] for i in sorted_indices]
test_ece = [test_ece[i] for i in sorted_indices]
worst_10_acc = [worst_10_acc[i] for i in sorted_indices]


# Création d'une figure avec plusieurs sous-graphiques
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle("Performance Metrics Over Rounds", fontsize=16)

# Tracer les courbes
axes[0, 0].plot(rounds, train_losses, label="Train Loss", color="b")
axes[0, 0].set_title("Train Loss")
axes[0, 0].set_xlabel("Round")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].grid()

axes[0, 1].plot(rounds, train_accuracies, label="Train Accuracy", color="g")
axes[0, 1].set_title("Train Accuracy")
axes[0, 1].set_xlabel("Round")
axes[0, 1].set_ylabel("Accuracy (%)")
axes[0, 1].grid()

axes[0, 2].plot(rounds, train_f1s, label="Train F1", color="r")
axes[0, 2].set_title("Train F1 Score")
axes[0, 2].set_xlabel("Round")
axes[0, 2].set_ylabel("F1 Score")
axes[0, 2].grid()

axes[1, 0].plot(rounds, train_ece, label="Train ECE", color="c")
axes[1, 0].set_title("Train ECE")
axes[1, 0].set_xlabel("Round")
axes[1, 0].set_ylabel("ECE")
axes[1, 0].grid()

axes[1, 1].plot(rounds, test_losses, label="Test Loss", color="b")
axes[1, 1].set_title("Test Loss")
axes[1, 1].set_xlabel("Round")
axes[1, 1].set_ylabel("Loss")
axes[1, 1].grid()

axes[1, 2].plot(rounds, test_accuracies, label="Test Accuracy", color="g")
axes[1, 2].set_title("Test Accuracy")
axes[1, 2].set_xlabel("Round")
axes[1, 2].set_ylabel("Accuracy (%)")
axes[1, 2].grid()

axes[2, 0].plot(rounds, test_f1s, label="Test F1", color="r")
axes[2, 0].set_title("Test F1 Score")
axes[2, 0].set_xlabel("Round")
axes[2, 0].set_ylabel("F1 Score")
axes[2, 0].grid()

axes[2, 1].plot(rounds, test_ece, label="Test ECE", color="c")
axes[2, 1].set_title("Test ECE")
axes[2, 1].set_xlabel("Round")
axes[2, 1].set_ylabel("ECE")
axes[2, 1].grid()

axes[2, 2].plot(rounds, worst_10_acc, label="Worst 10% Accuracy", color="m")
axes[2, 2].set_title("Worst 10% Accuracy")
axes[2, 2].set_xlabel("Round")
axes[2, 2].set_ylabel("Accuracy (%)")
axes[2, 2].grid()

# Ajustement de la disposition
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


