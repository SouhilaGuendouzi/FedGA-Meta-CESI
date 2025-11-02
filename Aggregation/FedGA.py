import torch
from torch import nn
import pygad
import pygad.torchga as tg
from torch.utils.data import DataLoader
import logging
import time
import numpy as np
import os
import json
import matplotlib.pyplot as plt
# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

loss_function = nn.CrossEntropyLoss()


def print_population(ga_instance):
    population = ga_instance.population
    print("Population Diversity:", np.std(population, axis=0))

def calculate_diversity(population):
    """
    Calcule la diversité d'une population en utilisant la variance des poids.
    """
    population_array = np.array(population)
    return np.mean(np.var(population_array, axis=0))

# Fonction de mutation désactivée
def no_mutation(offspring, ga_instance):
    # Retourner les descendants sans modification
    return offspring
def fitness(ga_instance,solution, sol_idx):
    """Calcul de la fitness d'une solution en évaluant la perte du modèle."""
    try:
        # Charger les poids dans le modèle
        if classification_status:
             model_weights_dict = tg.model_weights_as_dict(model=model.classification, weights_vector=solution)
            
             model.classification.load_state_dict(model_weights_dict)
        else:
            model_weights_dict = tg.model_weights_as_dict(model=model, weights_vector=solution)
            model.load_state_dict(model_weights_dict)
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in dataset:
                data, target = data.to(device), target.to(device)
                
                if dataset_name == 'usps':
                    target = target.long()
                    data = data.view(len(data), 1, 28, 28)
 
                prediction = model(data)
                try:
                   total_loss += loss_function(prediction, target).item()
                except:
                  target=target.squeeze(1)
                  total_loss += loss_function(prediction, target).item()
                _, predicted = prediction.max(1)
                correct += (predicted == target).sum().item()
                total += target.size(0)

        avg_loss = total_loss / len(dataset)
        # Vérification finale avant retour de la fitness
        if np.isnan(avg_loss) or np.isinf(avg_loss):
            return 0.0
        
        #print (f"Solution {sol_idx}: Loss = {avg_loss:.4f}")
        accuracy = correct / total
        #logging.debug(f"Solution {sol_idx}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")


        # La fitness est inversement proportionnelle à la perte
        fitness_score = 1.0 / (avg_loss + 1e-8)
        #fitness_score = accuracy
        return fitness_score

    except Exception as e:
        logging.error(f"Erreur lors de l'évaluation de la fitness : {e}")
        return 0.0


def callback_generation(ga_instance):
   
    diversity = calculate_diversity(ga_instance.population)
    diversity_tracking.append(diversity)

    """Callback pour afficher les informations après chaque génération."""
    logging.info(f"Génération {ga_instance.generations_completed}: Fitness max = {ga_instance.best_solution()[1]:.4f}: Diversity = {np.std(ga_instance.population, axis=0)}")
    save_path = f"../Diversity/FedGA_20/{FL_Round}_générations/diversity.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump({"generation": ga_instance.generations_completed, "diversity": diversity_tracking}, f, indent=4)
    
    time.sleep(1)

def FedGA(weights, model_, dataset_, dataset_name_, Round,classification=False, config=None):
    """
    Agrégation basée sur les algorithmes génétiques pour l'apprentissage fédéré.

    Args:
        weights (list): Liste des poids locaux sous forme de vecteurs.
        model_ (torch.nn.Module): Modèle PyTorch pour l'agrégation.
        dataset_ (DataLoader): Ensemble de données pour l'évaluation.
        dataset_name_ (str): Nom du dataset.
        config (dict, optional): Configuration pour l'algorithme génétique.

    Returns:
        dict: Meilleurs poids sous forme de `state_dict`.
    """
    global model, dataset, device, dataset_name, classification_status, FL_Round, diversity_tracking

    # Initialisation des variables globales
    classification_status = True #classification
    FL_Round = Round
    model = model_
    dataset = dataset_
    dataset_name = dataset_name_
    diversity_tracking = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Classification status: {classification_status}")
    path = f"../Diversity/FedGA_Meta_1/{FL_Round}_générations"
    if not os.path.exists(path):
        os.makedirs(path)

    # Conversion des poids initiaux
    #initial_population = [w.cpu().numpy() if isinstance(w, torch.Tensor) else w for w in weights]
     # Flatten weights into 1D vectors
    initial_population = [torch.cat([w.flatten() for w in state.values()]).cpu().numpy() for state in weights]

    #print("Initial population: ", len(initial_population), len(initial_population[0]))


    # Configuration par défaut
    """
    default_config = {
        "num_generations": 10,
        "num_parents_mating": len(initial_population),
        "sol_per_pop":len(initial_population),
        "parent_selection_type": "rank",  #  "tournament", #'sus',# "rank",
        "crossover_type":"single_point", #"scattered",
        "mutation_type": no_mutation, #"random",
        #"mutation_percent_genes": 10,

    }
    """
    default_config = {
        "num_generations": 10,
        "num_parents_mating": len(initial_population),
        "sol_per_pop":len(initial_population),
        "parent_selection_type": "tournament",  
        "crossover_type":"scattered", 
        "mutation_type" :"adaptive",  # new enhancement
        "mutation_percent_genes":(0.3, 0.1),  # 30% au début, 10% à la fin  # new enhancement
    }
    

    if config:
        default_config.update(config)

    # Initialisation de l'algorithme génétique
    ga_instance = pygad.GA(
        num_generations=default_config["num_generations"],
        num_parents_mating=default_config["num_parents_mating"],
        sol_per_pop=default_config["sol_per_pop"],
        initial_population=initial_population,
        fitness_func=fitness,
        parent_selection_type=default_config["parent_selection_type"],
        crossover_type=default_config["crossover_type"],
        mutation_type=default_config["mutation_type"],# "random"
        mutation_percent_genes=default_config["mutation_percent_genes"],
        on_generation=callback_generation,
    )

    # Exécution de l'algorithme génétique
    logging.info("Démarrage de l'évolution génétique.")
    ga_instance.run()

    # Extraction des meilleurs poids
    try: 
      solution, solution_fitness, solution_idx = ga_instance.best_solution()
      if (classification_status):
        best_solution_weights = tg.model_weights_as_dict(model=model.classification, weights_vector=solution)
        model.classification.load_state_dict(best_solution_weights)
      else:
        best_solution_weights = tg.model_weights_as_dict(model=model.classification, weights_vector=solution)
        model.load_state_dict(best_solution_weights)
    except:
      logging.info(f"Meilleure solution trouvée avec une fitness de {solution_fitness:.4f}.")
      return weights[0]

    return best_solution_weights
