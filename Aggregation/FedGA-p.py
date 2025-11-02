
import torch 
from torch import  nn
import pygad
import pygad.torchga as tg
from models.clayers import Feature_extractor_Layers

#https://neptune.ai/blog/train-pytorch-models-using-genetic-algorithm-with-pygad ==> Ahmed Gad




loss_function = nn.CrossEntropyLoss()


def fitness(solution, sol_idx): ## evaluation function

      # We need a NN model 
      # We have some vectors 
      # we have to find the optimal vector 
        loss=0.0 #.cpu()
        model_weights_dict = tg.model_weights_as_dict(model=model.cpu() ,weights_vector=solution)

        model.load_state_dict(model_weights_dict)

        for idx, (data, target) in enumerate(dataset):
           
           if (datasetName=='usps'):
               vector=[]
               for i in range(len(target)):
                   vector.append(target[i])
               vector=torch.tensor(vector)
               target=vector
               data = torch.reshape(data, (len(data), 1, 28, 28))
           #feature=features(data)
           prediction = model(data)
   
           loss += loss_function(prediction, target).detach().numpy() + 0.00000001  #0.00000001 is added to avoid dividing by zero when loss=0.0
          
        loss /= len(dataset.dataset)
        #print('Loss = {loss}'.format(loss=loss))  

        loss=  1.0 / (loss.item())  

        return loss
        

def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
     #print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

def FedGANew(w,modell,datasett,dataset_name):  #w ==> une liste de vector de poids ==> liste (10 veecteurs de poids)
   #print('begin FedGA')
   global  initial_population, w_size, model, dataset, features, datasetName
   dataset = datasett
   model   = modell
   features= Feature_extractor_Layers()
   initial_population=w
   datasetName=dataset_name
   
   try: 
       initial_population=initial_population.tolist()

   except :
      initial_population=initial_population
   
   for i in initial_population: print(i)
   
   num_generations =20# Number of generations.
   num_parents_mating =len(initial_population) #4 # Number of solutions to be selected as parents in the mating pool.
  
   #print(len(initial_population[0]),len(initial_population[1]),len(initial_population[2]),len(initial_population[3]),'emmmmmmmmmmmmmmmmmmmmmmm')
   parent_selection_type = "rank" # Type of parent selection.
   crossover_type = "single_point" # Type of the crossover operator.
   mutation_type = None # Type of the mutation operator.
   mutation_percent_genes = 10 # Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists.
   keep_parents = -1 # Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing.

# Create an instance of the pygad.GA class
   ga_instance = pygad.GA(num_generations=num_generations, 
                       num_parents_mating=num_parents_mating, 
                       initial_population=initial_population,
                       fitness_func=fitness,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       keep_parents=keep_parents,
                       on_generation=callback_generation)

# Start the genetic algorithm evolution.
   ga_instance.run()

# After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
   #ga_instance.plot_result(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)

# Returning the details of the best solution.  
   solution, solution_fitness, solution_idx = ga_instance.best_solution()
 
#Fetch the parameters of the best solution.
   best_solution_weights = tg.model_weights_as_dict(model=model,  weights_vector=solution)
                                                 
   model.load_state_dict(best_solution_weights)
   return best_solution_weights
   


 
  
  

