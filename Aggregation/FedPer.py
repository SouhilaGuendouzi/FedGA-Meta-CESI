import copy
import torch


# 1 M weights for each client 
# 3 clients 
# w_avg ==> 3 vectors, and each vector has 1M weights

# i NEED TO AVERAGE THE WEIGHTS OF EACH CLIENT
# w_avg ==> 1 vector, and this vector has 1M weights

def FedPer(w):
    
    w_avg = copy.deepcopy(w[0]) 

    for k in w_avg.keys():
        for i in range(1, len(w)):
           
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
 
    return w_avg