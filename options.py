import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments 
    parser.add_argument('--id', type=int, default=0, help="id of an entity") 
    parser.add_argument('--epochs', type=int, default=50, help="rounds of FL") # epochs number is the same as number of layers in FedMA
    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
    parser.add_argument('--num_edges', type=int, default=5, help="number of edges per fog")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_epochs', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--maml', type=bool, default=False, help="if the training is MAML")
    parser.add_argument('--local_bs', type=int, default=32, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=64, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    #maml arguments
    parser.add_argument('--meta_lr', type=float, default=0.01, help="meta learning rate")
    parser.add_argument('--num_tasks', type=int, default=2, help="number of tasks sampled per meta-update")

    #domain adaptation arguments
    parser.add_argument('--da', type=bool, default=False, help="if the training is domain adaptation")
    parser.add_argument('--lambda_coral', type=float, default=0.5, help="learning rate for domain adaptation")

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True', help="Whether use max pooling rather than strided convolutions")

    # FL arguments
    parser.add_argument('--aggr', type=str, default='NoFL', help="name of aggregation method; if NoFL, then no aggregation")
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', type=str, default='iid', help="iid or non_iid")
    parser.add_argument('--num_classes', type=int, default=1, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of images")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print', default=True)
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients', default=True)
     # socket parameters
    parser.add_argument('--portFog', type=int, default=0, help="server port") 
    parser.add_argument('--portCloud', type=int, default=0, help="server port") 


    #parser.add_argument('--id', type=int, default=1, help="ID of the edge entity")
    parser.add_argument('--ipAddress', type=str, default='localhost', help="IP address of the edge server")
    parser.add_argument('--port', type=int, default=2500, help="Port number of the edge server")
    parser.add_argument('--latitude', type=float, default=0.0, help="Latitude of the edge server location")
    parser.add_argument('--longitude', type=float, default=0.0, help="Longitude of the edge server location")
    parser.add_argument('--cpuMips', type=int, default=1000, help="CPU capacity of the edge server in MIPS")
    parser.add_argument('--ramGB', type=int, default=4, help="RAM of the edge server in GB")
    parser.add_argument('--storageGB', type=int, default=500, help="Storage of the edge server in GB")
    parser.add_argument('--latencyMs', type=int, default=10, help="Network latency of the edge server in ms")
    parser.add_argument('--networkB', type=int, default=100, help="Bandwidth of the edge server network in Mbps")
    parser.add_argument('--AiTask', type=str, default='task1', help="AI task assigned to the edge server")
    parser.add_argument('--portFlask', type=int, default=8080, help="Port number of the edge flask web server")
    parser.add_argument('--nbR', type=int, default=2, help="number of requests transactions")
    parser.add_argument('--bsize', type=int, default=20, help="size of the block")
    parser.add_argument('--task', type=str, default='mnist', help="the decision-making task")
    parser.add_argument('--number_of_edges', type=int, default=5, help="number of edges")
    parser.add_argument('--malicious_edges', type=float, default=0.2, help="% of malicious edges")
    parser.add_argument('--stragglers_edges', type=float, default=0.2, help="% of stragglers edges")
    parser.add_argument('--trust', type=int, default=20, help="number of edges")
    parser.add_argument('--history', type=str, default='session_100', help="the path of models history")
    parser.add_argument('--user', type=str, default='bguendouzi', help="the path of models history")
    parser.add_argument('--consensus', type=str, default='PoW', help="the consensus algorithm")
    parser.add_argument('--do_it', type=bool, default=False, help="execute our consensus or not")

    args = parser.parse_args()
    return args