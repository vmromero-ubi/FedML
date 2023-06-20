import logging

import numpy as np
import wandb

from .client_dsgd import ClientDSGD
from .client_pushsum import ClientPushsum
from .topology_manager import TopologyManager


def cal_regret(client_list, client_number, t):
    regret = 0
    for client in client_list:
        regret += np.sum(client.get_regret())

    regret = regret / (client_number * (t + 1))
    return regret

#the construction for FedML_decentralized_fl includes the client_number, client_id_list, streaming_data, model, model_cache, args
# client_number is used for topology generation
# client_id_list refers to the ids of clients participating in this simulation. This information is used to index the dataset herein referred to as streaming_data
# streaming_data is the dat partitioned across different clients participating in this simulation
# model refers to the model being optimized by the users
# model_cache is something passed to the client constructors
# args refers to the arguments of the current simulation
def FedML_decentralized_fl_mod(
    client_number, client_id_list, streaming_data, model, model_cache, args
):
    # network parameters
    b_symmetric = args.b_symmetric # a boolean value that determines if the network is symmetric (true) or asymmetric (false)
    topology_neighbors_num_undirected = args.topology_neighbors_num_undirected # the number of undirected neighbors for each participating node, used by the topology generator
    topology_neighbors_num_directed = args.topology_neighbors_num_directed # the number of directed neighbors for each participating node, used by the topology generator

    #client parameters
    lr_rate = args.learning_rate # a parameter that controlls the speed at which the network learns as a ratio of the gradients, passed at each client constructor
    batch_size = args.batch_size # the number of datapoints at each abtch, passed at each client constructor
    weight_decay = args.weight_decay # a parameters for weight decay, passed at each client constructor
    latency = args.latency # parameter for latency, passed at each client constructor
    time_varying = args.time_varying #parameter flag for time varying, passed at each client contructor
    
    #training parameters
    iteration_number_T = args.iteration_number # set the iteration number, i.e. the number of times the serverless network will train the network
    epoch = args.epoch # parameter for no of epochs, used to affect training duration, not passed at each node

    # create the network topology topology
    logging.info("generating topology")
    if b_symmetric:
        topology_manager = TopologyManager(
            client_number,
            True,
            undirected_neighbor_num=topology_neighbors_num_undirected,
        )
    else:
        topology_manager = TopologyManager(
            client_number,
            False,
            undirected_neighbor_num=topology_neighbors_num_undirected,
            out_directed_neighbor=topology_neighbors_num_directed,
        )
    topology_manager.generate_topology()
    logging.info("finished topology generation")
    if b_symmetric:
        logging.info("Generated symmetric  topology:\n {}".format(topology_manager.topology_symmetric))
    else:
        logging.info("Generated asymmetric topology:\n {}".format(topology_manager.topology_asymmetric))

    # # create all client instances (each client will create an independent model instance)
    # client_list = []
    # for client_id in client_id_list:
    #     client_data = streaming_data[client_id]
    #     # print("len = " + str(len(client_data)))

    #     if args.mode == "PUSHSUM":

    #         client = ClientPushsum(
    #             model,
    #             model_cache,
    #             client_id,
    #             client_data,
    #             topology_manager,
    #             iteration_number_T,
    #             learning_rate=lr_rate,
    #             batch_size=batch_size,
    #             weight_decay=weight_decay,
    #             latency=latency,
    #             b_symmetric=b_symmetric,
    #             time_varying=time_varying,
    #         )

    #     elif args.mode == "DOL":

    #         client = ClientDSGD(
    #             model,
    #             model_cache,
    #             client_id,
    #             client_data,
    #             topology_manager,
    #             iteration_number_T,
    #             learning_rate=lr_rate,
    #             batch_size=batch_size,
    #             weight_decay=weight_decay,
    #             latency=latency,
    #             b_symmetric=b_symmetric,
    #         )

    #     else:
    #         client = ClientDSGD(
    #             model,
    #             model_cache,
    #             client_id,
    #             client_data,
    #             topology_manager,
    #             iteration_number_T,
    #             learning_rate=lr_rate,
    #             batch_size=batch_size,
    #             weight_decay=weight_decay,
    #             latency=latency,
    #             b_symmetric=b_symmetric,
    #         )

    #     client_list.append(client)

    # log_file_path = "./log/decentralized_fl.txt"
    # f_log = open(log_file_path, mode="w+", encoding="utf-8")

    # for t in range(iteration_number_T * epoch):
    #     logging.info("--- Iteration %d ---" % t)

    #     if args.mode == "DOL" or args.mode == "PUSHSUM":
    #         for client in client_list:
    #             # line 4: Locally computes the intermedia variable
    #             client.train(t)

    #             # line 5: send to neighbors
    #             client.send_local_gradient_to_neighbor(client_list)

    #         # line 6: update
    #         for client in client_list:
    #             client.update_local_parameters()
    #     else:
    #         for client in client_list:
    #             client.train_local(t)

    #     regret = cal_regret(client_list, client_number, t)
    #     # print("regret = %s" % regret)

    #     wandb.log({"Average Loss": regret, "iteration": t})

    #     f_log.write("%f,%f\n" % (t, regret))

    # f_log.close()
    # wandb.save(log_file_path)
