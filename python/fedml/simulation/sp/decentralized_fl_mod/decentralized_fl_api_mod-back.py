import logging
import copy

import numpy as np
import wandb

from .client_dsgd_mod import ClientDSGDMod
from .client_pushsum_mod import ClientPushsumMod
from .topology_manager_mod import TopologyManagerMod
from fedml.ml.trainer.trainer_creator import create_model_trainer


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
class FedML_decentralized_fl_mod(object):
    def __init__(self, args, device, dataset, model):
        self.args = args
        self.device = device
        [ #get some information about the dataset
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ] = dataset
        #set this as self parameters
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num

        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.dataset_class_num = class_num
        #
        self.model_trainer = create_model_trainer(model, args) #not sure if this will be used, but it assumed that this should not be detrimental to us in any way        
        self.model = model
        self.client_list = []
        logging.info("INITIALIZING DECENTRALIZED TRAINING...")
        logging.info("ARGUMENTS = {}".format(vars(self.args)))
        logging.info("DEVICE = {}".format(self.device))
        logging.info("SIZE OF TRAINING GLOBAL {}/{}".format(len(self.train_global), self.train_data_num_in_total))
        logging.info("SIZE OF TESTING GLOBAL {}/{}".format(len(self.test_global), self.test_data_num_in_total))
        logging.info("MODEL = {}".format(self.model))

    # def __init__(self, client_number, client_id_list, dataset, model, model_cache, args):

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
        client_number = args.client_num_participant
        # create the network topology topology
        
        # prepare clients 
        client_id_list= [i for i in range(self.args.client_num_participant)]
        client_data_index = self.sample_without_rep()
        logging.info("Client list: {}".format(client_id_list))
        logging.info("Corrseponding datasets: {}".format(client_data_index))

        logging.info("generating topology for {} participating clients".format(client_number))
        if b_symmetric:
            self.topology_manager = TopologyManagerMod(
                client_number,
                True,
                undirected_neighbor_num=topology_neighbors_num_undirected,
            )
        else:
            self.topology_manager = TopologyManagerMod(
                client_number,
                False,
                undirected_neighbor_num=topology_neighbors_num_undirected,
                out_directed_neighbor=topology_neighbors_num_directed,
            )
        self.topology_manager.generate_topology()
        logging.info("finished topology generation")
        if b_symmetric:
            logging.info("Generated symmetric  topology:\n {}".format(self.topology_manager.topology_symmetric))
        else:
            logging.info("Generated asymmetric topology:\n {}".format(self.topology_manager.topology_asymmetric))
        
        self.setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, self.model_trainer)

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

    # this function is being prepared for when clients will be sampled over total population to introduce variations in the experiments
    # however, the current implementation assumes that the clients are sampled consequtively from 0 to number of participants 
    # as indicated in the configuration file 
    def sample_without_rep(self):
        client_idx_list = []
        total_clients = self.args.client_num_in_total
        participating_clients = self.args.client_num_participant

        while len(client_idx_list) < participating_clients:
            candidate = np.random.randint(0, total_clients)
            if candidate not in client_idx_list:
                client_idx_list.append(candidate)
        return client_idx_list

    def setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        for client_idx in  range(self.args.client_num_participant): 
            c = ClientPushsumMod(
                copy.deepcopy(model_trainer.get_model_params()), 
                client_idx, 
                train_data_local_dict[client_idx],
                test_data_local_dict[client_idx],
                train_data_local_num_dict[client_idx],
                self.topology_manager,
                self.args,
                self.device,
                model_trainer,
            )
            self.client_list.append(c)

    # def __init__(
    #     self,
    #     model,
    #     model_cache,
    #     client_id,
    #     training_data_local_dict,       #added from fedavg           
    #     test_data_local_dict,           #added from fedavg
    #     training_data_local_num_dict,   #added from fedavg
    #     topology_manager,
    #     iteration_number,
    #     learning_rate,
    #     batch_size,
    #     weight_decay,
    #     latency,
    #     b_symmetric,
    #     time_varying,
    #     args,                           #added from fedavg
    #     device,                         #added from fedavg
    #     model_trainer,                  #added from fedavg
    # ):
        #  logging.info("Setting up {}/{} participants for this simulation".format(self.args.client_num_participant, self.args.client_num_in_total))

    def train(self):
        logging.info("Training in a decentralized manner")
        for t in range(self.args.iteration_number * self.args.epochs):
            for client in self.client_list:
                client.train(t)
                client.send_local_gradient_to_neighbor(self.client_list)
            for client in self.client_list:
                client.update_local_parameters()
        logging.info("Training is completed.")

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
