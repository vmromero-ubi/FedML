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
        self.model_trainer = create_model_trainer(model, args)  #the class that allows us to train a model, this is implemented by fedml      
        self.model = model
        self.client_list = []
        logging.info("INITIALIZING DECENTRALIZED TRAINING...")
        logging.info("ARGUMENTS = {}".format(vars(self.args)))
        logging.info("DEVICE = {}".format(self.device))
        logging.info("SIZE OF TRAINING GLOBAL {}/{}".format(len(self.train_global), self.train_data_num_in_total))
        logging.info("SIZE OF TESTING GLOBAL {}/{}".format(len(self.test_global), self.test_data_num_in_total))
        logging.info("MODEL = {}".format(self.model))

        # prepare clients 
        client_id_list= [i for i in range(self.args.client_num_participant)]
        client_data_index = self.sample_without_rep() # this is not used yet, i think... lol
        logging.info("Client list: {}".format(client_id_list))
        # logging.info("Corrseponding datasets: {}".format(client_data_index))

        logging.info("generating topology for {} participating clients".format(self.args.client_num_participant))
        if self.args.b_symmetric:
            self.topology_manager = TopologyManagerMod(
                self.args.client_num_participant,
                True,
                undirected_neighbor_num=self.args.topology_neighbors_num_undirected,
            )
        else:
            self.topology_manager = TopologyManagerMod(
                self.args.client_num_participant,
                False,
                undirected_neighbor_num=self.args.topology_neighbors_num_undirected,
                out_directed_neighbor=self.args.topology_neighbors_num_directed,
            )
        self.topology_manager.generate_topology()
        logging.info("finished topology generation")
        if self.args.b_symmetric:
            logging.info("Generated symmetric  topology:\n {}".format(self.topology_manager.topology_symmetric))
        else:
            logging.info("Generated asymmetric topology:\n {}".format(self.topology_manager.topology_asymmetric))
        
        self.setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, self.model_trainer)


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

    def train(self):
        logging.info("DECENTRALIZED TRAINING STARTED")

        for t in range(self.args.iteration_number * self.args.epochs):
            train_metrics_pre = {"num_samples": [], "num_correct": [], "losses": []}
            test_metrics_pre = {"num_samples": [], "num_correct": [], "losses": []}
            train_metrics_post = {"num_samples": [], "num_correct": [], "losses": []}
            test_metrics_post = {"num_samples": [], "num_correct": [], "losses": []}            
            for client in self.client_list:
                #gather test metrics on train data 
                train_local_metrics = client.local_test(False)
                train_metrics_pre["num_samples"].append(copy.deepcopy(train_local_metrics["test_total"]))
                train_metrics_pre["num_correct"].append(copy.deepcopy(train_local_metrics["test_correct"]))
                train_metrics_pre["losses"].append(copy.deepcopy(train_local_metrics["test_loss"]))
                train_local_metrics = client.local_test(True)
                test_metrics_pre["num_samples"].append(copy.deepcopy(train_local_metrics["test_total"]))
                test_metrics_pre["num_correct"].append(copy.deepcopy(train_local_metrics["test_correct"]))
                test_metrics_pre["losses"].append(copy.deepcopy(train_local_metrics["test_loss"]))               
                # 
                client.train(t)
                client.send_local_gradient_to_neighbor(self.client_list)
            for client in self.client_list:
                if(self.args.weighted_aggregation):
                    client.update_local_parameters_weighted()
                else:
                    client.update_local_parameters()
            
                    # test on training dataset
            train_acc = sum(train_metrics_pre["num_correct"]) / sum(train_metrics_pre["num_samples"])
            train_loss = sum(train_metrics_pre["losses"]) / sum(train_metrics_pre["num_samples"])

            # test on test dataset
            test_acc = sum(test_metrics_pre["num_correct"]) / sum(test_metrics_pre["num_samples"])
            test_loss = sum(test_metrics_pre["losses"]) / sum(test_metrics_pre["num_samples"])
            logging.info(" {} Metrics: {} | {} | {} | {}".format(t, train_acc, train_loss, test_acc, test_loss))
            #update the topology, this currently does not work
            # if(self.args.time_varying):
            #   self.topology_manager.generate_topology()
            #   logging.info("Topo: {}".format(self.topology_manager.topology_symmetric))
            #   for client in self.client_list:
            #       client.update_topology(self.topology_manager)

        logging.info("TRAINING DONE.")

    #this is the local test on all clients function from the fedavg api
    # this cannot be used directly in the decentralized setting since we expect the models stored at each node to diverge. 
    # def _local_test_on_all_clients(self, round_idx):

    #     logging.info("################local_test_on_all_clients : {}".format(round_idx))

    #     train_metrics = {"num_samples": [], "num_correct": [], "losses": []}

    #     test_metrics = {"num_samples": [], "num_correct": [], "losses": []}

    #     client = self.client_list[0]

    #     for client_idx in range(self.args.client_num_in_total):
    #         """
    #         Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
    #         the training client number is larger than the testing client number
    #         """
    #         if self.test_data_local_dict[client_idx] is None:
    #             continue
    #         client.update_local_dataset(
    #             0,
    #             self.train_data_local_dict[client_idx],
    #             self.test_data_local_dict[client_idx],
    #             self.train_data_local_num_dict[client_idx],
    #         )
    #         # train data
    #         train_local_metrics = client.local_test(False)
    #         train_metrics["num_samples"].append(copy.deepcopy(train_local_metrics["test_total"]))
    #         train_metrics["num_correct"].append(copy.deepcopy(train_local_metrics["test_correct"]))
    #         train_metrics["losses"].append(copy.deepcopy(train_local_metrics["test_loss"]))

    #         # test data
    #         test_local_metrics = client.local_test(True)
    #         test_metrics["num_samples"].append(copy.deepcopy(test_local_metrics["test_total"]))
    #         test_metrics["num_correct"].append(copy.deepcopy(test_local_metrics["test_correct"]))
    #         test_metrics["losses"].append(copy.deepcopy(test_local_metrics["test_loss"]))

    #     # test on training dataset
    #     train_acc = sum(train_metrics["num_correct"]) / sum(train_metrics["num_samples"])
    #     train_loss = sum(train_metrics["losses"]) / sum(train_metrics["num_samples"])

    #     # test on test dataset
    #     test_acc = sum(test_metrics["num_correct"]) / sum(test_metrics["num_samples"])
    #     test_loss = sum(test_metrics["losses"]) / sum(test_metrics["num_samples"])

    #     stats = {"training_acc": train_acc, "training_loss": train_loss}
    #     if self.args.enable_wandb:
    #         wandb.log({"Train/Acc": train_acc, "round": round_idx})
    #         wandb.log({"Train/Loss": train_loss, "round": round_idx})

    #     mlops.log({"Train/Acc": train_acc, "round": round_idx})
    #     mlops.log({"Train/Loss": train_loss, "round": round_idx})
    #     logging.info(stats)

    #     stats = {"test_acc": test_acc, "test_loss": test_loss}
    #     if self.args.enable_wandb:
    #         wandb.log({"Test/Acc": test_acc, "round": round_idx})
    #         wandb.log({"Test/Loss": test_loss, "round": round_idx})

    #     mlops.log({"Test/Acc": test_acc, "round": round_idx})
    #     mlops.log({"Test/Loss": test_loss, "round": round_idx})
    #     logging.info(stats)