import random

import numpy as np
import torch
import logging
import copy

class ClientPushsumMod(object):
    def __init__(
        self,
        model,
        client_idx,
        local_training_data,            # added from fedavg           
        local_test_data,                # added from fedavg
        local_sample_number,            # added from fedavg
        topology_manager,
        args,                           # added from fedavg
        device,                         # added from fedavg
        model_trainer,                  # added from fedavg
    ):
        logging.info("Initializing new client...")
        self.model = model
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.device = device 
        self.model_trainer = model_trainer
        self.topology = topology_manager
        self.args = args
        logging.info("MODEL: {}".format(self.model))
        logging.info("CLIENT ID:{}".format(self.client_idx))
        logging.info("LOCAL_TRAINING_DATA:{}".format(self.local_training_data))
        logging.info("LOCAL_TESTING_DATA:{}".format(self.local_test_data))
        logging.info("TRAIING_DATA_LOCAL_NUM_DICT:{}".format(self.local_sample_number))
        logging.info("DEVICE:{}".format(self.device))
        logging.info("MODEL_TRAINER:{}".format(self.model_trainer))

        # neighbors_weight_dict used for receiving weights from other clients
        self.neighbors_weight_dict = dict()
        self.neighbors_omega_dict = dict()
        self.neighbors_topo_weight_dict = dict()

    #this code is not used.
    def train_local(self, iteration_id):
        self.optimizer.zero_grad()
        train_x = torch.from_numpy(self.streaming_data[iteration_id]["x"])
        train_y = torch.FloatTensor([self.streaming_data[iteration_id]["y"]])
        outputs = self.model(train_x)
        loss = self.criterion(outputs, train_y)  # pylint: disable=E1102
        loss.backward()
        self.optimizer.step()
        self.loss_in_each_iteration.append(loss)

    def train(self, iteration_id):
        # if self.client_idx == 0:
        #     logging.info("TRAINING... Client:{}, Iteration:{}, \nPRE_WEIGHTS:{}".format(self.client_idx, iteration_id, self.model))
        # logging.info("Pre-training weights: {} at client{} for it".format(self.model))
        self.model_trainer.set_id(self.client_idx)
        self.model_trainer.set_model_params(copy.deepcopy(self.model))
        self.model_trainer.train(self.local_training_data, self.device, self.args)
        weights = self.model_trainer.get_model_params()
        self.model = copy.deepcopy(weights)
        # if self.client_idx == 0:
        #     logging.info("TRAINING... Client:{}, Iteration:{}, \nPOST_WEIGHTS:{}".format(self.client_idx, iteration_id, self.model))
        # return weights

        # local test must be called only when this client is sure that its own model is loaded
        # in the model trainer. otherwise, it will test against some other clients data

    def update_topology(self, new_topology):
        self.topology = new_topology

    def local_test(self, b_use_test_dataset):
        # set the parameters at the model trainer to this node's model also set the id
        self.model_trainer.set_id(self.client_idx)
        self.model_trainer.set_model_params(self.model)
        if b_use_test_dataset:
            test_data = self.local_test_data

        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics
    
    def get_regret(self):
        return self.loss_in_each_iteration
    
    def send_local_weights_to_neighbors(self, client_list):
        return

    # simulation
    # more like send this node's weights to other nodes in the neighborhood
    # assumes that deep copy is not necessary here because they have been set post training
    def send_local_gradient_to_neighbor(self, client_list):
        # logging.info("Sending local gradient updates of node {} to its neighbors")
        #assume it is all symmetric
        neighbor_list = []
        for index in range (self.args.client_num_participant):
            if self.topology.topology_symmetric[self.client_idx][index]!=0 and index!=self.client_idx:
                # logging.info("Client {} should send weights to client {}".format(self.client_idx, index))
                neighbor_list.append(index)
                receiver_client = client_list[index]
                receiver_client.receive_neighbor_gradients(
                    self.client_idx,
                    self.model
                )
        # if(self.client_idx==1):
        #     logging.info("Sent to: {}".format(neighbor_list))

    def receive_neighbor_gradients(self, client_id, model_x):
        self.neighbors_weight_dict[client_id] = model_x
        # logging.info("Client{} received weights from client{}, len:{}".format(self.client_idx, client_id, len(self.neighbors_weight_dict)))

    def update_local_parameters(self):
        normalizer = 1/(1+len(self.neighbors_weight_dict)) # a naive implementation of weighte averaging. we assume that the number of samples at each ndoe is similar. 
        # if(self.client_idx == 1):
        #     logging.info("AGGREGATION... ID: {} | TOTAL: {} | NORM: {}".format(self.client_idx, self.args.client_num_participant, normalizer))
        # logging.info("AVAILABLE WEIGHTS: {}, NORMALIZER: {}".format(len(self.neighbors_weight_dict), normalizer))
        new_model = copy.deepcopy(self.model)
                                  
        for k in new_model.keys():
            # logging.info("AGGREGATING FOR:  {}".format(k))
            # logging.info("INITITAL: {}".format(new_model[k]))
            for i in self.neighbors_weight_dict:
                neighbor_weight = self.neighbors_weight_dict[i]
                # logging.info("WITH: {}".format(neighbor_weight[k]))
                new_model[k] += neighbor_weight[k]
                # logging.info("NEW: {}".format(new_model[k]))

            new_model[k] *= normalizer 
            # logging.info("NORMALIZED: {}".format(new_model[k]))
        self.model = copy.deepcopy(new_model)
        self.neighbors_weight_dict = dict() #reset my weight dict


