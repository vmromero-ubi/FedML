import random

import numpy as np
import torch
import logging

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
        logging.info("Training at client{} for iteration{}".format(self.client_idx, iteration_id))
        self.model_trainer.set_id(self.client_idx)
        self.model_trainer.set_model_params(self.model)
        self.model_trainer.train(self.local_training_data, self.device, self.args)
        weights = self.model_trainer.get_model_params()
        self.model = weights
        return weights

    def get_regret(self):
        return self.loss_in_each_iteration
    
    def send_local_weights_to_neighbors(self, client_list):
        return

    # simulation
    # more like send this node's weights to other nodes in the neighborhood
    def send_local_gradient_to_neighbor(self, client_list):
        logging.info("Sending local gradient updates of node {} to its neighbors")
        # for index in range(self.topology):
        # for index in range(len(self.topology)):
        #     if self.topology[index] != 0 and index != self.id:
        #         client = client_list[index]
        #         logging.info("@Client{}, {} is a neighbor, sending...".format(self.client_idx, index))
        #         client.receive_neighbor_gradients(
        #             self.id,
        #             self.model_x,
        #             self.topology[index],
        #             self.omega * self.topology[index],
        #         )

    def receive_neighbor_gradients(self, client_id, model_x, topo_weight, omega):
        self.neighbors_weight_dict[client_id] = model_x
        self.neighbors_topo_weight_dict[client_id] = topo_weight
        self.neighbors_omega_dict[client_id] = omega

    def update_local_parameters(self):
        logging.info("Updating_local_parameters at node {}".format(self.client_idx))
        # # update x_{t+1/2}
        # for x_paras in self.model_x.parameters():
        #     x_paras.data.mul_(self.topology[self.id])

        # for client_id in self.neighbors_weight_dict.keys():
        #     model_x = self.neighbors_weight_dict[client_id]
        #     topo_weight = self.neighbors_topo_weight_dict[client_id]
        #     for x_paras, x_neighbor in zip(
        #         list(self.model_x.parameters()), list(model_x.parameters())
        #     ):
        #         temp = x_neighbor.data.mul(topo_weight)
        #         # print("topo_weight=" + str(topo_weight))
        #         # print("x_neighbor=" + str(temp))
        #         x_paras.data.add_(temp)

        # # update omega
        # self.omega *= self.topology[self.id]
        # for client_id in self.neighbors_omega_dict.keys():
        #     self.omega += self.neighbors_omega_dict[client_id]

        # # print(self.omega)

        # # update parameter z (self.model)
        # for x_params, z_params in zip(
        #     list(self.model_x.parameters()), list(self.model.parameters())
        # ):
        #     # print("1.0 / self.omega=" + str(1.0 / self.omega))
        #     temp = x_params.data.mul(1.0 / self.omega)
        #     z_params.data.copy_(temp)
