import fedml
from fedml import FedMLRunner
import logging

if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()
    logging.info("Total number of clients in the network:{}".format(args.client_num_in_total))

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedml.data.load(args)

    # load model
    model = fedml.model.create(args, output_dim)

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model)
    fedml_runner.run()
