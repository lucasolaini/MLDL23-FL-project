import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--dataset', type=str, choices=['idda', 'femnist'], required=True, help='dataset name')
    parser.add_argument('--niid', action='store_true', default=False,
                        help='Run the experiment with the non-IID partition (IID by default). Only on FEMNIST dataset.')
    parser.add_argument('--model', type=str, choices=['deeplabv3_mobilenetv2', 'resnet18', 'cnn'], help='model name')
    parser.add_argument('--num_rounds', type=int, help='number of rounds')
    parser.add_argument('--num_epochs', type=int, help='number of local epochs')
    parser.add_argument('--clients_per_round', type=int, help='number of clients trained per round')
    parser.add_argument('--hnm', action='store_true', default=False, help='Use hard negative mining reduction or not')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--bs', type=int, default=64, help='batch size')
    parser.add_argument('--wd', type=float, default=10**-6, help='weight decay')
    parser.add_argument('--m', type=float, default=0.9, help='momentum')
    parser.add_argument('--print_train_interval', type=int, default=10, help='client print train interval')
    parser.add_argument('--print_test_interval', type=int, default=10, help='client print test interval')
    parser.add_argument('--eval_interval', type=int, default=10, help='eval interval')
    parser.add_argument('--test_interval', type=int, default=10, help='test interval')
    parser.add_argument('--backup_folder', type=str, help='directory in which the model states')
    parser.add_argument('--backup', action='store_true', default=False, help='Decides if to load the model state or not')
    parser.add_argument('--backup_path', type=str, default='', help='backup file path. Use only if --backup is True')
    parser.add_argument('--run_id', type=str, default='', help='wandb: id of the run interrupted. Use only if --backup is True')
    parser.add_argument('--clients_selection_strategy', type=str, choices=['uniform', 'high', 'low', 'powerofchoice'], default='uniform', help='clients selection strategy')
    parser.add_argument('--d', type=int, default=20, help='clients selected in power of choice')
    parser.add_argument('--dom_gen', action='store_true', default=False, help='use the rotated dataset if True')
    parser.add_argument('--leave_one_out', type=int, default=None, help='selects which rotated set is not used for training')
    parser.add_argument('--FedSR', action='store_true', default=False, help='use FedSR if True')
    parser.add_argument('--FedIR', action='store_true', default=False, help='use FedIR if True')
    parser.add_argument('--FedVC', action='store_true', default=False, help='use FedVC if True')
    parser.add_argument('--run_name', type=str, default='', help='wandb run name')
    parser.add_argument('--project_name', type=str, default='', help='wandb project name')
    parser.add_argument('--L2R_coeff', type=float, default=0.01, help='alpha L2R')
    parser.add_argument('--CMI_coeff', type=float, default=0.001, help='alpha CMI')

    return parser
