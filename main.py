# %%writefile /kaggle/working/UC-GAN-v.2/main.py

from torch.backends import cudnn
from data_loader import get_loader

from solver_substi import Solver_Substi
from solver_transpo import Solver_Transpo
from solver_rotor_enigma_typex import Solver_Rotor_Enigma_Typex

import argparse
import torch
# from pyenigma import enigma, rotor #https://github.com/cedricbonhomme/pyEnigma/tree/master

# from utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "" <- this wasn't the bug

# GPU_NUM = 0 # change to 0 coz I only have 1 GPU: GPU nb0. (run on GoogleColab)
GPU_NUM = 1 # change to 1 coz I have 2 GPU: GPU nb0 and GPU nb1. (run on Kaggle)

device = torch.device(
    f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(0) #argument is 0 coz I only have 1 GPU: GPU nb0. (run on GoogleColab)
torch.cuda.set_device(1) #argument is 1 coz I have 2 GPU: GPU nb0 and GPU nb1. (run on Kaggle)
# torch.cuda.set_device(device) if you trust the error WILL NEVER COME FROM the argument `device` (it means an error has already come from here.)

def str2bool(v):
    return v.lower() in ("true")

def main(config):
    # For fast training.
    cudnn.benchmark = True

    #todo: whyw the hell you receive imgs?
    #for substitution ciphers dataset, bmp files, already offered:
    data_loader = get_loader(
        config.data_image_dir,
        config.batch_size,
        config.mode,
        config.num_workers,
    )


    data_loader_test = get_loader(
        config.data_test_image_dir,
        config.batch_size,
        config.mode,
        config.num_workers,
    )

    #for transposition ciphers dataset:


    # Solver for training and testing:
    # Dynamically select the solver based on the solver_type argument:
    if config.solver_type == "substitution":
        solver = Solver_Substi(data_loader, data_loader_test, config)
    elif config.solver_type == "transposition":        
        solver = Solver_Transpo(data_loader, data_loader_test, config)
    elif config.solver_type == "rotor_enigma_typex":
        solver = Solver_Rotor_Enigma_Typex(data_loader, data_loader_test, config)
    else:
        raise ValueError("Invalid solver type")
    

    if config.mode == "train":
        solver.train()#preds_train = solver.train()
    elif config.mode == "test":
        solver.test()

    ## Report results: performance on train and valid/test sets
    # acc = accuracy_fn(preds_train, ytrain)
    # macrof1 = macrof1_fn(preds_train, ytrain)
    # print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model configuration (hyperparameters).
    parser.add_argument(
        "--c_dim", type=int, default=4, help="dimension of domain labels (1st dataset)"
    )
    parser.add_argument("--solver_type", type=str, default="rotor_enigma_typex",
                choices=["substitution", "transposition", "rotor_enigma_typex"],
                help="type of solver to use")
    parser.add_argument(
        "--g_conv_dim",
        type=int,
        default=32,
        help="number of conv filters in the first layer of G",
    )
    parser.add_argument(
        "--d_conv_dim",
        type=int,
        default=32,
        help="number of conv filters in the first layer of D",
    )
    parser.add_argument(
        "--lambda_cls",
        type=float,
        default=1,
        help="weight for domain classification loss",
    )
    parser.add_argument(
        "--lambda_rec", type=float, default=10, help="weight for reconstruction loss"
    )
    parser.add_argument(
        "--lambda_gp", type=float, default=10, help="weight for gradient penalty"
    )

    # Training configuration (hyperparameters).
    parser.add_argument("--batch_size", type=int,
                        default=32, help="mini-batch size")
    parser.add_argument(
        "--num_iters",
        type=int,
        default=1581600,#1318000,#1054400,  # 200epoch / 1ep = 5272
        help="number of total iterations for training D and G",
    )
    parser.add_argument(  # 0.0001
        "--g_lr", type=float, default=0.00018, help="learning rate for G"
    )
    parser.add_argument(  # 0.0001
        "--d_lr", type=float, default=0.00018, help="learning rate for D"
    )
    parser.add_argument(  # 0.5
        "--beta1", type=float, default=0, help="beta1 for Adam optimizer"
    )
    parser.add_argument(  # 0.999
        "--beta2", type=float, default=0.9, help="beta2 for Adam optimizer"
    )


    # Miscellaneous.
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "test"])

    
    # Parse the arguments up to this point to get the solver_type
    preliminary_args, _ = parser.parse_known_args()
    solver_type = preliminary_args.solver_type

    # Dynamically set the directory arguments based on the solver_type
    if solver_type == "substitution":
        # Directories.  #todo: the dataset in data_rotor_txt/test was a copypasta from data_rotor_txt/train just to let the code training part run.
        parser.add_argument("--data_image_dir", type=str, default=r"/kaggle/working/UC-GAN-v.2/data/train") # (run on Kaggle)

        parser.add_argument("--data_test_image_dir", #todo: the dataset in data_rotor_bmp/test was a copypasta from data_rotor_bmp/train just to let the code training part run. 
                            type=str, default=r"/kaggle/working/UC-GAN-v.2/data/test") # (run on Kaggle)

    elif solver_type == "transposition":
        # Directories.  #todo: the dataset in data_rotor_txt/test was a copypasta from data_rotor_txt/train just to let the code training part run.
        parser.add_argument("--data_image_dir", type=str, default=r"/kaggle/working/UC-GAN-v.2/data_transpo_bmp/train") # (run on Kaggle)

        parser.add_argument("--data_test_image_dir", #todo: the dataset in data_rotor_bmp/test was a copypasta from data_rotor_bmp/train just to let the code training part run. 
                            type=str, default=r"/kaggle/working/UC-GAN-v.2/data_transpo_bmp/test") # (run on Kaggle)

    elif solver_type == "rotor_enigma_typex":
        # Directories.
        parser.add_argument("--data_image_dir", type=str, default=r"/kaggle/working/UC-GAN-v.2/data_enigma_typex_bmp/train") # (run on Kaggle)

        parser.add_argument("--data_test_image_dir", #todo: the dataset in data_rotor_bmp/test was a copypasta from data_rotor_bmp/train just to let the code training part run. 
                            type=str, default=r"/kaggle/working/UC-GAN-v.2/data_enigma_typex_bmp/test") # (run on Kaggle)
    else:
        raise ValueError("Invalid solver type")

    parser.add_argument('-f')

    config = parser.parse_args()

    main(config)
