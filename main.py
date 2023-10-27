from torch.backends import cudnn
from data_loader import get_loader
from solver_substi import Solver
import argparse
import torch
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

    # Solver for training and testing.
    solver = Solver(data_loader, data_loader_test, config)

    if config.mode == "train":
        solver.train()
    elif config.mode == "test":
        solver.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument(
        "--c_dim", type=int, default=4, help="dimension of domain labels (1st dataset)"
    )
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

    # Training configuration.
    parser.add_argument("--batch_size", type=int,
                        default=32, help="mini-batch size")
    parser.add_argument(
        "--num_iters",
        type=int,
        default=1054400,  # 200epoch / 1ep = 5272
        help="number of total iterations for training D",
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

    # Directories.
    # parser.add_argument("--data_image_dir", type=str, default="/content/UC-GAN-Unified-cipher-generative-adversarial-network/data/train") # (run on GoogleColab)
    parser.add_argument("--data_image_dir", type=str, default="/kaggle/working/UC-GAN-v.2/data/train") # (run on Kaggle)


    parser.add_argument("--data_test_image_dir",
                        type=str, default="/kaggle/working/UC-GAN-v.2/data/test") # (run on Kaggle)
                        #default="/content/UC-GAN-Unified-cipher-generative-adversarial-network/data/test" if run on GoogleColab
    parser.add_argument('-f')

    config = parser.parse_args()

    main(config)