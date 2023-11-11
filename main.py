# %%writefile /kaggle/working/UC-GAN-v.2/main.py

from torch.backends import cudnn
from data_loader import get_loader
from solver_substi import Solver_Substi
from solver_transpo import Solver_Transpo
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
    
    # #  create an Enigma machine with desired rotor and reflector configurations:
    # engine = enigma.Enigma(rotor.ROTOR_Reflector_A, rotor.ROTOR_I,
    #                        rotor.ROTOR_II, rotor.ROTOR_III, key="AAA",
    #                        plugs="AV BS CG DL FU HZ IN KM OW RX")
    
    # # use the `engine` to encrypt the message "Hello World.":
    # secret = engine.encipher("Hello World") #= "Qgqop Vyzxp"

    # '''
    # Enigma encryption is symmetric, which means that the same settings is used to both encrypt or decrypt a message.
    # In a real Enigma machine, after each letter is encrypted, the rotors move, changing the internal state, coz the rotor advances its state in one-direction.
    # When you try to encrypt the already encrypted message, you're starting from a different rotor state, which leads to incorrect decryption.
    # To decrypt the message, you need to reset the rotor states to their initial positions, s.t. the Enigma is the same Enigma machine configuration, before calling the encipher method again.
    # If you're not resetting the machine state, encrypting the result of the first encryption will produce a different result than encrypting "Hello World" directly.
    # '''
    # # Reset the Enigma machine state after each encryption to decrypt the message:
    # engine = enigma.Enigma(rotor.ROTOR_Reflector_A, rotor.ROTOR_I,
    #                        rotor.ROTOR_II, rotor.ROTOR_III, key="AAA",
    #                        plugs="AV BS CG DL FU HZ IN KM OW RX")
    # print(engine.encipher(secret)) #decryption, = "Hello World"


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

    # Solver for training and testing: #todo: uncomment the solver_cipher you wanna test:
    solver = Solver_Substi(data_loader, data_loader_test, config)
    # solver = Solver_Transpo(data_loader, data_loader_test, config)
    # solver = Solver_Enigma(data_loader, data_loader_test, config)
    # solver = Solver_Rotor(data_loader, data_loader_test, config)


    if config.mode == "train":
        preds_train = solver.train()
    # elif config.mode == "test":
        # solver.test()
    

    ## Report results: performance on train and valid/test sets
    # acc = accuracy_fn(preds_train, ytrain)
    # macrof1 = macrof1_fn(preds_train, ytrain)
    # print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")



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
