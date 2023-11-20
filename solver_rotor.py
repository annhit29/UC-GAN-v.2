from model import Generator
from model import Discriminator
import torch
import torch.nn.functional as F
import numpy as np
import copy
import os

#----files for Lorenz machine:----
# import lib.lorenz_machine.lorenz.machines as lorenz_cipher
from lib.lorenz_machine.lorenz.machines import SZ40
# import lib.lorenz_machine.lorenz.patterns as lorenz_patterns
from lib.lorenz_machine.lorenz.patterns import KH_CAMS
# import telegrahy utility library
from lib.lorenz_machine.lorenz.telegraphy import Teleprinter

#----files for Enigma machine:----
from pyenigma import enigma, rotor

#----files for TypeX machine:----
from lib.typex_machine.enigma_typex import *


CHARACTERS_NBRS = 100 #type int

# ----Lorenz machine related functions----

# Encrypt line using Lorenz cipher w/ the model SZ40-KH_CAMS. Assume all msgs will be enc./dec. to lowercase:
def lorenz_encrypt(strPT):
    #1. transform the input PT from str to list format
    listPT = Teleprinter.encode(strPT)
    #2. 
    # use the `KH` pattern to encrypt the message.
    machine = SZ40(KH_CAMS)
    #3. encrypt the PT list format to CT list format
    listCT = machine.feed(listPT) # list CT
    #4. transform the CT list format to str format
    strCT = Teleprinter.decode(listCT) #str CT
    #Encryption done

    strCT = str(strCT).lower()

    return strCT

# Decrypt line using Lorenz cipher w/ the model SZ40-KH_CAMS:
def lorenz_decrypt(strCT):
    #1. transform the input CT from str to list format
    listCT = Teleprinter.encode(strCT)
    #2. reset the machine:
    # use the `KH` pattern to encrypt the message.
    machine = SZ40(KH_CAMS)
    #3. encrypt the CT list format to PT list format
    listPT = machine.feed(listCT)
    #4. transform the PT list format to str format
    strPT = Teleprinter.decode(listPT)
    #Decryption done

    strPT = str(strPT).lower()

    return strPT
    

# ----Enigma machine related functions----

# Create an Enigma machine with desired rotor and reflector configurations:
#`engine` is a global variable for Enigma
engine = enigma.Enigma(
    rotor.ROTOR_Reflector_A, rotor.ROTOR_I, rotor.ROTOR_II, rotor.ROTOR_III,
    key="ABC", plugs="AV BS CG DL FU HZ IN KM OW RX"
)

def reset_enigma_machine():
    """
    Reset the state of the Enigma machine by reinitializing with the default configuration.
    """
    return enigma.Enigma(
        rotor.ROTOR_Reflector_A, rotor.ROTOR_I, rotor.ROTOR_II, rotor.ROTOR_III,
        key="ABC", plugs="AV BS CG DL FU HZ IN KM OW RX"
    )

def enigma_encrypt_or_decrypt(inputMsg):
    reset_enigma_machine() #must always reset the machine before each msg encryption/decryption
    outputMsg = engine.encipher(inputMsg) # the encryption function and the decryption function are the same   
    return outputMsg


# ----TypeX machine related functions----
#----Set up the TypeX machine----
t_pb = ""  # Set your plugboard settings here if needed
t_rotors = "TYPEX_A TYPEX_B TYPEX_C TYPEX_D TYPEX_E"  # 5 TypeX rotors
t_reflector = "B"  # Choose the reflector for TypeX
t_ring_settings = "01 01 01 01 01"  # Set ring settings for each rotor
t_initial_positions = "A A A A A"  # Set initial positions for each rotor
#----Encryption/Decryption----
def typex_encrypt_or_decrypt(inputMsg):
    #1. change to uppercase letters (<- the curr only valid alphabets):
    inputMsg = str(inputMsg).upper() #todo: this typex only receives uppercase letters, I'll change it to receive A~Z and a~z. I mean: all the letters receivable by Brown Corpus.
    
    #must always reset the machine before each msg encryption/decryption (step2.~4.):
    #2. Create a Plugboard instance
    plugboard = Plugboard([PlugLead(mapping) for mapping in t_pb.split()])
    #3. Create a MultiRotor instance for TypeX
    multirotor = MultiRotor(t_rotors, t_reflector, t_ring_settings, t_initial_positions)
    #4. Create a TypeX instance with the MultiRotor and Plugboard
    typex = MultiEnigma(multirotor, plugboard)

    #5. Encrypt/Decrypt the message
    outputMsg = typex.encode_decode(inputMsg)
    #Encryption/Decryption done

    #6. change to lowercase letters
    outputMsg = str(outputMsg).lower()

    return outputMsg




"""Rotor machine class"""
class Solver_Rotor(object):
    """Solver for training and testing UC-GAN."""

    def __init__(self, data_loader, data_loader_test, config):
        """Initialize configurations."""

        # Data loader.
        self.data_loader = data_loader
        self.data_loader_test = data_loader_test

        # Model configurations.
        self.c_dim = config.c_dim #=4
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Miscellaneous.
        self.device = torch.device("cpu")# if torch.cpu.is_available() else "cpu")
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #fixme: for some reasons (ridiculous??) the code works w/ cpu but NOT gpu. (yet `torch.cuda.is_available()` = True)

        # Build the model and tensorboard.
        self.build_model()

    def build_model(self):
        """Create a generator and a discriminator."""
        
        self.G = Generator(self.g_conv_dim, self.c_dim)
        self.D = Discriminator(
            self.d_conv_dim, self.c_dim
        )
        self.g_optimizer = torch.optim.Adam(
            self.G.parameters(), self.g_lr, [self.beta1, self.beta2]
        )
        self.d_optimizer = torch.optim.Adam(
            self.D.parameters(), self.d_lr, [self.beta1, self.beta2]
        )

        self.G.to(self.device, dtype=torch.float)
        self.D.to(self.device, dtype=torch.float)

    def create_labels(self, c_org, c_dim=5):
        """Generate target domain labels for debugging and testing."""
        c_trg_list = []
        for i in range(c_dim):
            c_trg = self.label2onehot(torch.ones(c_org.size(0)) * i, c_dim)
            c_trg_list.append(c_trg.to(self.device, dtype=torch.float))
        return c_trg_list

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device, dtype=torch.float)
        dydx = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=weight,
            retain_graph=True,
            create_graph=True,
            only_inputs=True,
            allow_unused=False
        )[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        # return torch.mean((dydx_l2norm - 1) ** 2)
        return torch.mean((1-dydx_l2norm) ** 2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def classification_loss(self, logit, target):
        """Compute binary or softmax cross entropy loss."""
        return F.cross_entropy(logit, target)
    
    def StoE2(self, x):
        """from Simplex to Embedding"""
        x_total = torch.reshape(x, (x.size(0), 26, CHARACTERS_NBRS))
        emb = self.G.main[0].embed
        concat = self.G.main[0].concat
        #emb = self.G.main.module[0].embed

        xs = torch.empty(x.size(0), 256, CHARACTERS_NBRS)

        for q in range(x.size(0)):  # for each batch
            xs[q] = torch.matmul(emb, x_total[q])  # 256 * 26 * 26 * 100

        concat_batch = torch.empty(x.size(0), 256, CHARACTERS_NBRS)
        concat_batch = concat_batch.to(self.device)
        for i in range(x.size(0)):
            concat_batch[i] = concat

        xs = xs.to(self.device, dtype=torch.float)

        xs = torch.cat((xs, concat_batch), dim=1)

        return xs   # (batch, 256, 100)

    def Simplex(self, x):   # x : (batch_size, 1, sample * vocab_size)
        """Initialize to Simplex"""
        """continuous relaxation"""
        ############ one-hot simplex for prof partition ############
        arr = [0.,     0.0392, 0.0784, 0.1176, 0.1569, 0.1961,
               0.2353, 0.2745, 0.3137, 0.3529, 0.3922, 0.4314,
               0.4706, 0.5098, 0.5490, 0.5882, 0.6275, 0.6667,
               0.7059, 0.7451, 0.7843, 0.8235, 0.8627, 0.9020,
               0.9412, 1.0000]

        #x_total = []

        x_total = torch.empty(0, dtype=torch.float)
        for q in range(x.size(0)):  # x.size(0) = batchsize
            #simplex = []
            simplex = torch.empty(0, dtype=torch.float)
            for i in range(x.size(2)):
                tmp = [0 for k in range(len(arr))]
                for j in range(len(arr)):
                    if round(float(x[q][0][i]), 4) == round(float(arr[j]), 4):
                        tmp[j] = 1
                        # simplex.append(tmp)
                        tmp = torch.from_numpy(np.array(tmp, dtype=np.float32))
                        simplex = torch.cat((simplex, tmp))
                        break

            simplex = torch.from_numpy(np.array(simplex, dtype=np.float32))
            x_total = torch.cat((x_total, simplex))

        x_total = torch.reshape(x_total, (x.size(0), CHARACTERS_NBRS, 26))
        x_total = torch.transpose(x_total, 1, 2)
        x_total = x_total.to(self.device, dtype=torch.float)

        return x_total  # (batch, 26, 100)

    def myonehot(self, x):
        """onehot vector for x which is thde output from self.G"""
        onehot = torch.zeros_like(x)

        for q in range(onehot.size(0)):
            tt = torch.argmax(x[q], dim=0)
            for e in range(onehot.size(2)):
                onehot[q][tt[e]][e] = 1.
        onehot = onehot.to(self.device, dtype=torch.float)

        return onehot

    def train(self):
        """Train model within a single dataset."""
        # Set data loader.
        data_loader = self.data_loader # load data

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)

        # Start training from scratch or resume training.
        start_iters = 0

        accu_tmp = []   # for Lorenz
        accu_tmp2 = []  # for Enigma
        accu_tmp3 = []  # for TypeX

        # accu test from PT to CT
        accu_tmp01 = []  # Lorenz
        accu_tmp02 = []  # Enigma
        accu_tmp03 = []  # TypeX

        # accu test for each cipher emulation
        accu_CtoV = []
        accu_CtoS = []
        accu_VtoC = []
        accu_VtoS = []
        accu_StoC = []
        accu_StoV = []

        # Start training.
        print("Start training...")
        # start_time = time.time()
        for i in range(start_iters, self.num_iters, 10000):

            '''warmup_constant in lr_schemes.py'''

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, label_org = next(data_iter)

            # Compute loss with real images.

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            c_org = self.label2onehot(label_org, self.c_dim)
            c_trg = self.label2onehot(label_trg, self.c_dim)
            

            x_real = torch.reshape(
                x_real, (x_real.size(0), x_real.size(1), CHARACTERS_NBRS))

            # x_real = x_real.to(self.device)  # Input images.
            # Original domain labels.
            c_org = c_org.to(self.device, dtype=torch.float)
            # Target domain labels.
            c_trg = c_trg.to(self.device, dtype=torch.float)
            label_org = label_org.to(
                self.device
            )  # Labels for computing classification loss.
            label_trg = label_trg.to(
                self.device
            )  # Labels for computing classification loss.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # USING Simplex function
            ######################################
            x_groundtruth = self.Simplex(x_real)  # batch,  26, 100
            # print("x_groundtruth = ", x_groundtruth)
            x_real_tmp = self.StoE2(x_groundtruth)  # batch, 256, 100

            # embedding line : (100, 26) * (26, 256) = (100, 256)
            out_src, out_cls = self.D(x_real_tmp)

            d_loss_real = torch.mean((out_src-1)**2)
            d_loss_cls = self.classification_loss(out_cls, label_org)

            # x_groundtruth = x_groundtruth.to(self.device) # why convert to cuda if already cuda?
            x_real_tmp = x_real_tmp.to(self.device)

            # Compute loss with fake images.
            # print("tout va bien mais pas ap l'appel de G -> cf class Generator & Discriminator: `c_dim doit être = 4`!")
            x_fake = self.G(x_groundtruth, c_trg)
            x_fake = self.StoE2(x_fake)

            out_src, out_cls = self.D(x_fake.detach())

            d_loss_fake = torch.mean((out_src)**2)

            # Compute loss for gradient penalty.
            # modified from (x_real.size(0), 1, 1, 1) to (x_real.size(0), 1, 1)
            alpha = torch.rand(x_groundtruth.size(0), 1, 1).to(
                self.device, dtype=torch.float)
            x_hat = (alpha * x_real_tmp.data + (1 - alpha) * x_fake.data).requires_grad_(
                True
            )

            out_src, _ = self.D(x_hat)

            out_src = torch.reshape(
                out_src, (out_src.size(0), out_src.size(2)))

            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = (
                0.5 * (d_loss_real			# E[D(x)] (should be minus)
                       + d_loss_fake)			# E[D(G, (x, c))] (should be plus)
                + self.lambda_cls * d_loss_cls  # L^r_cls
                + self.lambda_gp * d_loss_gp  # Wasserstein penalty
            )

            self.d_optimizer.zero_grad()

            d_loss.backward(retain_graph=True)
            self.d_optimizer.step()

            loss_d = d_loss.item()


            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            x_cipher_fake = self.G(x_groundtruth, c_trg)
            x_cipher_rfake = self.StoE2(x_cipher_fake)
            out_src, out_cls = self.D(x_cipher_rfake)
            g1_loss_fake = torch.mean((out_src-1)**2)
            g_loss_cls = self.classification_loss(out_cls, label_trg)

            x_cipher_fake = self.myonehot(x_cipher_fake)

            x_reconst = self.G(x_cipher_fake, c_org)
            x_reconst = self.myonehot(x_reconst)

            g_loss_rec = torch.mean(torch.abs(x_cipher_fake - x_reconst))

            g_loss = (
                g1_loss_fake
                + self.lambda_rec * g_loss_rec
                + self.lambda_cls * g_loss_cls
            )

            self.g_optimizer.zero_grad()
            g_loss.backward(retain_graph=True)
            self.g_optimizer.step()

            loss_g = g_loss.item()

            print("Epoch", i, "D loss:", loss_d, "G_loss:", loss_g)

            with torch.no_grad():

                #### for testline ##############
                data_loader_test = self.data_loader_test

                data_iter_test = iter(data_loader_test)
                x_fixed_test, c_org_test = next(data_iter_test)

                iden = 0
                while (iden == 0):

                    id0 = []
                    for idxstest0 in range(c_org_test.size(0)):
                        if (c_org_test[idxstest0] == 0.):
                            id0.append(idxstest0)

                    id1 = []
                    for idxstest in range(c_org_test.size(0)):
                        if (c_org_test[idxstest] == 1.):
                            id1.append(idxstest)
                        continue

                    id2 = []
                    for idxstest2 in range(c_org_test.size(0)):
                        if (c_org_test[idxstest2] == 2.):
                            id2.append(idxstest2)
                        continue

                    id3 = []
                    for idxstest3 in range(c_org_test.size(0)):
                        if (c_org_test[idxstest3] == 3.):
                            id3.append(idxstest3)
                        continue

                    if len(id1) == 0 or len(id2) == 0 or len(id3) == 0:
                        iden == 0 #fixme: `=` instead of `==`
                        print("Fix batch again")
                        data_loader_test = self.data_loader_test

                        data_iter_test = iter(data_loader_test)
                        x_fixed_test, c_org_test = next(data_iter_test)
                        print("Fix batch again done")
                    else:
                        iden = 1

                x_fixed_test = torch.reshape(
                    x_fixed_test, (x_fixed_test.size(0), x_fixed_test.size(1), CHARACTERS_NBRS))

            c_fixed_list_test = self.create_labels(c_org_test, self.c_dim)

            # using simplex fct.
            x_fixed_total_test = self.Simplex(x_fixed_test)

            ###########################################################################
            ############################# cipher to plain #############################
            ###########################################################################
            x_fixed_fake_test = self.G(
                x_fixed_total_test, c_fixed_list_test[0])
            x_fixed_fake_test = self.myonehot(x_fixed_fake_test)

            e = 0
            accu = 0
            while e < len(id1):
                list4 = ''  # for idx(Lorenz CT)
                list5 = ''  # for recovered PT from idx(Lorenz CT)
                
                # Convert the one-hot encoded tensor to a string:
                for q in range(CHARACTERS_NBRS):
                    for w in range(26):
                        if (x_fixed_total_test[id1[e]][w][q].item() == 1.):
                            list4 += (chr(97+w))
                        if (x_fixed_fake_test[id1[e]][w][q].item() == 1.):
                            list5 += (chr(97+w))

                last = lorenz_decrypt(list4)  # for recovered Lorenz CT

                cnt = 0 #counter of how many characters in last differ from the corresponding characters in list5
                for q in range(len(list4)):
                    if (last[q] != list5[q]):
                        cnt += 1
                    else:
                        continue
                accu += float((len(list4) - cnt) / len(list4))
                e += 1
            accu = accu / len(id1)
            accu_tmp.append(accu)
            # Lorenz decryption done

            e = 0
            accu2 = 0
            while e < len(id2):
                list6 = ''  # for idx2(Enigma CT)
                list7 = ''  # for recovered PT from idx2(Enigma CT)

                for q in range(CHARACTERS_NBRS):
                    for w in range(26):
                        if (x_fixed_total_test[id2[e]][w][q].item() == 1.):
                            list6 += (chr(97+w))
                        if (x_fixed_fake_test[id2[e]][w][q].item() == 1.):
                            list7 += (chr(97+w))

                # Decrypt line
                last2 = enigma_encrypt_or_decrypt(list6) # for recovered Enigma

                cnt = 0
                for q in range(len(list6)):
                    if (last2[q] != list7[q]):
                        cnt += 1
                    else:
                        continue
                accu2 += float((len(list6) - cnt) / len(list6))
                e += 1
            accu2 = accu2 / len(id2)
            accu_tmp2.append(accu2)

            e = 0
            accu3 = 0
            while e < len(id3):
                list8 = ''  # for TypeX
                list9 = ''  # for recovered PT from list8

                for q in range(CHARACTERS_NBRS):
                    for w in range(26):
                        if (x_fixed_total_test[id3[e]][w][q].item() == 1.):
                            list8 += (chr(97+w))
                        if (x_fixed_fake_test[id3[e]][w][q].item() == 1.):
                            list9 += (chr(97+w))

                # Decrypt line
                last3 = typex_encrypt_or_decrypt(list8)    # for recovered TypeX

                cnt = 0
                for q in range(len(list8)):
                    if (last3[q] != list9[q]):
                        cnt += 1
                    else:
                        continue
                accu3 += float((len(list7) - cnt) / len(list8))
                e += 1
            accu3 = accu3 / len(id3)
            accu_tmp3.append(accu3)

            print("Lorenz to plain : ", accu)
            print("Enigma to plain : ", accu2)
            print("TypeX to plain : ", accu3)

            ###########################################################################
            ############################# plain to cipher #############################
            ###########################################################################

            # Target domain is (tt+1) (1 : Lorenz, 2 : Enigma, 3 : TypeX)
            for tt in range(3):
                x_fixed_fake_test = self.G(
                    x_fixed_total_test, c_fixed_list_test[tt+1])
                x_fixed_fake_test = self.myonehot(x_fixed_fake_test)

                iden = 0
                while (iden == 0):
                    ids = []
                    for idxstest in range(c_org_test.size(0)):
                        if (c_org_test[idxstest] == float(tt+1)):
                            ids.append(idxstest)
                        continue

                    if len(ids) == 0:
                        iden = 0
                        print("Fix batch again")
                        data_loader_test = self.data_loader_test

                        data_iter_test = self.data_loader_test
                        x_fixed_test, c_org_test = next(data_iter_test)
                        print("Fix batch again done")
                    else:
                        iden = 1


                e = 0
                accu1 = 0  # -> Lorenz
                accu2 = 0  # -> Enigma
                accu3 = 0  # -> TypeX
                while e < len(id0):
                    list4 = ''  # for plain
                    list5 = ''  # for target domain tt+1

                    for q in range(CHARACTERS_NBRS):
                        for w in range(26):
                            if (x_fixed_total_test[id0[e]][w][q].item() == 1.):
                                list4 += (chr(97+w))
                            if (x_fixed_fake_test[id0[e]][w][q].item() == 1.):
                                list5 += (chr(97+w))

                    # encrypt line
                    last = ''   # for recovered Lorenz

                    if (tt+1) == 1:  # Lorenz
                        last = lorenz_encrypt(list4)
                        #Lorenz done

                        cnt = 0
                        for q in range(len(list4)):
                            if (last[q] != list5[q]):
                                cnt += 1
                            else:
                                continue
                        accu1 += float((len(list4) - cnt) / len(list4))
                        e += 1

                    elif (tt+1) == 2:  # Enigma
                        last = enigma_encrypt_or_decrypt(list4)
                        # Enigma done

                        cnt = 0
                        for q in range(len(list4)):
                            if (last[q] != list5[q]):
                                cnt += 1
                            else:
                                continue
                        accu2 += float((len(list4) - cnt) / len(list4))
                        e += 1

                    else:  # TypeX
                        last = typex_encrypt_or_decrypt(list4)
                        # TypeX done

                        cnt = 0
                        for q in range(len(list5)):
                            if (last[q] != list5[q]):
                                cnt += 1
                            else:
                                continue
                        accu3 += float((len(list4) - cnt) / len(list4))
                        e += 1


                if (tt+1) == 1:
                    accu1 = accu1 / len(id0)
                    accu_tmp01.append(accu1)
                    print("plain to Lorenz : ", accu1)
                elif (tt+1) == 2:
                    accu2 = accu2 / len(id0)
                    accu_tmp02.append(accu2)
                    print("plain to Enigma : ", accu2)
                else:
                    accu3 = accu3 / len(id0)
                    accu_tmp03.append(accu3)
                    print("plain to Lorenz : ", accu3)


            ########################################################################
            ########################### from CT to CT ##############################
            ########################################################################

            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            # Target domain is (0 : plain, 2 : Enigma, 3 : TypeX)
            for tt in range(4):
                if tt == 1:
                    continue

                x_fixed_fake_test = self.G(
                    x_fixed_total_test, c_fixed_list_test[tt])
                x_fixed_fake_test = self.myonehot(x_fixed_fake_test)

                iden = 0
                while (iden == 0):
                    ids = []
                    for idxstest in range(c_org_test.size(0)):
                        if (c_org_test[idxstest] == float(tt)):
                            ids.append(idxstest)
                        continue

                    if len(ids) == 0:
                        iden = 0
                        print("Fix batch again")
                        data_loader_test = self.data_loader_test

                        data_iter_test = self.data_loader_test
                        x_fixed_test, c_org_test = next(data_iter_test)
                        print("Fix batch again done")
                    else:
                        iden = 1


                e = 0
                accu1 = 0  # -> 0
                accu2 = 0  # -> 2
                accu3 = 0  # -> 3
                while e < len(id1):
                    list4 = ''  # for plain
                    list5 = ''  # for target domain tt+1

                    for q in range(CHARACTERS_NBRS):
                        for w in range(26):
                            if (x_fixed_total_test[id1[e]][w][q].item() == 1.):
                                list4 += (chr(97+w))
                            if (x_fixed_fake_test[id1[e]][w][q].item() == 1.):
                                list5 += (chr(97+w))

                    # encrypt line
                    last = ''   # for recovered Lorenz

                    #decrypt to PT using Lorenz
                    list4 = lorenz_decrypt(list4)

                    if tt == 1 or tt == 0:  # Lorenz or plain
                        e += 1
                        continue

                    elif (tt) == 2:  # Enigma
                        # encrypt to Enigma
                        last = enigma_encrypt_or_decrypt(list4)
                        #Enigma done

                        cnt = 0
                        for q in range(len(list4)):
                            if (last[q] != list5[q]):
                                cnt += 1
                            else:
                                continue
                        accu2 += float((len(list4) - cnt) / len(list4))
                        e += 1

                    else:  # TypeX 
                        compare_list4 = ''
                        for q in range(len(list4)):
                            compare_list4 += list4[q]
                        last = typex_encrypt_or_decrypt(compare_list4)
                        # TypeX done

                        cnt = 0
                        for q in range(len(list5)):
                            if (last[q] != list5[q]):
                                cnt += 1
                            else:
                                continue
                        accu3 += float((len(list4) - cnt) / len(list4))
                        e += 1

                if tt == 1 or tt == 0:
                    continue
                elif tt == 2:
                    accu2 = accu2 / len(id1)
                    accu_CtoV.append(accu2)
                    print("Lorenz to Enigma : ", accu2)
                else:
                    accu3 = accu3 / len(id1)
                    accu_CtoS.append(accu3)
                    print("Lorenz to TypeX : ", accu3)

            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            # Target domain is (0 : plain, 1 : Lorenz, 3 : TypeX)
            for tt in range(4):
                if tt == 2:
                    continue

                x_fixed_fake_test = self.G(
                    x_fixed_total_test, c_fixed_list_test[tt])
                x_fixed_fake_test = self.myonehot(x_fixed_fake_test)

                iden = 0
                while (iden == 0):
                    ids = []
                    for idxstest in range(c_org_test.size(0)):
                        if (c_org_test[idxstest] == float(tt)):
                            ids.append(idxstest)
                        continue

                    if len(ids) == 0:
                        iden = 0
                        print("Fix batch again")
                        data_loader_test = self.data_loader_test

                        data_iter_test = self.data_loader_test
                        x_fixed_test, c_org_test = next(data_iter_test)
                        print("Fix batch again done")
                    else:
                        iden = 1


                e = 0
                accu1 = 0  # -> 0
                accu2 = 0  # -> 1
                accu3 = 0  # -> 3
                while e < len(id2):
                    list4 = ''  # for plain
                    list5 = ''  # for target domain tt+1

                    for q in range(CHARACTERS_NBRS):
                        for w in range(26):
                            if (x_fixed_total_test[id2[e]][w][q].item() == 1.):
                                list4 += (chr(97+w))
                            if (x_fixed_fake_test[id2[e]][w][q].item() == 1.):
                                list5 += (chr(97+w))

                    # encrypt line
                    last = ''   # for recovered Enigma

                    # decrypt Enigma CT to plain
                    list4 = enigma_encrypt_or_decrypt(list4)

                    if tt == 2 or tt == 0:  # Enigma or plain
                        e += 1
                        continue

                    elif (tt) == 1:  # Lorenz
                        # encrypt to Lorenz
                        last = lorenz_encrypt(list4)
                        # encrypt to Lorenz done

                        cnt = 0
                        for q in range(len(list4)):
                            if (last[q] != list5[q]):
                                cnt += 1
                            else:
                                continue
                        accu2 += float((len(list4) - cnt) / len(list4))
                        e += 1

                    else:  # TypeX
                        compare_list4 = ''
                        for q in range(len(list4)):
                            compare_list4 += list4[q]
                        # encrypt to TypeX
                        last = typex_encrypt_or_decrypt(compare_list4)
                        # encrypt to TypeX done

                        cnt = 0
                        for q in range(len(list5)):
                            if (last[q] != list5[q]):
                                cnt += 1
                            else:
                                continue
                        accu3 += float((len(list4) - cnt) / len(list4))
                        e += 1
                if tt == 2 or tt == 0:
                    continue
                elif tt == 1:
                    accu2 = accu2 / len(id2)
                    accu_VtoC.append(accu2)
                    print("Enigma to Lorenz : ", accu2)
                else:
                    accu3 = accu3 / len(id2)
                    accu_VtoS.append(accu3)
                    print("Enigma to TypeX : ", accu3)

            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            # Target domain is (0 : plain, 1 : Lorenz, 2 : Enigma)
            for tt in range(4):
                if tt == 3:
                    continue

                x_fixed_fake_test = self.G(
                    x_fixed_total_test, c_fixed_list_test[tt])
                x_fixed_fake_test = self.myonehot(x_fixed_fake_test)

                iden = 0
                while (iden == 0):
                    ids = []
                    for idxstest in range(c_org_test.size(0)):
                        if (c_org_test[idxstest] == float(tt)):
                            ids.append(idxstest)
                        continue

                    if len(ids) == 0:
                        iden = 0
                        print("Fix batch again")
                        data_loader_test = self.data_loader_test

                        data_iter_test = self.data_loader_test
                        x_fixed_test, c_org_test = next(data_iter_test)
                        print("Fix batch again done")
                    else:
                        iden = 1


                e = 0
                accu1 = 0  # -> 0
                accu2 = 0  # -> 1
                accu3 = 0  # -> 2
                while e < len(id3):
                    list4 = ''  # for plain
                    list5 = ''  # for target domain tt+1

                    for q in range(CHARACTERS_NBRS):
                        for w in range(26):
                            if (x_fixed_total_test[id3[e]][w][q].item() == 1.):
                                list4 += (chr(97+w))
                            if (x_fixed_fake_test[id3[e]][w][q].item() == 1.):
                                list5 += (chr(97+w))

                    # encrypt line
                    last = ''   # for recovered TypeX

                    # decrypt TypeX CT to plain
                    list4 = typex_encrypt_or_decrypt(list4)
                    # decrypt TypeX CT to plain done

                    if tt == 3 or tt == 0:  # TypeX or plain
                        e += 1
                        continue

                    elif (tt) == 1:  # Lorenz
                        last = lorenz_encrypt(list4)

                        cnt = 0
                        for q in range(len(list4)):
                            if (last[q] != list5[q]):
                                cnt += 1
                            else:
                                continue
                        accu2 += float((len(list4) - cnt) / len(list4))
                        e += 1

                    else:  # Enigma
                        last = enigma_encrypt_or_decrypt(list4)

                        cnt = 0
                        for q in range(len(list5)):
                            if (last[q] != list5[q]):
                                cnt += 1
                            else:
                                continue
                        accu3 += float((len(list4) - cnt) / len(list4))
                        e += 1

                if tt == 3 or tt == 0:
                    continue

                elif tt == 1:
                    accu2 = accu2 / len(id3)
                    accu_StoC.append(accu2)
                    print("TypeX to Lorenz : ", accu2)
                else:
                    accu3 = accu3 / len(id3)
                    accu_StoV.append(accu3)
                    print("TypeX to Enigma : ", accu3)

            os.makedirs("accumodels/", exist_ok=True)
            AC_path = os.path.join(
                "accumodels/", "{}-ACCU-AC.ckpt".format(i))
            torch.save({'accu_tmp01': accu_tmp01, 'accu_tmp02': accu_tmp02, 'accu_tmp03': accu_tmp03,
                        'accu_CtoV': accu_CtoV, 'accu_CtoS': accu_CtoS, 'accu_VtoC': accu_VtoC,
                        'accu_VtoS': accu_VtoS, 'accu_StoC': accu_StoC, 'accu_StoV': accu_StoV}, AC_path)