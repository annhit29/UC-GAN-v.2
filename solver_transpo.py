# %%writefile /kaggle/working/UC-GAN-v.2/lib/solver_transpo.py
from model import Generator
from model import Discriminator
import torch
import torch.nn.functional as F
import numpy as np
import copy
import os

#import transposition ciphers' modules:
import lib.transpo_rail_fence as rf
import lib.columnar as clnar
import lib.transposition as transpo


RAIL_FENCE_KEY = 4
COLUMNAR_KEY = "hackhack" #in anycase will = "hack" coz cf columnar.py  #order “3 1 2 4”
TRANSPO_KEY = 8

class Solver_Transpo(object):
    """Solver for training and testing UC-GAN-v.2"""

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
        # print("logit=", logit)
        # print("target = ", target)
        return F.cross_entropy(logit, target) #`torch.nn.functional.cross_entropy(input, target)` can receive a multi-class `target` (classes 0, 1, 2 and 3)

    def StoE2(self, x):
        """from Simplex to Embedding"""
        x_total = torch.reshape(x, (x.size(0), 26, 100))
        emb = self.G.main[0].embed
        concat = self.G.main[0].concat
        #emb = self.G.main.module[0].embed

        xs = torch.empty(x.size(0), 256, 100)

        for q in range(x.size(0)):  # for each batch
            xs[q] = torch.matmul(emb, x_total[q])  # 256 * 26 * 26 * 100

        concat_batch = torch.empty(x.size(0), 256, 100)
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
                    # print("round(float(x[q][0][i]), 2) = ", round(float(x[q][0][i]), 2))
                    # print("round(float(arr[j]), 2) = ", round(float(arr[j]), 2))
                    if round(float(x[q][0][i]), 2) - 0.02 <= round(float(arr[j]), 2) <= round(float(x[q][0][i]), 2) + 0.02: # round(float(x[q][0][i]), 2) +/-0.02 <- due to my implementation for generating BMPs in `txt2bmp/py`
                    # if round(float(x[q][0][i]), 4) == round(float(arr[j]), 4):
                        tmp[j] = 1
                        # simplex.append(tmp)
                        tmp = torch.from_numpy(np.array(tmp, dtype=np.float32))
                        simplex = torch.cat((simplex, tmp))
                        break

            simplex = torch.from_numpy(np.array(simplex, dtype=np.float32))
            x_total = torch.cat((x_total, simplex))
        
        # print("x.size(0) = ", x.size(0))
        x_total = torch.reshape(x_total, (x.size(0), 100, 26))
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
        data_loader = self.data_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)

        # Start training from scratch or resume training.
        start_iters = 0

        accu_tmp = []   # for rail_fence
        accu_tmp2 = []  # for columnar
        accu_tmp3 = []  # for transpo

        # accu test from PT to CT
        accu_tmp01 = []  # rail_fence
        accu_tmp02 = []  # columnar
        accu_tmp03 = []  # transpo

        # accu test for each cipher emulation
        accu_CtoV = []
        accu_CtoS = []
        accu_VtoC = []
        accu_VtoS = []
        accu_StoC = []
        accu_StoV = []

        # Start training.
        print("Start training...")
        print("outside for loop")
        # start_time = time.time()
        for i in range(start_iters, self.num_iters, 10000):
            # print("iiiiiiiiiii")

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
                x_real, (x_real.size(0), x_real.size(1), 100))

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
            print("coucou")
            print("x_groundtruth = ", x_groundtruth)
            x_real_tmp = self.StoE2(x_groundtruth)  # batch, 256, 100

            # embedding line : (100, 26) * (26, 256) = (100, 256)
            out_src, out_cls = self.D(x_real_tmp)

            d_loss_real = torch.mean((out_src-1)**2)
            d_loss_cls = self.classification_loss(out_cls, label_org)

            # print("still OK")
            # x_groundtruth = x_groundtruth.to(self.device) # todo: why convert to cuda if already cuda?
            # print("x_groundtruth.to = ", x_groundtruth)
            x_real_tmp = x_real_tmp.to(self.device)

            # Compute loss with fake images.
            # print("tout va bien mais pas ap l'appel de G -> cf class Generator & Discriminator: `c_dim doit être = 4`!")
            x_fake = self.G(x_groundtruth, c_trg) #error comes from label index outofrange,so c_trg
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

            # print(i)

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
                        iden = 0 #fixme: `=` instead of `==`
                        print("Fix batch again")
                        data_loader_test = self.data_loader_test

                        data_iter_test = iter(data_loader_test)
                        x_fixed_test, c_org_test = next(data_iter_test)
                        print("Fix batch again done")
                    else:
                        iden = 1

                x_fixed_test = torch.reshape(
                    x_fixed_test, (x_fixed_test.size(0), x_fixed_test.size(1), 100))

            c_fixed_list_test = self.create_labels(c_org_test, self.c_dim)

            # using simplex fct.
            x_fixed_total_test = self.Simplex(x_fixed_test) # the original text before any decryption

            ###########################################################################
            ############################# cipher to plain #############################
            ###########################################################################
            x_fixed_fake_test = self.G(
                x_fixed_total_test, c_fixed_list_test[0])
            x_fixed_fake_test = self.myonehot(x_fixed_fake_test) # an one-hot encoded tensor


            e = 0
            accu = 0
            while e < len(id1):
                list4 = ''  # for idx(rail-fence) <- store characters from the "real" ciphertext x_fixed_total_test
                list5 = ''  # for recovered PT from idx(rail-fence) <- store characters from the generated plaintext (so fake plaintext) 
                
                # Convert the one-hot encoded tensor to a string:
                for q in range(100):
                    for w in range(26):
                        if (x_fixed_total_test[id1[e]][w][q].item() == 1.):
                            list4 += (chr(97+w))
                        if (x_fixed_fake_test[id1[e]][w][q].item() == 1.):
                            list5 += (chr(97+w))
                

                # Decrypt line using rail-fence cipher:
                last = str(rf.decryptRailFence(list4, RAIL_FENCE_KEY)) # the plaintext obtained after decryption list4

                #hamming weight idea:
                cnt = 0 #counter of how many characters in `last` (real PT) differ from the corresponding characters in `list5` (fake PT)
                for q in range(len(list4)):
                    if (last[q] != list5[q]):
                        cnt += 1
                    else:
                        continue
                accu += float((len(list4) - cnt) / len(list4))
                #accuracy computed for rf cipher done

                e += 1
            accu = accu / len(id1)
            accu_tmp.append(accu)
            # Rf decryption done

            e = 0
            accu2 = 0
            while e < len(id2):
                list6 = ''  # for idx(columnar) <- store characters from the "real" ciphertext x_fixed_total_test
                list7 = ''  # for recovered PT from idx(columnar) <- store characters from the generated plaintext (so fake plaintext) 

                # Convert the one-hot encoded tensor to a string:
                for q in range(100):
                    for w in range(26):
                        if (x_fixed_total_test[id2[e]][w][q].item() == 1.):
                            list6 += (chr(97+w)) # Convert 'w' to an ASCII character using ofset=97 (∵ 97 in ASCII = 'a') then append to list4
                        if (x_fixed_fake_test[id2[e]][w][q].item() == 1.):
                            list7 += (chr(97+w))

                # Decrypt line using columnar cipher:
                last2 = clnar.row_decrypt(list6, COLUMNAR_KEY) # the plaintext obtained after decryption list6

                cnt = 0
                for q in range(len(list6)):
                    if (last2[q] != list7[q]):
                        #pk 16 au lieu de 100? fixed: coz the original code had `for j in range(len(k)):` instead of `for j in range(len(arr)):` that caused `IndexError: list index out of range` in `arr[j][h]`
                        # Dou vient la data? from the provided src. 
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
                list8 = ''  # for idx(transpo) <- store characters from the "real" ciphertext x_fixed_total_test
                list9 = ''  # for recovered PT from idx(transpo) <- store characters from the generated plaintext (so fake plaintext)
                
                # Convert the one-hot encoded tensor to a string:
                for q in range(100):
                    for w in range(26):
                        if (x_fixed_total_test[id3[e]][w][q].item() == 1.):
                            list8 += (chr(97+w))
                        if (x_fixed_fake_test[id3[e]][w][q].item() == 1.):
                            list9 += (chr(97+w))

                # Decrypt line using transposition cipher:
                last3 = transpo.decryptMessage(TRANSPO_KEY, list8) # the plaintext obtained after decryption list6

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

            print("Rail-fence to plain : ", accu)
            print("Columnar to plain : ", accu2)
            print("Transpo to plain : ", accu3)

            ###########################################################################
            ############################# plain to cipher #############################
            ###########################################################################

            # Target domain is (tt+1) (1 : Rail-fence, 2 : Columnar, 3 : Transpo)
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

                # print(len(ids))

                e = 0
                accu1 = 0  # -> Rail-fence
                accu2 = 0  # -> Columnar
                accu3 = 0  # -> Transpo
                while e < len(id0):
                    list4 = ''  # for plaintext
                    list5 = ''  # for target domain tt+1

                    for q in range(100):
                        for w in range(26):
                            if (x_fixed_total_test[id0[e]][w][q].item() == 1.):
                                list4 += (chr(97+w))
                            if (x_fixed_fake_test[id0[e]][w][q].item() == 1.):
                                list5 += (chr(97+w))

                    # encrypt line
                    last = ''   # for recovered rf

                    if (tt+1) == 1:  # rf
                        last = str(rf.encryptRailFence(list4, RAIL_FENCE_KEY))

                        cnt = 0
                        for q in range(len(list4)):
                            if (last[q] != list5[q]):
                                cnt += 1
                            else:
                                continue
                        accu1 += float((len(list4) - cnt) / len(list4))
                        e += 1

                    elif (tt+1) == 2:  # clnar
                        last = clnar.row_encrypt(list4, COLUMNAR_KEY)

                        cnt = 0
                        for q in range(len(list4)):
                            if (last[q] != list5[q]):
                                cnt += 1
                            else:
                                continue
                        accu2 += float((len(list4) - cnt) / len(list4))
                        e += 1

                    else:  # Transpo
                        last = transpo.encryptMessage(TRANSPO_KEY, list4)

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
                    print("plain to Rail-fence : ", accu1)
                elif (tt+1) == 2:
                    accu2 = accu2 / len(id0)
                    accu_tmp02.append(accu2)
                    print("plain to Columnar : ", accu2)
                else:
                    accu3 = accu3 / len(id0)
                    accu_tmp03.append(accu3)
                    print("plain to Transposition : ", accu3)


            ########################################################################
            ########################### from CT to CT ##############################
            ########################################################################

            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            # Target domain is (0 : plaintext, 2 : Columnar, 3 : Transpo)
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
                accu1 = 0  # -> 0: Plaintext
                accu2 = 0  # -> 2: Columnar
                accu3 = 0  # -> 3: Transpo
                while e < len(id1):
                    list4 = ''  # for plaintext
                    list5 = ''  # for target domain tt+1

                    for q in range(100):
                        for w in range(26):
                            if (x_fixed_total_test[id1[e]][w][q].item() == 1.):
                                list4 += (chr(97+w))
                            if (x_fixed_fake_test[id1[e]][w][q].item() == 1.):
                                list5 += (chr(97+w))

                    # encrypt line
                    last = ''   # for recovered Rail-fence

                    # decrypt to plain using Rail-fence:
                    list4 = str(rf.decryptRailFence(list4, RAIL_FENCE_KEY))

                    if tt == 1 or tt == 0:  # Rail-fence or Plaintext
                        e += 1
                        continue

                    elif (tt) == 2:  # Columnar
                        # encrypt to Columnar
                        last = clnar.row_encrypt(list4, COLUMNAR_KEY)
                        #Columnar done
                        
                        cnt = 0
                        for q in range(len(list4)):
                            if (last[q] != list5[q]):
                                cnt += 1
                            else:
                                continue
                        accu2 += float((len(list4) - cnt) / len(list4))
                        e += 1

                    else:  # Transposition
                        compare_list4 = ''
                        for q in range(len(list4)):
                            compare_list4 += list4[q]
                        last = transpo.encryptMessage(TRANSPO_KEY, compare_list4)

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
                    print("Rail-fence to Columnar : ", accu2)
                else:
                    accu3 = accu3 / len(id1)
                    accu_CtoS.append(accu3)
                    print("Rail-fence to Transposition : ", accu3)

            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            # Target domain is (0 : plain, 1 : Rail-fence, 3 : Transposition)
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
                accu2 = 0  # -> 1: rf
                accu3 = 0  # -> 3: Transposition
                while e < len(id2):
                    list4 = ''  # for plain
                    list5 = ''  # for target domain tt+1

                    for q in range(100):
                        for w in range(26):
                            if (x_fixed_total_test[id2[e]][w][q].item() == 1.):
                                list4 += (chr(97+w))
                            if (x_fixed_fake_test[id2[e]][w][q].item() == 1.):
                                list5 += (chr(97+w))

                    # encrypt line
                    last = ''   # for recovered Columnar

                    # decrypt to plain
                    list4 = clnar.row_decrypt(list4, COLUMNAR_KEY)                    

                    if tt == 2 or tt == 0:  # Columnar or plain
                        e += 1
                        continue

                    elif (tt) == 1:  # Rail-fence
                        # encrypt to Rail-fence
                        last = str(rf.encryptRailFence(list4, RAIL_FENCE_KEY))
                        #encrypt to rf done
                        
                        cnt = 0
                        for q in range(len(list4)):
                            if (last[q] != list5[q]):
                                cnt += 1
                            else:
                                continue
                        accu2 += float((len(list4) - cnt) / len(list4))
                        e += 1

                    else:  # Substitution
                        compare_list4 = ''
                        for q in range(len(list4)):
                            compare_list4 += list4[q]
                        last = transpo.encryptMessage(TRANSPO_KEY, compare_list4)

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
                    print("Columnar to Rail-fence : ", accu2)
                else:
                    accu3 = accu3 / len(id2)
                    accu_VtoS.append(accu3)
                    print("Columnar to Transposition : ", accu3)

            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            # Target domain is (0 : plain, 1 : Rail-fence, 2 : Columnar)
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
                accu1 = 0  # -> 0: Plaintext
                accu2 = 0  # -> 1: rf
                accu3 = 0  # -> 2: Columnar
                while e < len(id3):
                    list4 = ''  # for plain
                    list5 = ''  # for target domain tt+1

                    for q in range(100):
                        for w in range(26):
                            if (x_fixed_total_test[id3[e]][w][q].item() == 1.):
                                list4 += (chr(97+w))
                            if (x_fixed_fake_test[id3[e]][w][q].item() == 1.):
                                list5 += (chr(97+w))

                    # encrypt line
                    last = ''   # for recovered Transposition

                    # decrypt to plain
                    list4 = transpo.decryptMessage(TRANSPO_KEY, list4)

                    if tt == 3 or tt == 0:  # transposition or plain
                        e += 1
                        continue

                    elif (tt) == 1:  # Rf
                        last = str(rf.encryptRailFence(list4, RAIL_FENCE_KEY))                        
                        cnt = 0
                        for q in range(len(list4)):
                            if (last[q] != list5[q]):
                                cnt += 1
                            else:
                                continue
                        accu2 += float((len(list4) - cnt) / len(list4))
                        e += 1

                    else:  # Columnar
                        last = clnar.row_encrypt(list4, COLUMNAR_KEY)

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
                    print("Transpo to Rail-fence : ", accu2)
                else:
                    accu3 = accu3 / len(id3)
                    accu_StoV.append(accu3)
                    print("Transpo to Columnar : ", accu3)

            os.makedirs("accumodels/", exist_ok=True)
            AC_path = os.path.join(
                "accumodels/", "{}-ACCU-AC.ckpt".format(i))
            torch.save({'accu_tmp01': accu_tmp01, 'accu_tmp02': accu_tmp02, 'accu_tmp03': accu_tmp03,
                        'accu_CtoV': accu_CtoV, 'accu_CtoS': accu_CtoS, 'accu_VtoC': accu_VtoC,
                        'accu_VtoS': accu_VtoS, 'accu_StoC': accu_StoC, 'accu_StoV': accu_StoV}, AC_path)