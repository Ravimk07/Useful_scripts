import numpy as np
import torch
from torchvision import datasets, transforms
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import math

def cosineLR(epoch):
    initialLR = 1e-4
    decay_steps = 300
    alpha = 0
    epoch = min(epoch, decay_steps)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * epoch / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    return initialLR * decayed

class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_DataLoader: torch.utils.data.Dataset,
                 validation_DataLoader: torch.utils.data.Dataset = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 epochs: int = 100,
                 epoch: int = 0,
                 notebook: bool = False,
                 path2write: str= None
                 ):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.device = device
        self.epochs = epochs
        self.epoch = epoch
        self.notebook = notebook
        self.training_loss = []
        self.validation_loss = []
        self.learning_rate = []
        self.path2write= path2write
        self.writer = SummaryWriter()



    def run_trainer(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange
        self.model.to(self.device)
        #self.criterion.to(self.device)
        progressbar = trange(self.epochs, desc='Progress', disable=False)
        loss_max= 1000
        for i in progressbar:
            print('Epoch:', i)
            self.lr_scheduler.step(i)
            #self.optimizer.param_groups[0]['lr']= cosineLR(i)
            """Epoch counter"""
            self.epoch += 1  # epoch counter

            """Training block"""
            train_loss, train_accu= self._train()
            self.writer.add_scalar("Loss/train", train_loss, i)
            self.writer.add_scalar("Accuracy/train", train_accu, i)
            """Validation block"""
            val_loss, val_accu=self._validate()
            self.writer.add_scalar("Loss/val", val_loss, i)
            self.writer.add_scalar("Accuracy/val",val_accu, i)
            #self.writer.add_scalar("Learning_rate", self.learning_rate[i], i)
            self.writer.add_scalar("Learning_rate", self.optimizer.param_groups[0]['lr'], i)
            """Learning rate scheduler block"""
            '''if self.lr_scheduler is not None:
                if self.validation_DataLoader is not None and self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    self.lr_scheduler.batch(self.validation_loss[i])  # learning rate scheduler step with validation loss
                else:
                    self.lr_scheduler.batch()  # learning rate scheduler step'''


            if loss_max > val_loss:
                print('score2 ({:.6f} --> {:.6f}).  Saving model ...'.format(loss_max, val_loss))
                checkpoint_dict = {
                    'epoch': self.epoch,
                    'model': self.model,
                    'optimizer':  self.optimizer,
                    'scheduler_cosine': self.optimizer.state_dict()

                }
                save_model= "bestmodel" +str(i)+ ".pth"

                torch.save(self.model.state_dict(), os.path.join(self.path2write, save_model))
                #torch.save(checkpoint_dict, best_file_checkpoint)
                #save_checkpoint(checkpoint_dict_sai, checkpoint_sai, best_file_checkpoint_sai, is_best=True)
                loss_max = val_loss
            save_model = "epoch" + str(i) + ".pth"

            # torch.save(self.model.state_dict(), os.path.join(self.path2write, save_model))
            #save_checkpoint(checkpoint_dict_sai, checkpoint_sai, best_file_checkpoint_sai, is_best=False)
        return self.training_loss, self.validation_loss, self.learning_rate

    def _train(self):

        '''if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:'''
        from tqdm import tqdm, trange

        self.model.train()  # train mode
        train_losses = []  # accumulate the losses here
        correct= 0
        total=0
        batch_iter = tqdm(enumerate(self.training_DataLoader), 'Training', total=len(self.training_DataLoader),
                          disable= False)
        to_tensor = transforms.ToTensor()
        for i, (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)
            self.optimizer.zero_grad()  # zerograd the parameters
            out = self.model(input)  # one forward pass
            loss = self.criterion(out, torch.argmax(target, axis=1))  # calculate loss
            # loss = self.criterion(out, target)  # calculate loss

            loss_value = loss.item()

            train_losses.append(loss_value)
            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters

            _, predicted= out.max(1)
            _, label= target.max(1)
            total += (target.size(0)* target.size(2)* target.size(3))
            correct += predicted.eq(label).sum().item()
            batch_iter.set_description(f'Training: (loss {loss_value:.4f})')  # update progressbar


        self.training_loss.append(np.mean(train_losses))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

        batch_iter.close()
        accu = correct / total
        return np.mean(train_losses), accu

    def _validate(self):

        '''if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:'''
        from tqdm import tqdm, trange

        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses here
        batch_iter = tqdm(enumerate(self.validation_DataLoader), 'Validation', total=len(self.validation_DataLoader),
                          disable=False)
        total=0
        correct =0
        for i, (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)

            with torch.no_grad():
                out = self.model(input)
                loss = self.criterion(out, torch.argmax(target,axis=1))
                # loss = self.criterion(out, target)

                loss_value = loss.item()

                valid_losses.append(loss_value)
                _, predicted = out.max(1)
                _, label = target.max(1)
                total += (target.size(0) * target.size(2) * target.size(3))
                correct += predicted.eq(label).sum().item()
                batch_iter.set_description(f'Validation: (loss {loss_value:.4f})')

        self.validation_loss.append(np.mean(valid_losses))

        batch_iter.close()
        accu = correct / total
        return np.mean(valid_losses), accu