import datetime
import os

import torch
import torch.nn as nn

from torchvision.utils import save_image
from Tools.progress_bar import print_progress_bar


class PytorchGAN(nn.Module):
    def __init__(self, criterion='bce', logger=None, opt='adam', device='cpu', ckpt_save_path=None, tag=''):
        super().__init__()
        self.opt_type = opt if opt in ['sgd', 'adam'] else ValueError
        self.optG = None
        self.optD = None
        self.log = logger
        self.device = device
        self.ckpt_save_path = ckpt_save_path
        self.state = {}
        self.criterion = nn.MSELoss(reduction='mean') if criterion == 'mse' else nn.BCELoss(reduction='mean') if criterion == 'bce' else criterion

        self.best_criterion = {'loss' : 10**10,  'v_loss': 10**10, 'acc': -1, 'v_acc': -1}
        self.best_model = None
        self.best_epoch = None

        # /!\ the overriding class must implement a discriminator and a generator extending nn.Module
        self.generator_input_shape = None
        self.generator     = None
        self.discriminator = None

        # useful stuff that can be needed for during fit
        self.verbose  = None
        self.n_epochs = None
        self.n        = None
        self.tag      = tag

        self.save_images_freq = None

    def _train_epoch(self, dataloader):
        epoch_disc_real_loss  = 0
        epoch_disc_fake_loss  = 0
        epoch_disc_tot_loss   = 0
        epoch_gen_loss        = 0
        for idx, batch in enumerate(dataloader):
            batch_x, batch_y = batch
            batch_x = batch_x.to(self.device)
            batch_size = batch_x.size(0)

            # Compute the loss the for the discriminator with real images
            self.discriminator.zero_grad()
            label = torch.full(batch_size, 1, dtype=torch.float, device=self.device)
            real_disc_out = self.discriminator(batch_x).view(-1)
            disc_real_loss = self.criterion(real_disc_out, label)

            # Compute the loss the for the discriminator with fake images
            noise_shape = [batch_size] + list(self.generator_input_shape)
            noise = torch.randn(noise_shape, device=self.device)
            fake_images = self.generator(noise)
            label.fill_(-1)  # changing the label
            fake_disc_out = self.discriminator(fake_images.detach()).view(-1)  # we do not backprop on the generator
            disc_fake_loss = self.criterion(fake_disc_out, label)

            # Finally back prop on disc
            disc_real_loss.backward()
            disc_fake_loss.backward()
            disc_tot_loss = disc_real_loss + disc_fake_loss
            self.optD.step()

            # Training the generator
            self.generator.zero_grad()
            label.fill_(1)  # for the generator, all image are real as we construct them
            out = self.discriminator(fake_images).view(-1)  # this time we want to backprop on the generator
            gen_loss = self.criterion(out, label)
            gen_loss.backward()
            self.optG.step()

            # Update running losses
            epoch_disc_real_loss += disc_real_loss.item()
            epoch_disc_fake_loss += disc_fake_loss.item()
            epoch_disc_tot_loss  += disc_tot_loss.item()
            epoch_gen_loss       += gen_loss.item()
            if self.verbose == 1:
                print_progress_bar(idx, len(dataloader))

        return epoch_disc_real_loss / len(dataloader), epoch_disc_fake_loss / len(dataloader), epoch_disc_tot_loss / len(dataloader), epoch_gen_loss / len(dataloader),

    def _validate(self, dataloader):
        epoch_disc_real_loss = 0
        epoch_disc_fake_loss = 0
        epoch_disc_tot_loss = 0
        epoch_gen_loss = 0
        for idx, batch in enumerate(dataloader):
            batch_x, batch_y = batch
            batch_x = batch_x.to(self.device)
            batch_size = batch_x.size(0)

            # Compute the loss the for the discriminator with real images
            self.discriminator.zero_grad()
            label = torch.full(batch_size, 1, dtype=torch.float, device=self.device)
            real_disc_out = self.discriminator(batch_x).view(-1)
            disc_real_loss = self.criterion(real_disc_out, label)

            # Compute the loss the for the discriminator with fake images
            noise_shape = [batch_size] + list(self.generator_input_shape)
            noise = torch.randn(noise_shape, device=self.device)
            fake_images = self.generator(noise)
            label.fill_(-1)  # changing the label
            fake_disc_out = self.discriminator(fake_images.detach()).view(-1)  # we do not backprop on the generator
            disc_fake_loss = self.criterion(fake_disc_out, label)

            disc_tot_loss = disc_real_loss + disc_fake_loss

            # Training the generator
            gen_loss = disc_fake_loss

            # Update running losses
            epoch_disc_real_loss += disc_real_loss.item()
            epoch_disc_fake_loss += disc_fake_loss.item()
            epoch_disc_tot_loss += disc_tot_loss.item()
            epoch_gen_loss += gen_loss.item()
            if self.verbose == 1:
                print_progress_bar(idx, len(dataloader))

        return epoch_disc_real_loss / len(dataloader), epoch_disc_fake_loss / len(dataloader), epoch_disc_tot_loss / len(dataloader), epoch_gen_loss / len(dataloader),

    def fit(self, dataloader, n_epochs, lr, validation_data=None, verbose=1, save_images_freq=None, save_criterion='loss', ckpt=None, **kwargs):
        assert self.generator is not None, 'Model does not seem to have a generator, assign the generator to the self.generator attribute'
        assert self.discriminator is not None, 'Model does not seem to have a discriminator, assign the discriminator to the self.discriminator attribute'
        assert self.generator_input_shape is not None, 'Could not find the generator input shape, please specify this attribute before fitting the model'

        if self.opt_type == 'sgd':
            self.optG = torch.optim.SGD(params=self.generator.parameters(), lr=lr)
            self.optD = torch.optim.SGD(params=self.discriminator.parameters(), lr=lr)
        elif self.opt_type == 'adam':
            self.optG = torch.optim.Adam(params=self.generator.parameters(), lr=lr)
            self.optD = torch.optim.Adam(params=self.discriminator.parameters(), lr=lr)
        else:
            raise ValueError('Unknown optimizer')

        start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        start_epoch = 0
        self.verbose = verbose
        self.save_images_freq = save_images_freq

        if ckpt:
            state = torch.load(ckpt)
            start_epoch = state['epoch']
            self.load_state_dict(state['state_dict'])
            for g in self.optD.param_groups:
                g['lr'] = state['lr']
            for g in self.optG.param_groups:
                g['lr'] = state['lr']

        self.n_epochs = n_epochs
        for n in range(start_epoch, n_epochs):
            self.n = n
            train_loss, train_bce, train_kll = self._train_epoch(dataloader)
            val_loss, val_bce, val_kll = 0, 0, 0
            if validation_data is not None:
                with torch.no_grad():
                    val_loss, val_bce, val_kll = self._validate(validation_data)

            epoch_result = {'loss': train_loss, 'bce': train_bce, 'kll': train_kll, 'v_loss': val_loss, 'v_bce': val_bce, 'v_kll': val_kll}
            if self.log:
                self.log('Train loss', train_loss, n)
                self.log('Val loss', val_loss, n)
                # TODO logging

            if epoch_result[save_criterion] <= self.best_criterion[save_criterion]:
                # TODO saving res
                self.best_criterion = epoch_result
                self.__save_state(n)

            if n % verbose == 0:
                # TODO verbose
                print('Epoch {:3d} loss: {:1.4f} BCE loss: {:1.4f} KLL loss {:1.4f} | Validation loss: {:1.4f} BCE loss: {:1.4f} KLL loss {:1.4f} | Best epoch {:3d}'.format(n, train_loss, train_bce, train_kll, val_loss, val_bce, val_kll, self.best_epoch))

            if self.ckpt_save_path:
                self.state['lr'] = lr
                self.state['epoch'] = n
                self.state['state_dict'] = self.state_dict()
                if not os.path.exists(self.ckpt_save_path):
                    os.mkdir(self.ckpt_save_path)
                torch.save(self.state, os.path.join(self.ckpt_save_path, f'ckpt_{start_time}_epoch{n}.ckpt'))

    def save(self, lr, n):
        self.state['lr'] = lr
        self.state['epoch'] = n
        self.state['state_dict'] = self.state_dict()
        if not os.path.exists(self.ckpt_save_path):
            os.mkdir(self.ckpt_save_path)
        torch.save(self.state, os.path.join(self.ckpt_save_path, f'ckpt_{self.tag}{self.start_time}_epoch{n}.ckpt'))

    def load(self, ckpt_path):
        state = torch.load(ckpt_path)
        self.load_state_dict(state['state_dict'])

    def __save_state(self, n):
        self.best_epoch = n
        self.best_model = self.state_dict()

    def __load_saved_state(self):
        if self.best_model is None:
            raise ValueError('No saved model available')
        self.load_state_dict(self.best_model)
