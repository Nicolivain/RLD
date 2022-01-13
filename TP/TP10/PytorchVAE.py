import datetime
import os

import torch
import torch.nn as nn

from progress_bar import print_progress_bar


class PytorchVAE(nn.Module):
    def __init__(self, criterion='bce', logger=None, opt='adam', device='cpu', ckpt_save_path=None):
        super().__init__()
        self.opt_type = opt if opt in ['sgd', 'adam'] else ValueError
        self.opt = None
        self.log = logger
        self.device = device
        self.ckpt_save_path = ckpt_save_path
        self.state = {}
        self.criterion = nn.MSELoss(reduction='sum') if criterion == 'mse' else nn.BCELoss(reduction='sum') if criterion == 'bce' else criterion

        self.best_criterion = {'loss' : 10**10,  'v_loss': 10**10, 'acc': -1, 'v_acc': -1}
        self.best_model = None
        self.best_epoch = None

        # useful stuff that can be needed for during fit
        self.verbose  = None
        self.n_epochs = None
        self.n        = None

    def _train_epoch(self, dataloader):
        epoch_loss = 0
        epoch_bce  = 0
        epoch_kll  = 0
        for idx, batch in enumerate(dataloader):
            batch_x, batch_y = batch
            batch_x = batch_x.to(self.device)

            z, mu, logvar = self.encode(batch_x)
            kll = self.lattent_reg(mu, logvar)
            xhat = self.decode(z)
            bce = self.criterion(xhat, batch_x)
            loss = bce + kll

            loss.backward()
            epoch_loss += loss.item()
            epoch_bce  += bce.item()
            epoch_kll  += kll.item()
            self.opt.step()
            self.opt.zero_grad()

            if self.verbose == 1:
                print_progress_bar(idx, len(dataloader))

        return epoch_loss / len(dataloader.dataset), epoch_bce / len(dataloader.dataset), epoch_kll / len(dataloader.dataset)

    def _validate(self, dataloader):
        epoch_loss = 0
        epoch_bce = 0
        epoch_kll = 0
        for idx, batch in enumerate(dataloader):
            batch_x, batch_y = batch
            batch_x = batch_x.to(self.device)

            z, mu, logvar = self.encode(batch_x)
            kll = self.lattent_reg(mu, logvar)
            xhat = self.decode(z)
            bce = self.criterion(xhat, batch_x)
            loss = bce + kll

            epoch_loss += loss.item()
            epoch_bce += bce.item()
            epoch_kll += kll.item()
            self.opt.step()
            self.opt.zero_grad()

            if self.verbose == 1:
                print_progress_bar(idx, len(dataloader))

        return epoch_loss / len(dataloader.dataset), epoch_bce / len(dataloader.dataset), epoch_kll / len(dataloader.dataset)

    def fit(self, dataloader, n_epochs, lr, validation_data=None, verbose=1, save_criterion='loss', ckpt=None, **kwargs):
        if self.opt_type == 'sgd':
            self.opt = torch.optim.SGD(params=self.parameters(), lr=lr)
        elif self.opt_type == 'adam':
            self.opt = torch.optim.Adam(params=self.parameters(), lr=lr)
        else:
            raise ValueError('Unknown optimizer')

        start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        start_epoch = 0
        self.verbose = verbose

        if ckpt:
            state = torch.load(ckpt)
            start_epoch = state['epoch']
            self.load_state_dict(state['state_dict'])
            for g in self.opt.param_groups:
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
                self.log('Train BCE', train_bce, n)
                self.log('Val BCE', val_bce, n)
                self.log('Train KLL', train_kll, n)
                self.log('Val KLL', val_kll, n)

            if epoch_result[save_criterion] <= self.best_criterion[save_criterion]:
                self.best_criterion = epoch_result
                self.__save_state(n)

            if n % verbose == 0:
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
        torch.save(self.state, os.path.join(self.ckpt_save_path, f'ckpt_{self.start_time}_epoch{n}.ckpt'))

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
