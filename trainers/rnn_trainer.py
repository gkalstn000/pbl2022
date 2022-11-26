"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from models.rnn_model import RNNModel


class RNNTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt):
        super(RNNTrainer, self).__init__()

        self.opt = opt
        self.rnn_model = RNNModel(opt)

        self.generated = None
        if opt.isTrain:
            self.optimizer_estimator  = self.rnn_model.create_optimizers(opt)

            self.old_lr = opt.lr

    def run_estimator(self, data):
        self.optimizer_estimator.zero_grad()
        E_losses, estimated = self.rnn_model(data, mode='estimate')
        E_loss = sum(E_losses.values()).mean()
        E_loss.backward()
        self.optimizer_estimator.step()
        self.E_losses = E_losses
        self.estimated = estimated

    def run_feature_selector(self, data):
        selected_features = self.rnn_model(data, mode='feature_select')

    # 변수명 수정 해야함.
    # def run_classifier(self, data):
    #     self.optimizer_G.zero_grad()
    #     g_losses, generated = self.rnn_model(data, mode='generator')
    #     g_loss = sum(g_losses.values()).mean()
    #     g_loss.backward()
    #     self.optimizer_G.step()
    #     self.g_losses = g_losses
    #     self.generated = generated
    #
    # def run_feature_selector(self, data):
    #     self.optimizer_D.zero_grad()
    #     d_losses = self.rnn_model(data, mode='discriminator')
    #     d_loss = sum(d_losses.values()).mean()
    #     d_loss.backward()
    #     self.optimizer_D.step()
    #     self.d_losses = d_losses

    def get_latest_losses(self):
        return {**self.E_losses}

    def get_latest_generated(self):
        return self.generated

    def update_learning_rate(self, epoch):
        self.update_learning_rate(epoch)

    def save(self, epoch):
        self.rnn_model.save(epoch)

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_estimator = new_lr
            else:
                new_lr_estimator = new_lr / 2

            for param_group in self.optimizer_estimator.param_groups:
                param_group['lr'] = new_lr_estimator
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr
