from skorch.callbacks import Checkpoint, EarlyStopping
import numpy as np

class SaveEveryNEpochs(Checkpoint):
    def __init__(self,every=10, **kwargs):
        super().__init__( **kwargs)
        self.every = every

    def on_epoch_end(self, net, **kwargs):
        # Check if the current epoch is a multiple of 'every'
        if (net.history[-1]['epoch']) != net.max_epochs:
            if (net.history[-1]['epoch']) % self.every == 0:
                super().on_epoch_end(net, **kwargs)

class SaveFirstEpoch(Checkpoint):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_epoch_end(self, net, **kwargs):
        if (net.history[-1]['epoch'])  == 1:
            super().on_epoch_end(net, **kwargs)

class SaveLastEpoch(Checkpoint):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_epoch_end(self, net, **kwargs):
        if (net.history[-1]['epoch'])  == net.max_epochs:
            super().on_epoch_end(net, **kwargs)

class EarlyStoppingCheckpoint(EarlyStopping):
    def __init__(self, dirname,f_params,
                    f_optimizer,
                    f_history,
                    f_criterion,
                    fn_prefix,
                    **kwargs):
        # Initialize the EarlyStopping callback
        super().__init__(**kwargs)
        self.checkpoint_last_epoch = Checkpoint(dirname=dirname,
                                     f_params=f_params,
                                     f_optimizer=f_optimizer,
                                     f_history=f_history,
                                     f_criterion=f_criterion,
                                     fn_prefix=fn_prefix,
                                     load_best=False,
                                     monitor=None)
        self.checkpoint_best_epoch = Checkpoint(dirname=dirname,
                                monitor = None,
                                f_params='best_model.pt',
                                f_optimizer=None,   
                                f_history=None,
                                f_criterion=None,
                                fn_prefix = "",
                                load_best=True)
    
    def on_train_end(self, net, **kwargs):
        # Call the Checkpoint callback
        self.checkpoint_last_epoch.on_epoch_end(net, **kwargs)
        # Call the EarlyStopping callback
        super().on_train_end(net, **kwargs)
        # Save the best model
        self.checkpoint_best_epoch.on_epoch_end(net, **kwargs)