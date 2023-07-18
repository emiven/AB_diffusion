# import the necessary packages
import torch
from torch.optim import Adam

from AB_diffusion import ABTrainer


class ABFineTuner(ABTrainer):
    def __init__(
        self,
        pretrained_model,
        layers_to_copy=None,
        layers_to_freeze=None,
        layers_to_initialize=None,
        learning_rates=None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        # Copy layers from the pretrained model
        if layers_to_copy:
            self.copy_layers(pretrained_model, layers_to_copy)

        # Freeze layers in the model
        if layers_to_freeze:
            self.freeze_layers(layers_to_freeze)

        # Randomly initialize specific layers
        if layers_to_initialize:
            self.initialize_layers(layers_to_initialize)

        # Set different learning rates for specific layers
        if learning_rates:
            self.set_learning_rates(learning_rates)

    def copy_layers(self, pretrained_model, layers_to_copy):
        # Copy the specified layers from the pretrained model to the fine-tuned model
        for name, param in pretrained_model.named_parameters():
            if any(layer_name in name for layer_name in layers_to_copy):
                self.model.load_state_dict({name: param})

    def freeze_layers(self, layers_to_freeze):
        # Freeze the specified layers in the model
        for name, param in self.model.named_parameters():
            if any(layer_name in name for layer_name in layers_to_freeze):
                param.requires_grad = False

    def initialize_layers(self, layers_to_initialize):
        # Randomly initialize the specified layers
        for name, param in self.model.named_parameters():
            if any(layer_name in name for layer_name in layers_to_initialize):
                torch.nn.init.xavier_uniform_(param)

    def set_learning_rates(self, learning_rates):
        # Set different learning rates for specific layers
        param_groups = []
        for name, param in self.model.named_parameters():
            for layer_name, lr in learning_rates.items():
                if layer_name in name:
                    param_groups.append({'params': param, 'lr': lr})
        self.opt = Adam(param_groups, lr=self.train_lr, betas=self.adam_betas)
