from torch.optim import Adam


## utility functions for transfer learning


# Function to copy specific layers from a pretrained model to a new model
def copy_params(model_new, model_pretrained, params_not_to_copy):
    """Copies params from a pretrained model to a new model.
    
    Args:
        model_new (nn.Module): The new model to which the layers will be copied.
        model_pretrained (nn.Module): The pretrained model from which the layers will be copied.
        layers_not_to_copy (list): List of layer names that should not be copied from the pretrained model.

    """
    for layer_name in model_pretrained.state_dict():
        if layer_name not in params_not_to_copy:
            model_new.load_state_dict({layer_name: model_pretrained.state_dict()[layer_name]}, strict=False)
        else:
            print(f'Not copying layer {layer_name}')

# Function to freeze specific layers in a model
def freeze_params(model, param_names):
    """Freezes specific layers in a model.

    Args:
        model (nn.Module): The model whose layers will be frozen.
        param_names (list): List of layer names to freeze.

    """
    for name, param in model.named_parameters():
        if any(layer in name for layer in param_names):
            param.requires_grad = False

# Function to create an Adam optimizer with different learning rates for specific layers
def create_adam_optimizer(model, param_names, learning_rate_pretrained, learning_rate_reinit, betas=(0.9, 0.999)):
    """Creates an Adam optimizer with different learning rates for specific layers.

    Args:
        model (nn.Module): The model for which the optimizer will be created.
        param_names (list): List of parameter names to reinitialize.
        learning_rate_pretrained (float): Learning rate for the pretrained parameters.
        learning_rate_reinit (float): Learning rate for the reinitialized parameters.
        betas (tuple, optional): Coefficients used for computing running averages of gradient and its square. Defaults to (0.9, 0.999).

    Returns:
        Adam: The created Adam optimizer.

    """
    parameters = []
    
    for name, param in model.named_parameters():
        if any(param_name in name for param_name in param_names):
            parameters.append({'params': param, 'lr': learning_rate_reinit})
        else:
            parameters.append({'params': param, 'lr': learning_rate_pretrained})
            
    optimizer = Adam(parameters, betas=betas)
    return optimizer

# Function to reinitialize specific layers in a model
def reinit_params(model, param_names):
    """Reinitializes specific params in a model.

    Args:
        model (nn.Module): The model whose layers will be reinitialized.
        param_names (list): List of parameter names to reinitialize.

    """
    for param_name in param_names:
        module = model
        for attr in param_name.split('.')[:-1]:
            module = getattr(module, attr)
        if hasattr(module, 'reset_parameters'):
            print(f'Reinitializing {param_name}')
            module.reset_parameters()
