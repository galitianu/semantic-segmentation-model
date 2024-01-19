from torch import optim, nn
from torch.utils.data import DataLoader

import wandb
from dataset import LFWDataset
from model.UNet import UNet
from train import train

sweep_config = {
    'method': 'random'
}

metric = {
    'name': 'validation-loss',
    'goal': 'minimize'
}

sweep_config['metric'] = metric

parameters_dict = {
    'optimizer': {
        'values': ['adam', 'sgd']
    },
}

sweep_config['parameters'] = parameters_dict

parameters_dict.update({
    'epochs': {
        'value': 10}
})

parameters_dict.update({
    'learning_rate': {
        # a flat distribution between 0 and 0.1
        'distribution': 'uniform',
        'min': 0,
        'max': 0.1
    },
    'batch_size': {
        'distribution': 'q_log_uniform_values',
        'q': 8,
        'min': 4,
        'max': 12,
    }
})

validation_table = wandb.Table(columns=["Image", "Prediction", "Ground Truth"])


grid_sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'validation-loss',
        'goal': 'minimize'
    },
    'parameters': {
        'optimizer': {
            'values': ['adam', 'sgd']
        },
        'epochs': {
            'values': [5, 10]
        },
        'learning_rate': {
            'values': [0.01, 0.001, 0.0001]
        },
        'batch_size': {
            'values': [6, 8, 10]
        }
    }
}



def sweep_run():
    with wandb.init() as run:
        config = wandb.config

        device = "mps"  # Device for computation

        # Load and prepare the training data
        train_dataset = LFWDataset(download=False, base_folder='lfw_dataset', split_name="train", transforms=None)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, pin_memory=True, shuffle=True,
                                  sampler=None,
                                  num_workers=0)

        # Load and prepare the validation data
        val_dataset = LFWDataset(download=False, base_folder='lfw_dataset',
                                 split_name="val", transforms=None)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, pin_memory=True, shuffle=False, sampler=None,
                                num_workers=0)

        encoder_channels = [3, 64, 128, 256, 512]
        decoder_depths = [256, 128, 64]
        num_classes = 3  # Number of classes in the segmentation problem

        # Setup model, optimizer, criterion based on sweep parameters
        model = UNet(encoder_channels, decoder_depths, num_classes)
        model = model.to(device)

        if config.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)
        elif config.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)

        # Define the loss function - CrossEntropyLoss for multi-class segmentation
        criterion = nn.CrossEntropyLoss()
        # Training loop
        for epoch in range(config.epochs):
            train(train_loader, val_loader, model, optimizer, criterion, epoch, config.epochs, device, validation_table,
                  3)


if __name__ == '__main__':
    sweep_id = wandb.sweep(grid_sweep_config, project="semantic-segmentation-model")
    wandb.agent(sweep_id, sweep_run)
