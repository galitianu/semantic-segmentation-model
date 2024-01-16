import os
import torch
import wandb
import re


class ModelCheckpoint:
    def __init__(self, num_checkpoints, checkpoint_dir, decreasing_metric=True):
        self.checkpoint_dir = checkpoint_dir
        self.num_checkpoints = num_checkpoints
        self.decreasing_metric = decreasing_metric
        self.best_metric_val = float('inf') if decreasing_metric else -float('inf')
        self.checkpoints = []
        for filename in os.listdir(self.checkpoint_dir):
            if filename.endswith('.pt'):
                # Extract metric value from filename
                match = re.search(r"metric_([+-]?[0-9]*[.]?[0-9]+).pt$", filename)
                if match:
                    metric_val = float(match.group(1))
                    self.checkpoints.append((os.path.join(self.checkpoint_dir, filename), metric_val))

            # Update the best metric value
        if self.checkpoints:
            self.cleanup()
            sorted_checkpoints = sorted(self.checkpoints, key=lambda x: x[1], reverse=not self.decreasing_metric)
            self.best_metric_val = sorted_checkpoints[0][1]
            self.checkpoints = sorted_checkpoints[:self.num_checkpoints]

    def __call__(self, model, epoch, metric_val):
        must_save = metric_val < self.best_metric_val if self.decreasing_metric else metric_val > self.best_metric_val
        if must_save:
            self.best_metric_val = metric_val
            model_path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch}_metric_{metric_val}.pt")
            torch.save(model.state_dict(), model_path)

            # Log the model as an artifact
            self.write_artifact("model_artifact", model_path, metric_val)

            # Add checkpoint to the list and perform cleanup if necessary
            self.checkpoints.append((model_path, metric_val))
            self.cleanup()

    def cleanup(self):
        if len(self.checkpoints) > self.num_checkpoints:
            # Sort checkpoints by metric, delete the least performing ones
            sorted_checkpoints = sorted(self.checkpoints, key=lambda x: x[1], reverse=not self.decreasing_metric)
            for checkpoint in sorted_checkpoints[self.num_checkpoints:]:
                os.remove(checkpoint[0])
            self.checkpoints = sorted_checkpoints[:self.num_checkpoints]

    def write_artifact(self, name, model_path, metric_val):
        artifact = wandb.Artifact(name, type='model', metadata={'metric': metric_val})
        artifact.add_file(model_path)
        wandb.run.log_artifact(artifact)
