import os
import yaml
from easydict import EasyDict
from utils.utils import mkdir_if_missing


def create_config(config_file_exp):
    """
    Create configuration from experiment config file.

    Args:
        config_file_exp: Path to experiment config file (required, must contain root_dir)

    The config file should contain 'root_dir' field. If not provided, 'outputs' will be used as default.
    """
    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream)
    cfg = EasyDict()
    # Copy
    for k, v in config.items():
        cfg[k] = v
    # Get root_dir from config file, or use default
    root_dir = cfg.get('root_dir', 'outputs')
    # Set paths for traning task (These directories are needed in every stage)
    base_dir = os.path.join(root_dir, cfg['exp_name'])
    mkdir_if_missing(base_dir)
    cfg['output_dir'] = base_dir
    cfg['checkpoint_last'] = os.path.join(base_dir, 'last.pth.tar')
    cfg['checkpoint'] = os.path.join(base_dir, 'checkpoint.pth.tar')
    cfg['list_log'] = os.path.join(base_dir, 'list_log.txt')
    cfg['model'] = os.path.join(base_dir, 'model.pth.tar')
    cfg['train_list_log'] = os.path.join(base_dir, 'train_list_log.txt')
    cfg['val_list_log'] = os.path.join(base_dir, 'val_list_log.txt')
    cfg['x_xhat_list_log'] = os.path.join(base_dir, 'x_xhat_list_log.txt')
    cfg['15_unet_losses'] = os.path.join(base_dir, '15_unet_losses.txt')
    figures_dir = os.path.join(base_dir, 'figures')
    mkdir_if_missing(figures_dir)
    cfg['figures_base'] = figures_dir
    if 'training_stages' in cfg and cfg['training_stages'] is not None:
        stage1_epochs = cfg['training_stages'].get('stage1_epochs', 0)
        stage2_epochs = cfg['training_stages'].get('stage2_epochs', 0)
        total_epochs = stage1_epochs + stage2_epochs
        if total_epochs > 0:
            cfg['epochs'] = total_epochs
            print(f"Training stages detected: Stage 1 = {stage1_epochs} epochs, Stage 2 = {stage2_epochs} epochs")
            print(f"Total epochs automatically set to: {total_epochs}")
    return cfg
