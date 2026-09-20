import argparse
import os
import torch
import matplotlib.pyplot as plt
import shutil
import time
import torch.distributed as dist
import torch.multiprocessing as mp

from utils.config import create_config
from trains.flowpet_trainer import is_main_process
from utils.common_config import (get_train_dataset, get_val_dataset, get_imaging_system,
                                 get_model, get_criterion, get_optimizer, get_train_dataloader,
                                 get_val_dataloader, get_train_dataloader_LOOCV, get_val_dataloader_LOOCV, get_scheduler)


def get_trainer_functions(config):
    """Return trainer functions."""
    backbone = config.get('backbone', '').upper()
    if backbone != 'FLOWPET':
        raise ValueError(f"This release only supports the FlowPET backbone, got: {backbone}")
    from trains.flowpet_trainer import unrolling_train, unrolling_val
    return unrolling_train, unrolling_val
from termcolor import colored, cprint
import json
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Load the config file
FLAGS = argparse.ArgumentParser(description='FlowPET')
FLAGS.add_argument('--config_exp', required=True, help='Location of experiments config file (must contain root_dir)')
FLAGS.add_argument('--distributed', action='store_true', help='Enable distributed training')
FLAGS.add_argument('--world_size', type=int, default=None, help='Number of GPUs for distributed training')
FLAGS.add_argument('--rank', type=int, default=None, help='Rank of the current process for distributed training')


def get_error_metric_name(config):
    """Return error metric name."""
    modelity = config.get('imaging_system', 'PET').upper()
    if modelity == 'MRI':
        return 'nmse'
    else:
        return 'rmse'


def setup_distributed(rank, world_size, backend='nccl'):
    """Set up distributed."""
    if dist.is_initialized():
        return
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12355'
    os.environ.setdefault('NCCL_SOCKET_IFNAME', 'lo')
    os.environ.setdefault('NCCL_IB_DISABLE', '1')
    dist.init_process_group(
        backend=backend,
        init_method='env://',
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed."""
    if dist.is_initialized():
        dist.destroy_process_group()


def setup_device(rank=None):
    """Set up device."""
    if rank is not None:
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(rank)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    return device


def setup_imaging_system(config):
    """Set up imaging system."""
    return get_imaging_system(config)


def setup_model(config, device, log_file=None, imaging_system=None, prefix="", use_ddp=False, tokenizer=None):
    """Set up model."""
    if is_main_process():
        cprint(f"{prefix}Loading model...", 'cyan')
    model = get_model(config, imaging_system, tokenizer)
    model = model.to(device)
    if hasattr(model, 'critic') and model.critic is not None:
        image_size = config.get('image_size', 256)
        dummy_input = torch.randn(1, 1, image_size, image_size).to(device)
        try:
            with torch.no_grad():
                _ = model.critic(dummy_input)
            if is_main_process():
                cprint(f"{prefix}Critic initialized with dummy batch (size={image_size})", 'cyan')
        except Exception as e:
            if is_main_process():
                cprint(f"{prefix}Warning: Critic initialization failed: {e}, will initialize during first forward pass", 'yellow')
    model_params = sum(param.numel() for param in model.parameters()) / 1e6
    if is_main_process():
        cprint(f"{prefix}Model: {model.__class__.__name__}, Parameters: {model_params:.2f}M", 'cyan')
        if log_file:
            with open(log_file, 'a') as f:
                f.write("\nModel Information:\n")
                f.write(f"  Model Class: {model.__class__.__name__}\n")
                f.write(f"  Parameter Count: {model_params:.2f}M\n")
                f.write(f"  Device: {device}\n")
                if use_ddp:
                    f.write(f"  Distributed Training: Enabled (World Size: {dist.get_world_size()})\n")
                f.write("="*80 + "\n")
    if use_ddp and dist.is_initialized():
        local_rank = dist.get_rank()
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )
        if is_main_process():
            cprint(f"{prefix}Model wrapped with DistributedDataParallel (device_ids=[{local_rank}], find_unused_parameters=True)", 'cyan')
    return model


def setup_criterion(config, device, log_file=None, prefix=""):
    """Set up criterion."""
    if is_main_process():
        cprint(f"{prefix}Retrieving criterion...", 'cyan')
    criterion_name = config.get('criterion', 'mse_loss') if hasattr(config, 'get') else getattr(config, 'criterion', 'mse_loss')
    criterion = get_criterion(criterion_name)
    if is_main_process():
        cprint(f"{prefix}Criterion: {criterion.__class__.__name__}", 'cyan')
    return criterion.to(device)


def setup_optimizer(config, model, log_file=None, prefix=""):
    """Set up optimizer."""
    if is_main_process():
        cprint(f"{prefix}Retrieving optimizer...", 'cyan')
    optimizer = get_optimizer(config, model)
    if is_main_process():
        cprint(f"{prefix}Optimizer: {optimizer.__class__.__name__}", 'cyan')
    if log_file:
        with open(log_file, 'a') as f:
            f.write("\nOptimizer Information:\n")
            f.write(f"  Optimizer: {str(optimizer)}\n")
            f.write("="*80 + "\n")
    return optimizer


def setup_scheduler(config, optimizer, log_file=None, prefix=""):
    """Set up scheduler."""
    if config.get('scheduler', None) is not None:
        scheduler = get_scheduler(config, optimizer)
        if is_main_process():
            cprint(f"{prefix}Learning Rate Scheduler: {scheduler.__class__.__name__}", 'cyan')
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-8
        )
        if is_main_process():
            cprint(f"{prefix}Learning Rate Scheduler: ReduceLROnPlateau (monitor SSIM)", 'cyan')
    if log_file:
        with open(log_file, 'a') as f:
            f.write(f"\nLearning Rate Scheduler: {scheduler.__class__.__name__}\n")
            if config.get('scheduler_kwargs'):
                f.write(f"  kwargs={config['scheduler_kwargs']}\n")
            f.write("="*80 + "\n")
    return scheduler


def setup_tokenizer(config, device, log_file=None, prefix=""):
    """Keep the released pipeline in image space."""
    if config.get('use_tokenizer') or config.get('use_latent_space'):
        raise ValueError('The released FlowPET pipeline supports image-space training only')
    return None


def setup_datasets(config, imaging_system, is_loocv=False, fold_idx=None, log_file=None, prefix=""):
    """Set up datasets."""
    if is_main_process():
        cprint(f"{prefix}Building datasets and dataloaders...", 'magenta')
    train_dataset = get_train_dataset(config, imaging_system)
    val_dataset = get_val_dataset(config, imaging_system)
    if is_loocv and fold_idx is not None:
        train_dataloader = get_train_dataloader_LOOCV(config, train_dataset, fold_idx, slices_per_case=81)
        val_dataloader = get_val_dataloader_LOOCV(config, val_dataset, fold_idx, slices_per_case=81)
    else:
        train_dataloader = get_train_dataloader(config, train_dataset)
        val_dataloader = get_val_dataloader(config, val_dataset)
    if is_main_process():
        train_db_name = config.get('train_db_name', 'Unknown')
        val_db_name = config.get('val_db_name', 'Unknown')
        cprint(f"{prefix}Dataset Information:", 'magenta')
        cprint(f"{prefix}  Training Dataset: {train_db_name}, Size: {len(train_dataset)}", 'magenta')
        cprint(f"{prefix}  Validation Dataset: {val_db_name}, Size: {len(val_dataset)}", 'magenta')
        if dist.is_initialized():
            cprint(f"{prefix}  Distributed Training: Enabled (World Size: {dist.get_world_size()})", 'magenta')
        if log_file:
            with open(log_file, 'a') as f:
                f.write("\nDataset Information:\n")
                f.write(f"  Training Dataset: {train_db_name}\n")
                f.write(f"  Training Set Size:   {len(train_dataset)}\n")
                f.write(f"  Validation Dataset: {val_db_name}\n")
                f.write(f"  Validation Set Size: {len(val_dataset)}\n")
                if dist.is_initialized():
                    f.write(f"  Distributed Training: Enabled (World Size: {dist.get_world_size()})\n")
                f.write("="*80 + "\n")
    return train_dataloader, val_dataloader


def load_checkpoint(config, device, model, optimizer, scheduler=None):
    """Load checkpoint."""
    config['global_step'] = 0
    last_checkpoint_path = config.get('checkpoint_last')
    best_checkpoint_path = config.get('checkpoint')
    best_ssim = 0.0
    no_improve_count = 0
    start_epoch = 0
    if last_checkpoint_path and os.path.exists(last_checkpoint_path):
        checkpoint = torch.load(last_checkpoint_path, map_location=device, weights_only=False)
        model_state = checkpoint['model_state']
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            if not any(k.startswith('module.') for k in model_state.keys()):
                model_state = {f'module.{k}': v for k, v in model_state.items()}
        elif any(k.startswith('module.') for k in model_state.keys()):
            model_state = {k.replace('module.', ''): v for k, v in model_state.items() if k.startswith('module.')}
        model.load_state_dict(model_state)
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        if config.get('max_steps') and 'global_step' not in checkpoint:
            raise ValueError('This epoch-based checkpoint has no optimizer step count; use a fresh output directory')
        config['global_step'] = checkpoint.get('global_step', 0)
        if scheduler and checkpoint.get('scheduler_state'):
            scheduler.load_state_dict(checkpoint['scheduler_state'])
        best_ssim = checkpoint.get('best_ssim', 0.0)
        no_improve_count = checkpoint.get('no_improve_count', 0)
        start_epoch = checkpoint.get('epoch', 0)
        if is_main_process():
            cprint(f"Found last checkpoint, loading...", 'blue')
    elif best_checkpoint_path and os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model_state = checkpoint['model_state']
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            if not any(k.startswith('module.') for k in model_state.keys()):
                model_state = {f'module.{k}': v for k, v in model_state.items()}
        elif any(k.startswith('module.') for k in model_state.keys()):
            model_state = {k.replace('module.', ''): v for k, v in model_state.items() if k.startswith('module.')}
        model.load_state_dict(model_state)
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        if config.get('max_steps') and 'global_step' not in checkpoint:
            raise ValueError('This epoch-based checkpoint has no optimizer step count; use a fresh output directory')
        config['global_step'] = checkpoint.get('global_step', 0)
        if scheduler and checkpoint.get('scheduler_state'):
            scheduler.load_state_dict(checkpoint['scheduler_state'])
        best_ssim = checkpoint.get('best_ssim', 0.0)
        no_improve_count = checkpoint.get('no_improve_count', 0)
        start_epoch = checkpoint.get('epoch', 0)
        if is_main_process():
            cprint(f"Found best checkpoint, loading...", 'blue')
    else:
        if is_main_process():
            cprint(f"No checkpoint found, initializing from scratch...", 'blue')
    return best_ssim, no_improve_count, start_epoch


def save_checkpoint(config, epoch, model, optimizer, scheduler, best_ssim, no_improve_count,
                    checkpoint_type='best', fold_idx=None):
    """Save checkpoint."""
    if not is_main_process():
        return
    model_state = model.state_dict()
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
    checkpoint_dict = {
        'epoch': epoch + 1,
        'global_step': config.get('global_step', 0),
        'model_state': model_state,
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict() if scheduler else None,
        'best_ssim': best_ssim,
        'no_improve_count': no_improve_count,
    }
    if fold_idx is not None:
        checkpoint_dict['fold_idx'] = fold_idx
    if checkpoint_type == 'best':
        checkpoint_path = config['checkpoint']
        torch.save(checkpoint_dict, checkpoint_path)
        cprint(f"Saved best checkpoint: {checkpoint_path}", 'blue')
    elif checkpoint_type == 'last':
        checkpoint_path = config['checkpoint_last']
        torch.save(checkpoint_dict, checkpoint_path)
        cprint(f"Saved last checkpoint: {checkpoint_path}", 'blue')


def save_metrics_plot(metrics_dict, save_path, config, title="Training Metrics", fold_idx=None):
    """Save metrics plot."""
    error_metric = get_error_metric_name(config)
    error_metric_upper = error_metric.upper()
    fig, axs = plt.subplots(3, 2, figsize=(10, 12))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    loss_list = metrics_dict.get('loss', [])
    ssim_list = metrics_dict.get('ssim', [])
    psnr_list = metrics_dict.get('psnr', [])
    error_list = metrics_dict.get(error_metric, [])
    ssim_v_list = metrics_dict.get('ssim_val', [])
    psnr_v_list = metrics_dict.get('psnr_val', [])
    error_v_list = metrics_dict.get(f'{error_metric}_val', [])
    axs[0, 0].plot(loss_list, label="train_loss")
    axs[0, 0].set_title("Training Loss", loc="right")
    axs[1, 0].plot(ssim_list, label="train_ssim")
    axs[1, 0].set_ylim(0.7, 1.0)
    axs[1, 0].set_title("Training SSIM", loc="right")
    axs[2, 0].plot(psnr_list, label="train_psnr")
    axs[2, 0].set_ylim(25, 45)
    axs[2, 0].set_title("Training PSNR", loc="right")
    axs[0, 1].plot(ssim_v_list, label="val_ssim")
    axs[0, 1].set_ylim(0.7, 1.0)
    axs[0, 1].set_title("Validation SSIM", loc="right")
    axs[1, 1].plot(psnr_v_list, label="val_psnr")
    axs[1, 1].set_ylim(25, 45)
    axs[1, 1].set_title("Validation PSNR", loc="right")
    axs[2, 1].plot(error_v_list, label=f"val_{error_metric}")
    if error_metric == 'rmse':
        axs[2, 1].set_ylim(0, 0.1)
    else:  # nmse
        axs[2, 1].set_ylim(0, 1.0)
    axs[2, 1].set_title(f"Validation {error_metric_upper}", loc="right")
    if fold_idx is not None:
        title = f"Fold {fold_idx} Metrics"
    fig.suptitle(title, fontsize=16)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def log_metrics_to_file(metrics_file, epoch, loss, ssim, psnr, error_val, ssim_v, psnr_v, error_v, config=None):
    """Log metrics to file."""
    with open(metrics_file, 'a') as f:
        f.write(f"{epoch+1}\t{loss:.6f}\t{ssim:.6f}\t{psnr:.6f}\t{error_val:.6f}\t{ssim_v:.6f}\t{psnr_v:.6f}\t{error_v:.6f}\n")


def log_epoch_to_file(log_file, epoch, total_epochs, train_duration, val_duration,
                      loss, ssim, psnr, error_val, ssim_v, psnr_v, error_v, config, prefix=""):
    """Log epoch to file."""
    error_metric = get_error_metric_name(config)
    error_metric_upper = error_metric.upper()
    with open(log_file, 'a') as f:
        f.write(f"\n{prefix}[Epoch {epoch+1}/{total_epochs}]\n")
        f.write(f"  Training Time:   {train_duration:.2f} s\n")
        f.write(f"  Validation Time: {val_duration:.2f} s\n")
        f.write(f"  Total Time:      {train_duration + val_duration:.2f} s\n")
        f.write(f"  Training Metrics -> Loss: {loss:.4f}, SSIM: {ssim:.4f}, PSNR: {psnr:.4f}, {error_metric_upper}: {error_val:.4f}\n")
        f.write(f"  Validation Metrics -> SSIM_val: {ssim_v:.4f}, PSNR_val: {psnr_v:.4f}, {error_metric_upper}_val: {error_v:.4f}\n")
        f.write("-" * 80 + "\n")


def initialize_log_file(log_file, config, is_resume=False):
    """Handle log file."""
    if not os.path.exists(log_file):
        config_to_save = dict(config)
        if not config_to_save.get('LOOCV', False):
            config_to_save.pop('LOOCV_num', None)
        with open(log_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("Experiment Configuration:\n")
            f.write(json.dumps(config_to_save, indent=4, ensure_ascii=False) + "\n")
            f.write("="*80 + "\n")
            f.write(f"Experiment Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n")
    elif is_resume:
        with open(log_file, 'a') as f:
            f.write("\n" + "="*80 + "\n")
            f.write(f"Resuming training at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n")


def initialize_metrics_file(metrics_file, config):
    """Handle metrics file."""
    if not os.path.exists(metrics_file):
        error_metric = get_error_metric_name(config)
        with open(metrics_file, 'w') as f:
            f.write(f"epoch\tloss\tssim\tpsnr\t{error_metric}\tssim_val\tpsnr_val\t{error_metric}_val\n")


def load_historical_metrics(metrics_file, config):
    """Load historical metrics."""
    error_metric = get_error_metric_name(config)
    metrics = {
        'loss': [],
        'ssim': [],
        'psnr': [],
        error_metric: [],
        'ssim_val': [],
        'psnr_val': [],
        f'{error_metric}_val': []
    }
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            header = next(f).strip().split('\t')
            if 'rmse' in header:
                error_key = 'rmse'
                error_val_key = 'rmse_val'
            elif 'nmse' in header:
                error_key = 'nmse'
                error_val_key = 'nmse_val'
            else:
                error_key = error_metric
                error_val_key = f'{error_metric}_val'
            for line in f:
                values = line.strip().split('\t')
                epoch, loss, ssim, psnr, error_val, ssim_v, psnr_v, error_v = map(float, values)
                metrics['loss'].append(loss)
                metrics['ssim'].append(ssim)
                metrics['psnr'].append(psnr)
                if error_key != error_metric:
                    metrics[error_key] = metrics.get(error_key, [])
                    metrics[error_key].append(error_val)
                    metrics[error_val_key] = metrics.get(error_val_key, [])
                    metrics[error_val_key].append(error_v)
                metrics[error_metric].append(error_val)
                metrics['ssim_val'].append(ssim_v)
                metrics['psnr_val'].append(psnr_v)
                metrics[f'{error_metric}_val'].append(error_v)
    return metrics


def copy_config_file(config_path, output_dir):
    """Handle config file."""
    if config_path and os.path.exists(config_path):
        try:
            shutil.copy2(config_path, os.path.join(output_dir, os.path.basename(config_path)))
            if is_main_process():
                cprint(f"Copied experiment config to: {output_dir}", 'green')
        except Exception as e:
            if is_main_process():
                cprint(f"Warning: Failed to copy config files: {e}", 'yellow')


def train_single_epoch(train_dataloader, model, criterion, optimizer, epoch, device, config, imaging_system, tokenizer=None, prefix=""):
    """Run one training epoch."""
    if dist.is_initialized() and hasattr(train_dataloader.sampler, 'set_epoch'):
        train_dataloader.sampler.set_epoch(epoch)
    if is_main_process():
        cprint(f"{prefix}[Epoch {epoch+1}] Training...", 'white', 'on_blue')
    train_start = time.time()
    unrolling_train, _ = get_trainer_functions(config)
    loss, ssim, psnr, error_val = unrolling_train(
        train_dataloader, model, criterion, optimizer, epoch, device, config,
        imaging_system=imaging_system, tokenizer=tokenizer
    )
    train_duration = time.time() - train_start
    return loss, ssim, psnr, error_val, train_duration


def validate_single_epoch(val_dataloader, model, criterion, optimizer, epoch, device, config, imaging_system, tokenizer=None, prefix=""):
    """Run one validation epoch."""
    if is_main_process():
        cprint(f"{prefix}[Epoch {epoch+1}] Validation...", 'white', 'on_blue')
    val_start = time.time()
    _, unrolling_val = get_trainer_functions(config)
    ssim_v, psnr_v, error_v = unrolling_val(
        val_dataloader, model, criterion, optimizer, epoch, device, config,
        imaging_system=imaging_system, tokenizer=tokenizer
    )
    val_duration = time.time() - val_start
    return ssim_v, psnr_v, error_v, val_duration


def train_loop(config, device, model, criterion, optimizer, scheduler, train_dataloader, val_dataloader,
               imaging_system, start_epoch, best_ssim, no_improve_count, metrics, log_file, metrics_file,
               figures_dir, es_patience, prefix="", fold_idx=None, tokenizer=None):
    """Train until the optimizer-step budget is exhausted."""
    total_epochs = config.get('epochs', 1)
    step_schedule = config.get('scheduler_interval') == 'step'
    max_steps = config.get('max_steps')
    if step_schedule:
        if scheduler is None:
            raise ValueError('scheduler_interval=step requires a scheduler')
        optimizer._flowpet_scheduler = scheduler
    if max_steps:
        import math
        updates_per_epoch = math.ceil(len(train_dataloader) / config.get('gradient_accumulation_steps', 1))
        if updates_per_epoch == 0:
            raise ValueError('Training loader is empty')
        remaining = max(0, max_steps - config.get('global_step', 0))
        total_epochs = start_epoch + math.ceil(remaining / updates_per_epoch)
    if is_main_process():
        cprint(f"{prefix}Starting training from epoch {start_epoch}/{total_epochs}...", 'green')
        if log_file:
            with open(log_file, 'a') as f:
                f.write("\nTraining Process:\n")
    error_metric = get_error_metric_name(config)
    epoch = start_epoch
    for epoch in range(start_epoch, total_epochs):
        sampler = getattr(train_dataloader, 'sampler', None)
        if hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)
        epoch_start_time = time.time()
        if is_main_process():
            cprint(f"{prefix}Epoch {epoch+1}/{total_epochs}", 'yellow')
            cprint("-" * 40, 'yellow')
        loss, ssim, psnr, error_val, train_duration = train_single_epoch(
            train_dataloader, model, criterion, optimizer, epoch, device, config, imaging_system, tokenizer=tokenizer, prefix=prefix
        )
        if is_main_process():
            metrics['loss'].append(loss)
            metrics['ssim'].append(ssim)
            metrics['psnr'].append(psnr)
            metrics[error_metric].append(error_val)
        ssim_v, psnr_v, error_v, val_duration = validate_single_epoch(
            val_dataloader, model, criterion, optimizer, epoch, device, config, imaging_system, tokenizer=tokenizer, prefix=prefix
        )
        if is_main_process():
            metrics['ssim_val'].append(ssim_v)
            metrics['psnr_val'].append(psnr_v)
            metrics[f'{error_metric}_val'].append(error_v)
        if is_main_process() and (epoch + 1) % config.get('save_every_epoch', 5) == 0:
            save_checkpoint(config, epoch, model, optimizer, scheduler, best_ssim,
                          no_improve_count, checkpoint_type='last', fold_idx=fold_idx)
        if is_main_process():
            if log_file:
                log_epoch_to_file(log_file, epoch, total_epochs, train_duration, val_duration,
                                loss, ssim, psnr, error_val, ssim_v, psnr_v, error_v, config, prefix)
            if metrics_file:
                log_metrics_to_file(metrics_file, epoch, loss, ssim, psnr, error_val, ssim_v, psnr_v, error_v, config)
        if not step_schedule:
            if config.get('scheduler', None) is not None:
                scheduler.step()
            else:
                scheduler.step(ssim_v)
        if is_main_process():
            if ssim_v > best_ssim:
                best_ssim = ssim_v
                no_improve_count = 0
                save_checkpoint(config, epoch, model, optimizer, scheduler, best_ssim,
                              no_improve_count, checkpoint_type='best', fold_idx=fold_idx)
                cprint(f"{prefix}[Epoch {epoch+1}] Validation SSIM improved to {best_ssim:.4f}", 'blue')
            else:
                no_improve_count += 1
                cprint(f"{prefix}[Epoch {epoch+1}] Validation SSIM did not improve (current {ssim_v:.4f}, best {best_ssim:.4f}), no-improve count: {no_improve_count}/{es_patience}", 'yellow')
            if not max_steps and no_improve_count >= es_patience:
                cprint(f"{prefix}No validation SSIM improvement for {es_patience} consecutive epochs, triggering Early Stopping.", 'red')
                break
            if figures_dir:
                plot_filename = f"metrics_fold{fold_idx}.png" if fold_idx is not None else "metrics.png"
                plot_path = os.path.join(figures_dir, plot_filename)
                save_metrics_plot(metrics, plot_path, config, fold_idx=fold_idx)
                cprint(f"{prefix}[Epoch {epoch+1}] Saved metric plot: {plot_path}", 'blue')
        if dist.is_initialized():
            should_stop = torch.tensor(1 if not max_steps and no_improve_count >= es_patience else 0, device=device)
            dist.all_reduce(should_stop, op=dist.ReduceOp.MAX)
            if should_stop.item() == 1:
                break
        if max_steps and config.get('global_step', 0) >= max_steps:
            save_checkpoint(config, epoch, model, optimizer, scheduler, best_ssim,
                            no_improve_count, checkpoint_type='last', fold_idx=fold_idx)
            break
    return best_ssim, no_improve_count, epoch


def train_loocv(config, args, rank=None, world_size=None):
    """Train one leave-one-out cross-validation run."""
    if not config.get('LOOCV', False):
        raise ValueError("train_loocv() called but LOOCV is not enabled in config")
    if 'LOOCV_num' not in config:
        raise ValueError("LOOCV is enabled but 'LOOCV_num' is not specified in config")
    use_ddp = rank is not None and world_size is not None
    if use_ddp:
        if not dist.is_initialized():
            setup_distributed(rank, world_size)
            device = setup_device(rank)
        else:
            local_rank = int(os.environ.get('LOCAL_RANK', rank))
            device = setup_device(local_rank)
    else:
        device = setup_device()
    base_output_dir = config['output_dir']
    if is_main_process():
        os.makedirs(base_output_dir, exist_ok=True)
        copy_config_file(args.config_exp, base_output_dir)
    if use_ddp:
        dist.barrier()
    es_patience = config.get('ES_PATIENCE', 200)
    error_metric = get_error_metric_name(config)
    error_metric_upper = error_metric.upper()
    SSIM_by_fold = []
    PSNR_by_fold = []
    ERROR_by_fold = []
    loocv_num = config['LOOCV_num']
    for fold_idx in range(loocv_num):
        fold_config = dict(config)
        fold_name = f"LOOCV_{fold_idx}"
        fold_dir = os.path.join(base_output_dir, fold_name)
        if is_main_process():
            os.makedirs(fold_dir, exist_ok=True)
        logs_dir = os.path.join(fold_dir, "logs")
        figures_dir = os.path.join(fold_dir, "figures")
        checkpoints_dir = os.path.join(fold_dir, "checkpoints")
        if is_main_process():
            os.makedirs(logs_dir, exist_ok=True)
            os.makedirs(figures_dir, exist_ok=True)
            os.makedirs(checkpoints_dir, exist_ok=True)
        if use_ddp:
            dist.barrier()
        fold_config['output_dir'] = fold_dir
        fold_config['figures_base'] = figures_dir
        fold_config['checkpoint'] = os.path.join(checkpoints_dir, f"best_checkpoint_fold{fold_idx}.pth")
        fold_config['checkpoint_last'] = os.path.join(checkpoints_dir, f"last_checkpoint_fold{fold_idx}.pth")
        fold_config['model'] = os.path.join(checkpoints_dir, f"final_model_fold{fold_idx}.pth")
        fold_config['list_log'] = os.path.join(logs_dir, f"metrics_fold{fold_idx}.txt")
        fold_log_file = os.path.join(logs_dir, f"log_{fold_idx}.out")
        if is_main_process():
            initialize_log_file(fold_log_file, fold_config)
            initialize_metrics_file(fold_config['list_log'], fold_config)
        prefix = f"[Fold {fold_idx}] "
        imaging_system = setup_imaging_system(fold_config)
        if fold_config.get('use_tokenizer', False):
            tokenizer = setup_tokenizer(fold_config, device, fold_log_file if is_main_process() else None, prefix)
        else:
            tokenizer = None
        model = setup_model(fold_config, device, fold_log_file if is_main_process() else None, imaging_system, prefix, use_ddp=use_ddp, tokenizer=tokenizer)
        criterion = setup_criterion(fold_config, device, None, prefix)
        optimizer = setup_optimizer(fold_config, model, fold_log_file if is_main_process() else None, prefix)
        scheduler = setup_scheduler(fold_config, optimizer, fold_log_file if is_main_process() else None, prefix)
        best_ssim, no_improve_count, start_epoch = load_checkpoint(
            fold_config, device, model, optimizer, scheduler
        )
        if is_main_process():
            cprint(f"{prefix}Resuming from epoch {start_epoch}, previous best SSIM: {best_ssim:.4f}, no-improve count: {no_improve_count}", 'blue')
        train_dataloader, val_dataloader = setup_datasets(
            fold_config, imaging_system, is_loocv=True, fold_idx=fold_idx,
            log_file=fold_log_file if is_main_process() else None, prefix=prefix
        )
        error_metric = get_error_metric_name(fold_config)
        if is_main_process():
            metrics = {
                'loss': [],
                'ssim': [],
                'psnr': [],
                error_metric: [],
                'ssim_val': [],
                'psnr_val': [],
                f'{error_metric}_val': []
            }
        else:
            metrics = {
                'loss': [],
                'ssim': [],
                'psnr': [],
                error_metric: [],
                'ssim_val': [],
                'psnr_val': [],
                f'{error_metric}_val': []
            }
        best_ssim, no_improve_count, final_epoch = train_loop(
            fold_config, device, model, criterion, optimizer, scheduler,
            train_dataloader, val_dataloader, imaging_system, start_epoch,
            best_ssim, no_improve_count, metrics, fold_log_file if is_main_process() else None,
            fold_config['list_log'] if is_main_process() else None,
            figures_dir, es_patience, prefix, fold_idx, tokenizer=tokenizer
        )
        if is_main_process():
            with open(fold_log_file, 'a') as f:
                if no_improve_count < es_patience:
                    actual_end_epoch = final_epoch + 1
                else:
                    actual_end_epoch = final_epoch - no_improve_count + 1
                f.write("\n")
                f.write(f"Fold {fold_idx} training finished after {fold_config.get('global_step', 0)} optimizer steps "
                        f"({actual_end_epoch} validation epochs).\n")
                f.write(f"Best validation SSIM: {best_ssim:.4f}\n")
                f.write(f"Best checkpoint path: {fold_config['checkpoint']}\n")
                f.write(f"Final model path: {fold_config['model']}\n")
                f.write(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*80 + "\n")
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model_state = {k.replace('module.', ''): v for k, v in model.state_dict().items()}
            else:
                model_state = model.state_dict()
            torch.save(model_state, fold_config['model'])
            cprint(f"{prefix}Saved final model: {fold_config['model']}", 'green')
            SSIM_by_fold.append(best_ssim)
            PSNR_by_fold.append(metrics['psnr_val'][-1] if metrics['psnr_val'] else 0)
            error_metric = get_error_metric_name(fold_config)
            ERROR_by_fold.append(metrics[f'{error_metric}_val'][-1] if metrics[f'{error_metric}_val'] else 0)
    if is_main_process():
        error_metric = get_error_metric_name(config)
        error_metric_upper = error_metric.upper()
        avg_ssim = sum(SSIM_by_fold) / len(SSIM_by_fold)
        avg_psnr = sum(PSNR_by_fold) / len(PSNR_by_fold)
        avg_error = sum(ERROR_by_fold) / len(ERROR_by_fold)
        ssim_std = np.std(SSIM_by_fold)
        psnr_std = np.std(PSNR_by_fold)
        error_std = np.std(ERROR_by_fold)
        cprint("Average Metrics Across All Folds:", 'magenta')
        cprint(f"Average SSIM: {avg_ssim:.4f}", 'magenta')
        cprint(f"Average PSNR: {avg_psnr:.4f}", 'magenta')
        cprint(f"Average {error_metric_upper}: {avg_error:.4f}", 'magenta')
        cprint("="*80, 'magenta')
        with open(os.path.join(config['output_dir'], 'Final_LOOCV_metrics.txt'), 'w') as f:
            f.write("Average Metrics Across All Folds:\n")
            f.write(f"Average SSIM: {avg_ssim:.4f} ± {ssim_std:.4f}\n")
            f.write(f"Average PSNR: {avg_psnr:.4f} ± {psnr_std:.4f}\n")
            f.write(f"Average {error_metric_upper}: {avg_error:.4f} ± {error_std:.4f}\n")
            f.write("="*80 + "\n")
        cprint("All LOOCV folds have been completed.", 'magenta')
    if use_ddp:
        cleanup_distributed()


def train_single_model(config, args, rank=None, world_size=None):
    """Train one FlowPET model."""
    use_ddp = rank is not None and world_size is not None
    if use_ddp:
        if not dist.is_initialized():
            setup_distributed(rank, world_size)
            device = setup_device(rank)
        else:
            local_rank = int(os.environ.get('LOCAL_RANK', rank))
            device = setup_device(local_rank)
    else:
        device = setup_device()
    if is_main_process():
        os.makedirs(config['output_dir'], exist_ok=True)
    if use_ddp:
        dist.barrier()
    log_file = os.path.join(config['output_dir'], 'log.out')
    metrics_log_file = os.path.join(config['output_dir'], 'metrics.txt')
    if is_main_process():
        initialize_log_file(log_file, config, is_resume=os.path.exists(log_file))
        initialize_metrics_file(metrics_log_file, config)
        copy_config_file(args.config_exp, config['output_dir'])
    es_patience = config.get('ES_PATIENCE', 200)
    error_metric = get_error_metric_name(config)
    if is_main_process():
        metrics = load_historical_metrics(metrics_log_file, config)
    else:
        metrics = {
            'loss': [],
            'ssim': [],
            'psnr': [],
            error_metric: [],
            'ssim_val': [],
            'psnr_val': [],
            f'{error_metric}_val': []
        }
    imaging_system = get_imaging_system(config)
    tokenizer = setup_tokenizer(config, device, log_file if is_main_process() else None, "")
    model = setup_model(config, device, log_file if is_main_process() else None, imaging_system, "", use_ddp=use_ddp, tokenizer=tokenizer)
    criterion = setup_criterion(config, device, None)
    optimizer = setup_optimizer(config, model, log_file if is_main_process() else None)
    scheduler = setup_scheduler(config, optimizer, log_file if is_main_process() else None)
    best_ssim, no_improve_count, start_epoch = load_checkpoint(
        config, device, model, optimizer, scheduler
    )
    train_dataloader, val_dataloader = setup_datasets(config, imaging_system, log_file=log_file if is_main_process() else None)
    best_ssim, no_improve_count, final_epoch = train_loop(
            config, device, model, criterion, optimizer, scheduler,
            train_dataloader, val_dataloader, imaging_system, start_epoch,
            best_ssim, no_improve_count, metrics, log_file if is_main_process() else None,
            metrics_log_file if is_main_process() else None,
            config.get('figures_base'), es_patience, tokenizer=tokenizer
        )
    if is_main_process():
        with open(log_file, 'a') as f:
            if no_improve_count < es_patience:
                actual_end_epoch = final_epoch + 1
            else:
                actual_end_epoch = final_epoch - no_improve_count + 1
            f.write("\n")
            f.write(f"Training finished after {config.get('global_step', 0)} optimizer steps "
                    f"({actual_end_epoch} validation epochs).\n")
            f.write(f"Best validation SSIM: {best_ssim:.4f}\n")
            f.write(f"Best checkpoint path: {config['checkpoint']}\n")
            f.write(f"Final model path: {config['model']}\n")
            f.write(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n")
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = {k.replace('module.', ''): v for k, v in model.state_dict().items()}
        else:
            model_state = model.state_dict()
        torch.save(model_state, config['model'])
        cprint(f"Saved final model: {config['model']}", 'green')
    if use_ddp:
        cleanup_distributed()


def main_worker(rank, world_size, args):
    """Handle worker."""
    torch.manual_seed(0)
    config = create_config(args.config_exp)
    if 'LOOCV' in config and config['LOOCV']:
        if 'output_dir' not in config or not config['output_dir']:
            raise ValueError("Please set 'output_dir' in the configuration as the common parent directory for all folds.")
        train_loocv(config, args, rank=rank, world_size=world_size)
    else:
        train_single_model(config, args, rank=rank, world_size=world_size)


def main():
    """Run the command-line entry point."""
    torch.manual_seed(0)
    args = FLAGS.parse_args()
    config = create_config(args.config_exp)
    use_torchrun = 'RANK' in os.environ and 'LOCAL_RANK' in os.environ and 'WORLD_SIZE' in os.environ
    if args.distributed or use_torchrun:
        if use_torchrun:
            rank = int(os.environ['RANK'])
            local_rank = int(os.environ['LOCAL_RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            torch.cuda.set_device(local_rank)
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl', timeout=dist.default_pg_timeout)
            os.environ.setdefault('NCCL_SOCKET_IFNAME', 'lo')
            os.environ.setdefault('NCCL_IB_DISABLE', '1')
            if is_main_process():
                cprint(f"Using torchrun: rank={rank}, local_rank={local_rank}, world_size={world_size}", 'green')
            if 'LOOCV' in config and config['LOOCV']:
                if 'output_dir' not in config or not config['output_dir']:
                    raise ValueError("Please set 'output_dir' in the configuration as the common parent directory for all folds.")
                train_loocv(config, args, rank=rank, world_size=world_size)
            else:
                train_single_model(config, args, rank=rank, world_size=world_size)
            return
        else:
            if args.world_size is None:
                world_size = torch.cuda.device_count()
                if world_size == 0:
                    raise RuntimeError("No CUDA devices available for distributed training")
            else:
                world_size = args.world_size
            if world_size <= 1:
                if is_main_process():
                    cprint("Warning: Only 1 GPU available, falling back to single GPU training", 'yellow')
                args.distributed = False
            if args.distributed:
                if is_main_process():
                    cprint(f"Starting distributed training with {world_size} GPUs (using mp.spawn)", 'green')
                if 'MASTER_ADDR' not in os.environ:
                    os.environ['MASTER_ADDR'] = 'localhost'
                if 'MASTER_PORT' not in os.environ:
                    import socket
                    sock = socket.socket()
                    sock.bind(('', 0))
                    port = sock.getsockname()[1]
                    sock.close()
                    os.environ['MASTER_PORT'] = str(port)
                    if is_main_process():
                        cprint(f"Using MASTER_PORT={port}", 'green')
                os.environ.setdefault('NCCL_SOCKET_IFNAME', 'lo')
                os.environ.setdefault('NCCL_IB_DISABLE', '1')
                try:
                    current_method = mp.get_start_method(allow_none=True)
                    if current_method is None:
                        import platform
                        system = platform.system()
                        if system == 'Linux':
                            try:
                                mp.set_start_method('fork')
                            except (RuntimeError, ValueError):
                                try:
                                    mp.set_start_method('spawn')
                                except (RuntimeError, ValueError) as e:
                                    if is_main_process():
                                        cprint(f"Warning: Could not set multiprocessing start method: {e}", 'yellow')
                        elif system in ['Windows', 'Darwin']:
                            try:
                                mp.set_start_method('spawn')
                            except (RuntimeError, ValueError) as e:
                                if is_main_process():
                                    cprint(f"Warning: Could not set multiprocessing start method: {e}", 'yellow')
                        else:
                            try:
                                mp.set_start_method('spawn')
                            except (RuntimeError, ValueError) as e:
                                if is_main_process():
                                    cprint(f"Warning: Could not set multiprocessing start method: {e}", 'yellow')
                except RuntimeError as e:
                    pass
                except Exception as e:
                    if is_main_process():
                        cprint(f"Warning: Error setting multiprocessing start method: {e}", 'yellow')
                mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)
                return
    if 'LOOCV' in config and config['LOOCV']:
        if 'output_dir' not in config or not config['output_dir']:
            raise ValueError("Please set 'output_dir' in the configuration as the common parent directory for all folds.")
        train_loocv(config, args)
    else:
        train_single_model(config, args)

if __name__ == "__main__":
    main()
