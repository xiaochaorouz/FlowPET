#!/usr/bin/env python3
"""Command-line entry point for FlowPET evaluation."""

import argparse
import sys
import os
import json
import numpy as np
import torch


def count_parameters(model):
    """Count trainable and total parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'total_params_M': total_params / 1e6,
        'trainable_params_M': trainable_params / 1e6
    }


def calculate_flops(model, input_shape=(1, 1, 128, 128), device='cuda'):
    """Estimate floating-point operations with THOP."""
    try:
        from thop import profile
        dummy_input = torch.randn(input_shape).to(device)
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        flops = macs * 2
        return {
            'macs': macs,
            'flops': flops,
            'flops_G': flops / 1e9,
            'macs_G': macs / 1e9,
            'params': params,
            'params_M': params / 1e6
        }
    except Exception as e:
        print(f"FLOPs calculation failed: {e}")
        return {
            'macs': 0,
            'flops': 0,
            'flops_G': 0,
            'macs_G': 0,
            'params': 0,
            'params_M': 0,
            'error': str(e)
        }


def analyze_model_complexity(model, model_name, input_shape=(1, 1, 128, 128), device='cuda'):
    """Report parameter counts and optional FLOPs."""
    print(f"\nAnalyzing model complexity: {model_name}")
    param_info = count_parameters(model)
    print(f"Total parameters: {param_info['total_params']:,} ({param_info['total_params_M']:.2f}M)")
    print(f"Trainable parameters: {param_info['trainable_params']:,} ({param_info['trainable_params_M']:.2f}M)")
    flops_info = calculate_flops(model, input_shape, device)
    if 'error' not in flops_info:
        print(f"MACs: {flops_info['macs']:,} ({flops_info['macs_G']:.2f}G)")
        print(f"FLOPs (MACs × 2): {flops_info['flops']:,} ({flops_info['flops_G']:.2f}G)")
    else:
        print(f"FLOPs calculation failed: {flops_info['error']}")
    return {
        'model_name': model_name,
        'parameters': param_info,
        'flops': flops_info
    }


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate FlowPET reconstructions')
    parser.add_argument('--base_dir', type=str, required=True,
                       help='Directory containing model subdirectories')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Compute device (cuda, cuda:0, cuda:1, or cpu)')
    parser.add_argument('--data_selection', type=str, default='ultra_ultra_low',
                       choices=['ultra_ultra_low', 'ultra_low', 'low', 'full'],
                       help='Input dose level')
    parser.add_argument('--val_db_name', type=str, default=None,
                       help='Validation dataset name (default: model config)')
    parser.add_argument('--proposed_model', type=str, default='FlowPET_Pediatric_1percent',
                       help='FlowPET model directory name in the results dictionary')
    parser.add_argument('--load_keys', type=str, nargs='+', default=None,
                       help='Data keys to load, e.g. --load_keys full ultra_ultra_low')
    parser.add_argument('--metric', type=str, default='rmse',
                       choices=['ssim', 'psnr', 'rmse'],
                       help='Metric used to rank samples')
    parser.add_argument('--metric_order', type=str, default='auto',
                       choices=['desc', 'asc', 'auto'],
                       help='Ranking order (auto selects the appropriate direction)')
    parser.add_argument('--num_samples', type=int, default=1,
                       help='Number of samples to evaluate')
    parser.add_argument('--top_k', type=int, default=1,
                       help='Number of highest-ranked samples to save')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='Index of the first sample')
    parser.add_argument('--skip_models', type=str, nargs='*', default=[],
                       help='Model directory names to skip')
    parser.add_argument('--only_models', type=str, nargs='*', default=[],
                       help='Evaluate only these model directory names')
    parser.add_argument('--skip_256', action='store_true', default=False,
                       help="Skip model names containing '256'")
    parser.add_argument('--skip_large', action='store_true', default=False,
                       help="Skip model names containing 'large'")
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Output directory (default: base_dir/batch_evaluation_results)')
    parser.add_argument('--vmin', type=float, default=0,
                       help='Minimum image display value')
    parser.add_argument('--vmax', type=float, default=1,
                       help='Maximum image display value')
    parser.add_argument('--colormap', type=str, default='gray_r',
                       help='Reconstruction colormap (default: gray_r)')
    parser.add_argument('--error_cmap', type=str, default='Blues',
                       help='Error-map colormap (default: Blues)')
    parser.add_argument('--error_vmax', type=float, default=0.5,
                       help='Shared maximum error-map value (default: 0.5)')
    parser.add_argument('--test_mode', action='store_true',
                       help='Process a single sample')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--reconstruct_3d', action='store_true', default=False,
                       help='Enable 3D reconstruction')
    parser.add_argument('--start_slice', type=int, default=2019,
                       help='First slice index for 3D reconstruction')
    parser.add_argument('--end_slice', type=int, default=2691,
                       help='Exclusive final slice index for 3D reconstruction')
    parser.add_argument('--save_nii', action='store_true', default=False,
                       help='Save 3D reconstructions in NIfTI format')
    parser.add_argument('--simulate_data', action='store_true', default=False,
                       help='Generate simulated sinograms from full-dose images')
    parser.add_argument('--count', type=float, default=2e5,
                       help='Target count for simulated data (default: 2e5)')
    parser.add_argument('--use_loocv', action='store_true', default=False,
                       help='Enable leave-one-out cross-validation mode')
    parser.add_argument('--loocv_fold', type=int, default=0,
                       help='LOOCV fold index')
    parser.add_argument('--loocv_slices_per_case', type=int, default=81,
                       help='Number of slices per case (default: 81)')
    parser.add_argument('--analyze_complexity', action='store_true', default=False,
                       help='Report parameter counts and FLOPs')
    parser.add_argument('--input_shape', type=str, default='1,1,128,128',
                       help='Input shape for FLOPs calculation: batch,channel,height,width')
    parser.add_argument('--val_sampling_steps', type=int, nargs='+', default=[4],
                       help='Validation sampling steps (default: 4)')
    parser.add_argument('--solver', type=str, default='leapfrog',
                       choices=['leapfrog', 'euler', 'rk4'],
                       help='Numerical solver (default: leapfrog)')
    return parser.parse_args()


def build_save_dir(base_save_dir, data_selection, val_db_name, start_slice, end_slice, use_loocv=False, loocv_fold=None):
    """Build save dir."""
    data_selection_map = {
        'ultra_ultra_low': '_uul',
        'ultra_low': '_ul',
        'low': '_low',
        'full': '_full'
    }
    data_selection_suffix = data_selection_map.get(data_selection, f'_{data_selection}')
    dir_name = f"{val_db_name}{data_selection_suffix}_s{start_slice}_e{end_slice}"
    if use_loocv and loocv_fold is not None:
        dir_name = os.path.join(dir_name, f"loocv_fold_{loocv_fold}")
    save_dir = os.path.join(base_save_dir, dir_name)
    return save_dir


def modify_batch_evaluation_config(args):
    """Apply command-line evaluation settings."""
    import evaluation as batch_evaluation
    batch_evaluation.VMIN = args.vmin
    batch_evaluation.VMAX = args.vmax
    batch_evaluation.COLORMAP = args.colormap
    batch_evaluation.ERROR_CMAP = args.error_cmap
    batch_evaluation.ERROR_VMAX = args.error_vmax

    def custom_main():
        """Run evaluation with command-line settings."""
        print("=" * 60)
        print("FlowPET evaluation")
        print("=" * 60)
        base_dir = args.base_dir
        device = torch.device(args.device)
        data_selection = args.data_selection
        num_samples = 1 if args.test_mode else args.num_samples
        top_k = min(args.top_k, num_samples)
        print(f"Model directory: {base_dir}")
        print(f"Device: {device}")
        print(f"Samples: {num_samples}")
        print(f"Top samples to save: {top_k}")
        print(f"Input dose level: {data_selection}")
        print(f"Ranking metric: {args.metric} ({args.metric_order})")
        print(f"Start index: {args.start_idx}")
        print(f"Image colormap: {args.colormap}")
        print(f"Error-map colormap: {args.error_cmap}")
        if args.error_vmax is not None:
            print(f"Error-map maximum: {args.error_vmax}")
        else:
            print("Error-map maximum: automatic")
        if args.use_loocv:
            print(f"LOOCV fold: {args.loocv_fold}; slices per case: {args.loocv_slices_per_case}")
        if args.reconstruct_3d:
            print(f"3D reconstruction slices: [{args.start_slice}, {args.end_slice})")
            if args.save_nii:
                print("NIfTI output enabled")
        if args.skip_models:
            print(f"Skipped models: {args.skip_models}")
        if args.only_models:
            print(f"Selected models: {args.only_models}")
        if args.skip_256:
            print("Skipping model names containing '256'")
        if args.skip_large:
            print("Skipping model names containing 'large'")
        first_model_dir = None
        for d in os.listdir(base_dir):
            d_path = os.path.join(base_dir, d)
            if os.path.isdir(d_path) and d not in {'__pycache__', 'recon', 'batch_evaluation_results', '.git', '.vscode'} and '256' not in d:
                print(f"Reference model config: {d}")
                first_model_dir = d_path
                break
        if first_model_dir is None:
            print(f"No model directories found in {base_dir}")
            sys.exit(1)
        first_config = batch_evaluation.load_model_config(first_model_dir)
        first_config['batch_size'] = 1
        if args.val_db_name:
            first_config['val_db_name'] = args.val_db_name
            print(f"Validation dataset: {args.val_db_name}")
        if args.load_keys is not None:
            first_config['load_keys'] = args.load_keys
            print(f"Data keys: {args.load_keys}")
        else:
            print(f"Data keys from config: {first_config.get('load_keys', ['full'])}")
        if args.simulate_data:
            first_config['simulate_data'] = True
            first_config['count'] = args.count
            print(f"Simulated sinograms enabled; target count: {args.count}")
        else:
            first_config['simulate_data'] = False
            print("Using sinograms projected from dataset inputs")
        first_config['solver'] = args.solver
        if isinstance(args.val_sampling_steps, int):
            val_sampling_steps_list = [args.val_sampling_steps]
        else:
            val_sampling_steps_list = args.val_sampling_steps
        print(f"Sampling steps: {val_sampling_steps_list}; solver: {args.solver}")
        config = first_config
        val_db_name = config['val_db_name']
        print(f"Configuration source: {first_model_dir}")
        imaging_system = batch_evaluation.get_imaging_system(config)
        dataset = batch_evaluation.get_val_dataset(config, imaging_system)
        if args.use_loocv:
            val_loader = batch_evaluation.get_val_dataloader_LOOCV(config, dataset,
                                                                  leave_out_case=args.loocv_fold,
                                                                  slices_per_case=args.loocv_slices_per_case)
            print(f"LOOCV dataset size: {len(dataset)} (fold {args.loocv_fold})")
        else:
            val_loader = batch_evaluation.get_val_dataloader(config, dataset)
            print(f"Dataset size: {len(dataset)}")
        print("\nLoading models")
        models = {}
        model_complexity_info = {}
        skip_dirs = {'__pycache__', 'recon', 'batch_evaluation_results', '.git', '.vscode'}
        skip_dirs.update(args.skip_models)
        for method_dir in os.listdir(base_dir):
            if os.path.isfile(os.path.join(base_dir, method_dir)):
                continue
            if method_dir in skip_dirs or not os.path.isdir(os.path.join(base_dir, method_dir)):
                continue
            if args.skip_256 and '256' in method_dir:
                print(f"Skipping {method_dir}: name contains '256'")
                continue
            if args.skip_large and 'large' in method_dir:
                print(f"Skipping {method_dir}: name contains 'large'")
                continue
            if args.only_models and method_dir not in args.only_models:
                print(f"Skipping {method_dir}: not selected")
                continue
            method_path = os.path.join(base_dir, method_dir)
            if args.use_loocv:
                loocv_dir = os.path.join(method_path, f'LOOCV_{args.loocv_fold}')
                checkpoints_dir = os.path.join(loocv_dir, 'checkpoints')
                expected_checkpoint = os.path.join(checkpoints_dir, f'best_checkpoint_fold{args.loocv_fold}.pth')
                if not os.path.exists(loocv_dir):
                    print(f"Skipping {method_dir}: LOOCV_{args.loocv_fold} directory not found")
                    continue
                if not os.path.exists(checkpoints_dir):
                    print(f"Skipping {method_dir}: checkpoints directory not found")
                    continue
                if not os.path.exists(expected_checkpoint):
                    print(f"Skipping {method_dir}: {os.path.basename(expected_checkpoint)} not found")
                    continue
                print(f"Found LOOCV checkpoint: {expected_checkpoint}")
            else:
                checkpoint_path = os.path.join(method_path, 'checkpoint.pth.tar')
                if not os.path.exists(checkpoint_path):
                    print(f"Skipping {method_dir}: checkpoint not found")
                    continue
            try:
                if args.use_loocv:
                    model, model_config = batch_evaluation.load_loocv_model(method_path, args.loocv_fold, device)
                    print(f"Loaded LOOCV model: {method_dir} (fold {args.loocv_fold})")
                else:
                    model, model_config = batch_evaluation.load_model(method_path, device)
                    print(f"Loaded model: {method_dir}")
                models[method_dir] = {'model': model, 'config': model_config}
                if args.analyze_complexity:
                    input_shape = tuple(map(int, args.input_shape.split(',')))
                    complexity_info = analyze_model_complexity(
                        model, method_dir,
                        input_shape=input_shape,
                        device=device
                    )
                    model_complexity_info[method_dir] = complexity_info
            except Exception as e:
                print(f"Failed to load model {method_dir}: {e}")
                continue
        print(f"Loaded {len(models)} model(s)")
        if args.analyze_complexity and model_complexity_info:
            base_save_dir = args.save_dir if args.save_dir else os.path.join(base_dir, "batch_evaluation_results")
            save_root = build_save_dir(
                base_save_dir, data_selection, val_db_name,
                args.start_slice, args.end_slice,
                args.use_loocv, args.loocv_fold if args.use_loocv else None
            )
            os.makedirs(save_root, exist_ok=True)
            complexity_json_path = os.path.join(save_root, "model_complexity.json")
            with open(complexity_json_path, 'w', encoding='utf-8') as f:
                json.dump(model_complexity_info, f, indent=2, ensure_ascii=False)
            print(f"Saved model complexity report: {complexity_json_path}")
        all_sample_scores = {}
        for sampling_steps in val_sampling_steps_list:
            print(f"\n{'='*60}")
            print(f"Sampling steps: {sampling_steps}")
            print(f"{'='*60}")
            config['val_sampling_steps'] = sampling_steps
            print(f"\nEvaluating samples {args.start_idx} through {args.start_idx + num_samples - 1}")
            sample_scores = []
            for i in range(num_samples):
                sample_idx = args.start_idx + i
                if sample_idx >= len(dataset):
                    print(f"Sample index {sample_idx} is outside the dataset; stopping")
                    break
                try:
                    results = batch_evaluation.reconstruct_single_sample(
                        sample_idx, models, imaging_system, config, device, dataset, data_selection
                    )
                    if args.proposed_model in results and 'full' in results:
                        metrics = batch_evaluation.calculate_metrics(results['full'], results[args.proposed_model])
                        if args.metric == 'ssim':
                            score = metrics['ssim']
                        elif args.metric == 'psnr':
                            score = metrics['psnr']
                        elif args.metric == 'rmse':
                            score = metrics['rmse']
                        else:
                            score = metrics['ssim']
                        sample_scores.append({
                            'sample_idx': sample_idx,
                            'proposed_score': score,
                            'metrics': metrics,
                            'results': results
                        })
                        print(f"Sample {sample_idx}: {args.proposed_model} {args.metric.upper()} = {score:.4f}")
                    else:
                        print(f"Sample {sample_idx}: missing {args.proposed_model} or full-dose data")
                except Exception as e:
                    print(f"Failed to process sample {sample_idx}: {e}")
                    continue
            if args.metric_order == 'auto':
                reverse_sort = (args.metric in ['ssim', 'psnr'])
            else:
                reverse_sort = (args.metric_order == 'desc')
            sample_scores.sort(key=lambda x: x['proposed_score'], reverse=reverse_sort)
            best_samples = sample_scores[:top_k]
            metric_values = {}
            for sample_info in sample_scores:
                ground_truth = sample_info['results'].get('full')
                if ground_truth is None:
                    continue
                for model_name, reconstruction in sample_info['results'].items():
                    if model_name in {'ultra_ultra_low', 'full'}:
                        continue
                    values = batch_evaluation.calculate_metrics(ground_truth, reconstruction)
                    model_values = metric_values.setdefault(
                        model_name, {'ssim': [], 'psnr': [], 'rmse': []}
                    )
                    for metric_name, value in values.items():
                        model_values[metric_name].append(value)
            aggregate_metrics = {}
            for model_name, model_values in metric_values.items():
                aggregate_metrics[model_name] = {
                    metric_name: {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                    }
                    for metric_name, values in model_values.items()
                }
                aggregate_metrics[model_name]['count'] = len(model_values['rmse'])
                summary = aggregate_metrics[model_name]
                print(
                    f"{model_name}: SSIM={summary['ssim']['mean']:.4f}±{summary['ssim']['std']:.4f}, "
                    f"PSNR={summary['psnr']['mean']:.2f}±{summary['psnr']['std']:.2f}, "
                    f"RMSE={summary['rmse']['mean']:.4f}±{summary['rmse']['std']:.4f} "
                    f"(n={summary['count']})"
                )
            print(f"\nTop {len(best_samples)} sample(s) at {sampling_steps} sampling steps:")
            for i, sample_info in enumerate(best_samples):
                print(f"Rank {i + 1}: sample {sample_info['sample_idx']}, "
                      f"{args.proposed_model} {args.metric.upper()} = {sample_info['proposed_score']:.4f}")
            all_sample_scores[sampling_steps] = {
                'sample_scores': sample_scores,
                'best_samples': best_samples,
                'aggregate_metrics': aggregate_metrics,
            }
        if 'save_root' not in locals():
            base_save_dir = args.save_dir if args.save_dir else os.path.join(base_dir, "batch_evaluation_results")
            save_root = build_save_dir(
                base_save_dir, data_selection, val_db_name,
                args.start_slice, args.end_slice,
                args.use_loocv, args.loocv_fold if args.use_loocv else None
            )
            os.makedirs(save_root, exist_ok=True)
        for sampling_steps in val_sampling_steps_list:
            best_samples = all_sample_scores[sampling_steps]['best_samples']
            print(f"\nSaving top samples for {sampling_steps} sampling steps to {save_root}")
            for i, sample_info in enumerate(best_samples):
                sample_idx = sample_info['sample_idx']
                print(f"Saving sample {sample_idx} (rank {i + 1})")
                batch_evaluation.save_sample_results(
                    sample_idx, sample_info['results'], save_root,
                    solver=config.get('solver', 'leapfrog'),
                    sampling_steps=sampling_steps
                )
        for sampling_steps in val_sampling_steps_list:
            best_samples = all_sample_scores[sampling_steps]['best_samples']
            sample_scores = all_sample_scores[sampling_steps]['sample_scores']
            metrics_data = {
                'evaluation_info': {
                    'total_samples': len(sample_scores),
                    'best_samples_count': len(best_samples),
                    'data_selection': data_selection,
                    'metric': args.metric,
                    'metric_order': args.metric_order,
                    'models': list(models.keys()),
                    'start_idx': args.start_idx,
                    'num_samples': num_samples,
                    'top_k': top_k,
                    'use_loocv': args.use_loocv,
                    'loocv_fold': args.loocv_fold if args.use_loocv else None,
                    'loocv_slices_per_case': args.loocv_slices_per_case if args.use_loocv else None,
                    'solver': config.get('solver', 'leapfrog'),
                    'sampling_steps': sampling_steps
                },
                'aggregate_metrics': all_sample_scores[sampling_steps]['aggregate_metrics'],
                'best_samples': []
            }
            for sample_info in best_samples:
                sample_metrics = {
                    'sample_idx': sample_info['sample_idx'],
                    'proposed_score': sample_info['proposed_score'],
                    'proposed_metrics': sample_info['metrics'],
                    'selected_metric': args.metric,
                    'all_models_metrics': {}
                }
                if 'full' in sample_info['results']:
                    for model_name, recon in sample_info['results'].items():
                        if model_name not in ['ultra_ultra_low', 'full']:
                            metrics = batch_evaluation.calculate_metrics(sample_info['results']['full'], recon)
                            sample_metrics['all_models_metrics'][model_name] = metrics
                metrics_data['best_samples'].append(sample_metrics)
            json_path = os.path.join(save_root, f"evaluation_metrics_{config.get('solver', 'leapfrog')}_{sampling_steps}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, indent=2, ensure_ascii=False)
            print(f"Saved metrics for {sampling_steps} sampling steps: {json_path}")
        print(f"Evaluation complete. Results saved to {save_root}")
        if args.reconstruct_3d:
            print("\n" + "=" * 60)
            print("3D reconstruction")
            print("=" * 60)
            for sampling_steps in val_sampling_steps_list:
                print(f"\n{'='*60}")
                print(f"3D reconstruction; sampling steps: {sampling_steps}")
                print(f"{'='*60}")
                config['val_sampling_steps'] = sampling_steps
                try:
                    volume_results = batch_evaluation.reconstruct_3d_volume(
                        args.start_slice, args.end_slice, models, imaging_system, config,
                        device, dataset, data_selection
                    )
                    if args.save_nii:
                        dataset_name = val_db_name
                        if args.use_loocv:
                            dataset_name = f"{dataset_name}_loocv_fold_{args.loocv_fold}"
                        batch_evaluation.save_3d_volumes_as_nii(
                            volume_results, save_root, args.start_slice, args.end_slice,
                            dataset_name=dataset_name,
                            simulate_data=args.simulate_data,
                            count=args.count if args.simulate_data else None,
                            solver=config.get('solver', 'leapfrog'),
                            sampling_steps=sampling_steps
                        )
                    print(f"Completed 3D reconstruction with {sampling_steps} sampling steps")
                except Exception as e:
                    print(f"3D reconstruction failed at {sampling_steps} sampling steps: {e}")
                    import traceback
                    traceback.print_exc()
    batch_evaluation.main = custom_main


def analyze_models_complexity_for_notebook(base_dir, device='cuda', input_shape=(1, 1, 128, 128)):
    """Return model-complexity summaries for notebook use."""
    import os
    import json
    from evaluation import load_model
    print("=" * 60)
    print("Model complexity analysis")
    print("=" * 60)
    model_complexity_info = {}
    skip_dirs = {'__pycache__', 'recon', 'batch_evaluation_results', '.git', '.vscode'}
    for method_dir in os.listdir(base_dir):
        if os.path.isfile(os.path.join(base_dir, method_dir)):
            continue
        if method_dir in skip_dirs or not os.path.isdir(os.path.join(base_dir, method_dir)):
            continue
        method_path = os.path.join(base_dir, method_dir)
        checkpoint_path = os.path.join(method_path, 'checkpoint.pth.tar')
        if not os.path.exists(checkpoint_path):
            print(f"Skipping {method_dir}: checkpoint not found")
            continue
        try:
            model, model_config = load_model(method_path, device)
            complexity_info = analyze_model_complexity(
                model, method_dir,
                input_shape=input_shape,
                device=device
            )
            model_complexity_info[method_dir] = complexity_info
        except Exception as e:
            print(f"Failed to analyze model {method_dir}: {e}")
            continue
    print(f"\nAnalyzed {len(model_complexity_info)} model(s)")
    summary_data = []
    for model_name, info in model_complexity_info.items():
        summary_data.append({
            'model_name': model_name,
            'total_params_M': info['parameters']['total_params_M'],
            'trainable_params_M': info['parameters']['trainable_params_M'],
            'flops_G': info['flops']['flops_G'] if 'error' not in info['flops'] else 0,
            'flops_error': info['flops'].get('error', '')
        })
    return {
        'detailed_info': model_complexity_info,
        'summary_data': summary_data
    }

if __name__ == "__main__":
    args = parse_args()
    modify_batch_evaluation_config(args)
    try:
        from evaluation import main
        main()
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
