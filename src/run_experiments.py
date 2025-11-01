"""
FILE: run_experiments.py
Main script to run FedHoeffCrypt experiments
Save this as: run_experiments.py

This script runs experiments on:
- CICIDS2017
- CICIDS2018
- EdgeIIoT

With and without watermarking to compare performance impact
"""

import os
import sys
import numpy as np
from fedhoeffcrypt_integration import FedHoeffCryptExperiment


def run_experiment_with_config(config):
    """
    Run a single experiment with given configuration
    
    Args:
        config: Dictionary with experiment configuration
    """
    print("\n" + "="*80)
    print(f"RUNNING EXPERIMENT: {config['name']}")
    print("="*80)
    
    experiment = FedHoeffCryptExperiment(
        dataset_name=config['dataset_name'],
        n_clients=config['n_clients'],
        clients_per_round=config['clients_per_round'],
        n_rounds=config['n_rounds'],
        n_epochs=config['n_epochs'],
        watermark_enabled=config['watermark_enabled'],
        trigger_size=config['trigger_size'],
        random_state=config['random_state']
    )
    
    # Run experiment
    experiment.run_complete_experiment(config['file_path'])
    
    # Save with specific name
    results_file = f"results_{config['dataset_name']}_{config['exp_type']}.json"
    plots_file = f"plots_{config['dataset_name']}_{config['exp_type']}.png"
    
    experiment.save_results(results_file)
    experiment.plot_training_curves(plots_file)
    
    print(f"\n✓ Experiment '{config['name']}' completed!")
    print(f"  Results saved to: {results_file}")
    print(f"  Plots saved to: {plots_file}")


def run_quick_test():
    """
    Run a quick test with synthetic data (5 rounds, small dataset)
    """
    print("\n" + "="*80)
    print("QUICK TEST MODE - Using Synthetic Data")
    print("="*80)
    
    # Create synthetic dataset
    print("\nCreating synthetic IDS dataset...")
    from data_preprocessing import IDSDataPreprocessor
    
    np.random.seed(42)
    n_samples = 2000
    n_features = 20
    n_classes = 5
    
    # Generate synthetic data
    X = np.random.randn(n_samples, n_features)
    for i in range(n_classes):
        mask = np.random.choice(n_samples, size=n_samples//n_classes, replace=False)
        X[mask, i*2:(i+1)*2] += np.random.randn(len(mask), 2) * 3
    
    y = np.random.randint(0, n_classes, n_samples)
    
    # Create DataFrame
    import pandas as pd
    feature_cols = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_cols)
    
    attack_types = ['Normal', 'DDoS', 'PortScan', 'BruteForce', 'Infiltration']
    df['Label'] = [attack_types[i % len(attack_types)] for i in y]
    
    # Save to CSV
    test_file = 'synthetic_test_data.csv'
    df.to_csv(test_file, index=False)
    print(f"✓ Synthetic dataset created: {test_file}")
    
    # Run quick experiment
    config = {
        'name': 'Quick Test - Synthetic Data',
        'dataset_name': 'Synthetic',
        'file_path': test_file,
        'n_clients': 5,
        'clients_per_round': 3,
        'n_rounds': 5,  # Only 5 rounds for quick test
        'n_epochs': 3,  # Only 3 epochs for quick test
        'watermark_enabled': True,
        'trigger_size': 50,
        'random_state': 42,
        'exp_type': 'watermarked'
    }
    
    run_experiment_with_config(config)
    
    print("\n" + "="*80)
    print(" QUICK TEST COMPLETED!")
    print("="*80)
    print("\nIf this worked, you can now run full experiments with real datasets.")


def run_full_experiments():
    """
    Run complete experiments on all datasets
    """
    print("\n" + "="*80)
    print("FULL EXPERIMENTS MODE")
    print("="*80)
    
    # Define experiment configurations
    experiments = [
        # CICIDS2017 with watermarking
        {
            'name': 'CICIDS2017 with Watermarking',
            'dataset_name': 'CICIDS2017',
            'file_path': '../data/CICIDS2017.csv',
            'n_clients': 10,
            'clients_per_round': 5,
            'n_rounds': 20,  # Reduced for faster testing
            'n_epochs': 5,   # Reduced for faster testing
            'watermark_enabled': True,
            'trigger_size': 100,
            'random_state': 42,
            'exp_type': 'watermarked'
        },
        # CICIDS2017 without watermarking
        {
            'name': 'CICIDS2017 without Watermarking',
            'dataset_name': 'CICIDS2017',
            'file_path': '../data/CICIDS2017.csv',
            'n_clients': 10,
            'clients_per_round': 5,
            'n_rounds': 20,  # Reduced for faster testing
            'n_epochs': 5,   # Reduced for faster testing
            'watermark_enabled': False,
            'trigger_size': 100,
            'random_state': 42,
            'exp_type': 'no_watermark'
        }
    ]
    
    # Check which datasets are available
    available_experiments = []
    for exp in experiments:
        if os.path.exists(exp['file_path']):
            available_experiments.append(exp)
        else:
            print(f" Skipping {exp['name']} - file not found: {exp['file_path']}")
    
    if not available_experiments:
        print("\n No dataset files found!")
        print("\nPlease download datasets and place them in the 'data/' folder:")
        print("   data/CICIDS2017.csv")
        print("   data/CICIDS2018.csv")
        print("   data/EdgeIIoT.csv")
        print("\nOr run quick test with synthetic data: python run_experiments.py --quick")
        return
    
    # Run available experiments
    print(f"\nFound {len(available_experiments)} available experiments")
    
    for i, exp in enumerate(available_experiments):
        print(f"\n{'='*80}")
        print(f"Running experiment {i+1}/{len(available_experiments)}")
        print(f"{'='*80}")
        run_experiment_with_config(exp)
    
    print("\n" + "="*80)
    print(" ALL EXPERIMENTS COMPLETED!")
    print("="*80)


def compare_watermark_impact():
    """
    Compare results with and without watermarking
    """
    print("\n" + "="*80)
    print("WATERMARK IMPACT ANALYSIS")
    print("="*80)
    
    import json
    
    datasets = ['CICIDS2017', 'CICIDS2018', 'EdgeIIoT']
    
    for dataset in datasets:
        watermarked_file = f"results_{dataset}_watermarked.json"
        no_watermark_file = f"results_{dataset}_no_watermark.json"
        
        if os.path.exists(watermarked_file) and os.path.exists(no_watermark_file):
            print(f"\n{dataset} Comparison:")
            print("-" * 60)
            
            # Load results
            with open(watermarked_file, 'r') as f:
                watermarked = json.load(f)
            
            with open(no_watermark_file, 'r') as f:
                no_watermark = json.load(f)
            
            # Get final metrics
            wm_acc = watermarked['final_metrics']['accuracy']
            no_wm_acc = no_watermark['final_metrics']['accuracy']
            
            wm_prec = watermarked['final_metrics']['precision']
            no_wm_prec = no_watermark['final_metrics']['precision']
            
            wm_rec = watermarked['final_metrics']['recall']
            no_wm_rec = no_watermark['final_metrics']['recall']
            
            wm_f1 = watermarked['final_metrics']['f1_score']
            no_wm_f1 = no_watermark['final_metrics']['f1_score']
            
            print(f"Accuracy:  With WM: {wm_acc:.4f} | Without WM: {no_wm_acc:.4f} | Δ: {abs(wm_acc-no_wm_acc):.4f}")
            print(f"Precision: With WM: {wm_prec:.4f} | Without WM: {no_wm_prec:.4f} | Δ: {abs(wm_prec-no_wm_prec):.4f}")
            print(f"Recall:    With WM: {wm_rec:.4f} | Without WM: {no_wm_rec:.4f} | Δ: {abs(wm_rec-no_wm_rec):.4f}")
            print(f"F1-Score:  With WM: {wm_f1:.4f} | Without WM: {no_wm_f1:.4f} | Δ: {abs(wm_f1-no_wm_f1):.4f}")
            
            # Calculate average performance loss
            avg_loss = (abs(wm_acc-no_wm_acc) + abs(wm_prec-no_wm_prec) + 
                       abs(wm_rec-no_wm_rec) + abs(wm_f1-no_wm_f1)) / 4
            print(f"\nAverage Performance Impact: {avg_loss:.4f} ({avg_loss*100:.2f}%)")
        else:
            print(f"\n {dataset}: Results not found")


def main():
    """
    Main function to handle command line arguments
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Run FedHoeffCrypt Experiments')
    parser.add_argument(
        '--mode',
        choices=['quick', 'full', 'compare'],
        default='quick',
        help='Experiment mode: quick (test), full (all datasets), compare (analyze results)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        run_quick_test()
    elif args.mode == 'full':
        run_full_experiments()
    elif args.mode == 'compare':
        compare_watermark_impact()


if __name__ == "__main__":
    # If no arguments, show help
    if len(sys.argv) == 1:
        print("="*80)
        print("FedHoeffCrypt Experiment Runner")
        print("="*80)
        print("\nUsage:")
        print("  py -3.10 -X utf8 run_experiments.py --mode quick     # Quick test with synthetic data")
        print("  py -3.10 -X utf8 run_experiments.py --mode full      # Run all experiments")
        print("  py -3.10 -X utf8 run_experiments.py --mode compare   # Compare watermark impact")
        print("\nOr use the batch script:")
        print("  run_experiments.bat quick")
        print("  run_experiments.bat full")
        print("  run_experiments.bat compare")
        print("\nDefault mode is 'quick'")
        print("\nRunning quick test...")
        run_quick_test()
    else:
        main()