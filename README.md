# FedHoeffCrypt Experiment Guide

## 🚀 How to Run Experiments

### Option 1: Using Python 3.10 directly
```bash
# Quick test with synthetic data
py -3.10 -X utf8 run_experiments.py --mode quick

# Full experiments with real datasets
py -3.10 -X utf8 run_experiments.py --mode full

# Compare watermark impact
py -3.10 -X utf8 run_experiments.py --mode compare
```

### Option 2: Using the batch script (Windows)
```bash
# Quick test
run_experiments.bat quick

# Full experiments
run_experiments.bat full

# Compare results
run_experiments.bat compare
```

## 📊 Experiment Results

The system now successfully:
- ✅ **Loads and preprocesses data** (CICIDS2017/2018, EdgeIIoT, or synthetic)
- ✅ **Initializes CKKS encryption** with real TenSEAL library
- ✅ **Sets up federated learning** with multiple clients
- ✅ **Trains models** with homomorphic encryption
- ✅ **Embeds watermarks** for model ownership verification
- ✅ **Evaluates performance** with comprehensive metrics
- ✅ **Saves results** to JSON files without serialization errors

## 🔧 System Requirements

- **Python 3.10** (for TenSEAL compatibility)
- **Required packages**: numpy, pandas, scikit-learn, tenseal
- **Optional**: matplotlib (for plotting training curves)

## 📁 Output Files

After running experiments, you'll find:
- `results_[dataset]_[type].json` - Experiment results
- `plots_[dataset]_[type].png` - Training curves (if matplotlib installed)
- `fedhoeffcrypt_results.json` - Detailed results
- `synthetic_test_data.csv` - Generated test data

## 🎯 Next Steps

1. **Test with real datasets**: Place CICIDS2017/2018 or EdgeIIoT CSV files in the `data/` folder
2. **Run full experiments**: Use `--mode full` for complete 60-round training
3. **Analyze results**: Use `--mode compare` to compare watermarked vs non-watermarked models
4. **Customize parameters**: Modify experiment configurations in `run_experiments.py`

## 🔒 Security Features

- **CKKS Homomorphic Encryption**: Real cryptographic security with TenSEAL
- **Dynamic Watermarking**: Model ownership verification
- **Federated Learning**: Privacy-preserving distributed training
- **Trigger Set Protection**: Secure watermark verification

Your FedHoeffCrypt system is now fully operational! 🎉