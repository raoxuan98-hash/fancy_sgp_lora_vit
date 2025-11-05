# Parameter Grid Experiments for SGP LoRA

## Overview
Modified `sh/main_experiments_full_method.sh` to systematically test different combinations of `weight_temp` and `weight_p` parameters across all datasets.

## Parameters Tested

### weight_temp
- 1.0
- 2.0  
- 4.0

### weight_p
- 1.0
- 2.0

## Datasets
- cifar100_224
- imagenet-r
- cub200_224
- cars196_224

## Seeds
- 1993
- 1996
- 1997

## Total Experiments
3 (weight_temp) × 2 (weight_p) × 4 (datasets) × 3 (seeds) = **72 experiments**

## Key Changes Made

1. **Added parameter arrays:**
   ```bash
   WEIGHT_TEMP_VALUES=(1.0 2.0 4.0)
   WEIGHT_P_VALUES=(1.0 2.0)
   ```

2. **Modified loop structure** to create nested loops for all parameter combinations

3. **Updated log naming scheme** to include both parameters:
   - Format: `{dataset}_temp{weight_temp}_p{weight_p}_seed{seed}.log`
   - Example: `cifar100_224_temp1.0_p1.0_seed1993.log`

4. **Enhanced GPU assignment** to distribute experiments across available GPUs

5. **Added experiment summary** showing total combinations

## Usage
Run the modified script:
```bash
bash sh/main_experiments_full_method.sh
```

## Expected Output Structure
```
logs/full_method_YYYYMMDD_HHMMSS/
├── run_cifar100_224_temp1.0_p1.0.sh
├── run_cifar100_224_temp1.0_p2.0.sh
├── run_cifar100_224_temp2.0_p1.0.sh
