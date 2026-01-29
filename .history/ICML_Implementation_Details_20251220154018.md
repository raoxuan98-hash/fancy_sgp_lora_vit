# Implementation Details

## Model Architecture

Our method is built upon a Vision Transformer (ViT) backbone with Low-Rank Adaptation (LoRA) modules for parameter-efficient fine-tuning. We use ViT-Base/16 as the default backbone, pre-trained with various strategies including DINO, MAE, CLIP, and MoCoV3. The LoRA adapters are inserted into the attention layers of the ViT, with a default rank of 4, allowing for efficient adaptation to new tasks while preserving the pre-trained knowledge.

### LoRA Parameter Initialization

For standard LoRA implementation, we use two initialization strategies:

1. **Kaiming Uniform Initialization (Default)**:
   - LoRA matrix A: Initialized with Kaiming uniform distribution (a=√5)
   - LoRA matrix B: Initialized with zeros
   - This ensures that the initial LoRA adaptation has zero effect, preserving pre-trained knowledge at the start of training

2. **SVD-based Initialization (Optional)**:
   - LoRA matrix A: Initialized with top-r right singular vectors of the original weight matrix
   - LoRA matrix B: Initialized with zeros
   - This approach captures the principal directions of the original weight space, potentially leading to faster convergence

The LoRA adaptation is applied as: `W' = W + B·A`, where W is the original weight, A∈R^(r×in), and B∈R^(out×r), with r being the LoRA rank (default: 4).

### LoRA Variants

We implement several LoRA variants:
- **Basic LoRA**: Standard low-rank adaptation with rank-4 matrices
- **SGP LoRA**: Incorporates symmetric Gaussian projection with temperature scaling
- **NSP LoRA**: Implements negative subspace projection with regularization
- **Full LoRA**: Full parameter fine-tuning for comparison
- **Joint LoRA**: Joint training across all tasks

### Full Fine-Tuning Implementation

For the "Full LoRA" and "Full_NSP" variants, we implement selective full fine-tuning with the following strategy:

1. **Parameter Selection**:
   - **Primary targets**: All parameters in the MLP (FFN) modules of selected transformer blocks
   - **Secondary targets**: LayerNorm weights (when `include_norm=True`)
   - **Excluded parameters**: All bias terms and patch embedding parameters (by default)

2. **Layer-wise Strategy**:
   - By default, all transformer blocks are fine-tuned (can be limited to specific blocks)
   - Within each block, only the FFN layers (fc1 and fc2) are fully trainable
   - Attention layers (qkv projections) remain frozen to preserve pre-trained knowledge

3. **Parameter Efficiency**:
   - Approximately 15-20% of total model parameters are trainable
   - This provides a balance between adaptation capability and computational efficiency
   - Significantly more parameters than LoRA variants but fewer than complete model fine-tuning

4. **Training Configuration**:
   - Lower learning rate (5×10⁻⁶) compared to LoRA variants
   - Same optimization strategy (AdamW with cosine scheduling)
   - Weight decay of 3×10⁻⁵ applied to all trainable parameters

This selective fine-tuning approach allows for more substantial model adaptation than LoRA while maintaining reasonable computational efficiency and avoiding catastrophic forgetting of the pre-trained attention mechanisms.

### Null-Space Projection (NSP) for Full Fine-Tuning

For the "Full_NSP" variant, we implement null-space projection (NSP) on the full fine-tuning parameters to prevent catastrophic forgetting while maintaining the adaptation capacity of full parameter updates.

#### 1. NSP Mathematical Foundation

The null-space projection operates on the gradient space during backpropagation. For each trainable weight matrix W, we compute a projection matrix P that projects gradients onto the null-space of the feature covariance:

```
∇W_projected = ∇W · P
```

where P is constructed from the eigendecomposition of the feature covariance matrix Σ:

```
Σ = VΛV^T
P = V_null · V_null^T
```

The null-space consists of eigenvectors corresponding to the smallest eigenvalues, representing directions that minimally affect the learned representations.

#### 2. Projection Matrix Construction

We use two variants of projection matrix construction:

**Hard Projection (Default for Full_NSP)**:
- Threshold ε (default: 0.05) determines the cumulative eigenvalue ratio to preserve
- Eigenvectors with cumulative eigenvalue ratio < ε are selected for the null-space
- Final projection: `P = (1 - w) · V_null · V_null^T + w · I`
  - w (nsp_weight) controls interpolation between pure null-space and identity (default: 0.02)
  - I is the identity matrix

**Soft Projection (Optional)**:
- Applies weighted scaling to all eigenvectors based on their eigenvalues
- Weight function: `weights = 1 / (1 + β · log(1 + λ^p))`
  - β (weight_temp) controls the sharpness of the weighting (default: 5.0)
  - p (weight_p) controls the polynomial degree (default: 1.0)

#### 3. Implementation Details

**Parameter Selection**:
- Target modules: All FFN (fc1, fc2) weights in selected transformer blocks
- Excluded: All bias terms, attention layers, and patch embeddings
- Approximately 15-20% of total model parameters are trainable with NSP applied

**Gradient Projection Hook**:
- Registered to each trainable parameter during initialization
- Automatically projects gradients during backpropagation
- Computationally efficient with O(d²) complexity per layer (d: feature dimension)

**Covariance Calculation**:
- Feature covariance computed from forward activations
- Online accumulation during training for efficiency
- Regularized with ε = 1e-6 for numerical stability

#### 4. Training Configuration

**Hyperparameters**:
- Learning rate: 5×10⁻⁶ (lower than basic LoRA due to higher parameter count)
- NSP threshold (nsp_eps): 0.05 (default), controls null-space size
- NSP weight (nsp_weight): 0.02 (default), controls projection strength
- Optimizer: AdamW with weight decay 3×10⁻⁵

**Training Procedure**:
1. Initialize model with frozen pre-trained weights
2. Unfreeze only FFN weights in selected layers
3. Register gradient projection hooks
4. Compute feature covariances on current task data
5. Update projection matrices before each task
6. Train with projected gradients

#### 5. Advantages over Standard Full Fine-Tuning

1. **Forgetting Prevention**: By projecting gradients to the null-space, we minimize interference with previously learned knowledge
2. **Stability**: NSP provides regularization effect, leading to more stable training across tasks
3. **Efficiency**: Maintains most of the adaptation capacity of full fine-tuning while reducing catastrophic forgetting
4. **Flexibility**: The nsp_weight parameter allows smooth interpolation between full fine-tuning and pure NSP

#### 6. Memory and Computational Considerations

**Memory Overhead**:
- Projection matrices: O(d²) per layer (typically < 2MB per layer)
- Covariance matrices: O(d²) per layer (computed on-the-fly)
- Total overhead: ~5-10% increase over standard full fine-tuning

**Computational Cost**:
- Projection operation: O(d²) per backward pass
- Eigenvalue decomposition: O(d³) per task (infrequent operation)
- Overall training time: ~10-15% increase over standard full fine-tuning

This implementation provides a practical balance between the adaptation capacity of full fine-tuning and the stability of parameter-efficient methods like LoRA, making it particularly suitable for scenarios where substantial model adaptation is required without catastrophic forgetting.

## Training Procedure

### Optimization

We use AdamW optimizer with weight decay of 3×10⁻⁵ for most experiments. The learning rate is set according to the LoRA type:
- Basic/SGP/NSP LoRA: 1×10⁻⁴
- Full/Full_NSP LoRA: 5×10⁻⁶
- Joint_Full LoRA: 1×10⁻⁵

We employ a cosine learning rate schedule with 10% warmup steps. The batch size is set to 16 for all experiments.

### Knowledge Distillation

We implement feature-level knowledge distillation to mitigate catastrophic forgetting:
- Distillation weight (γ_kd): 0.5 for NSP variants, 0.0 for others
- Distillation type: Feature-based (feat) or cosine similarity (cos)
- Transform: Identity, linear, or weak nonlinear
- Teacher network updates after each task (except the first task to avoid initial KD loss)

### Loss Function

We use symmetric cross-entropy (SCE) loss with equal weights (α=0.5, β=0.5) to balance between standard cross-entropy and reverse cross-entropy, improving robustness to label noise.

## Data Configuration

### Datasets

**Within-Domain Experiments:**
- CIFAR-100: 10 initial classes, 10 classes per increment
- CUB200: 20 initial classes, 20 classes per increment
- Cars196: 20 initial classes, 20 classes per increment
- ImageNet-R: 20 initial classes, 20 classes per increment

**Cross-Domain Experiments:**
- Default datasets: CIFAR-100, ImageNet-R, Cars196, CUB200, Caltech-101, Oxford-Flower-102, Food-101
- Number of shots: 64 samples per class for few-shot learning
- Incremental splits: 2 splits per dataset (when enabled)

### Data Preprocessing

All images are resized to 224×224 and normalized using ImageNet statistics:
- Mean: [0.485, 0.456, 0.406]
- Std: [0.229, 0.224, 0.225]

## Distribution Compensation

We implement multiple distribution compensation strategies to address feature drift:

### Compensation Types

1. **SeqFT**: Sequential fine-tuning without compensation (baseline)
2. **SeqFT + Linear**: Linear transformation compensation with γ=0.1
3. **SeqFT + WeakNonlinear**: Two-layer MLP compensation
4. **SeqFT + Hopfield**: Hopfield attention-based compensation
   - Temperature: 0.05
   - Top-k: 400
5. **SeqFT + RFF**: Random Fourier Features compensation

### Feature Combination

We support three feature combination modes:
- **Combined**: Mix current and auxiliary data
- **Aux Only**: Use only auxiliary data for compensation
- **Current Only**: Use only current task data

### Auxiliary Data

We use ImageNet-1K training set as auxiliary data by default, sampling 2048 images per epoch. Other auxiliary datasets include CIFAR-10, SVHN, and Flickr8K.

## Classifier Construction

We build multiple classifier types from the compensated feature distributions:

### Gaussian Classifiers

1. **LDA (Linear Discriminant Analysis)**:
   - Regularization α: 0.1
   - Assumes equal covariance matrices across classes

2. **QDA (Quadratic Discriminant Analysis)**:
   - Regularization α₁: 0.2 (for covariance matrix)
   - Regularization α₂: 2.0 (for diagonal loading)
   - Regularization α₃: 0.5 (for final regularization)

### SGD Classifier

- Learning rate: 10× base learning rate
- Momentum: 0.9
- Trained on compensated features

## Evaluation Protocol

### Metrics

We report multiple evaluation metrics:
- **Data-wise accuracy**: Overall accuracy across all test samples
- **Task-wise accuracy**: Average accuracy per task
- **Class-wise accuracy**: Average accuracy per class
- **Cumulative accuracy**: Performance on all tasks seen so far

### Statistical Analysis

All results are averaged over 3 random seeds (1993, 1996, 1997) with mean and standard deviation reported. We perform statistical analysis to compare different methods and identify significant differences.

## Implementation Details

### Hardware

All experiments are conducted on NVIDIA GPUs with CUDA support. We monitor GPU memory usage throughout training and report peak memory consumption.

### Software

Our implementation is based on PyTorch and includes:
- Custom LoRA implementation with various projection types
- Distribution compensators with different transformation strategies
- Efficient Gaussian statistics computation and storage
- Multi-threaded data loading with persistent workers

### Reproducibility

We ensure reproducibility by:
- Setting random seeds for Python, NumPy, and PyTorch
- Using deterministic CuDNN algorithms
- Logging all hyperparameters and random seeds
- Saving model checkpoints after each task

## Hyperparameter Sensitivity

We conducted extensive ablation studies on key hyperparameters:

### LoRA Rank

We tested ranks {2, 4, 8} and found rank 4 provides the best balance between performance and efficiency.

### Compensation Temperature

For Hopfield compensation, we tested temperatures {0.01, 0.05, 0.1} and found 0.05 optimal.

### Regularization Strength

We systematically varied LDA and QDA regularization parameters and selected values that maximize validation performance while maintaining numerical stability.

## Computational Complexity

### Training Time

- Within-domain: 10-60 minutes per task depending on dataset size
- Cross-domain: 5-30 minutes per task depending on number of shots

### Memory Usage

- Peak GPU memory: 8-12 GB depending on model and dataset
- Additional memory for compensation: 1-2 GB

### Parameter Count

- LoRA parameters: ~0.1% of total model parameters
- Compensation parameters: Negligible (<0.01% of total)
- Total trainable parameters: ~0.15% of full model

## Limitations and Future Work

Our approach has several limitations:
- Requires careful tuning of regularization parameters
- Compensation methods add computational overhead
- Performance varies significantly across domains

Future work will focus on:
- Automatic hyperparameter tuning
- More efficient compensation strategies
- Domain adaptation techniques for cross-domain scenarios