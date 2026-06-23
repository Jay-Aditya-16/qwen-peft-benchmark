# Benchmarking LoRA, DoRA, QLoRA, and QDoRA on Qwen2.5-3B Using a Google Colab T4

## Introduction

Parameter-Efficient Fine-Tuning (PEFT) methods have become the standard approach for adapting large language models without retraining billions of parameters. While LoRA remains the most widely adopted technique, newer methods such as DoRA and their quantized variants promise improved adaptation quality with minimal overhead.

To understand the practical trade-offs between these methods, I benchmarked LoRA, DoRA, QLoRA, and QDoRA on Qwen2.5-3B-Instruct using a single NVIDIA T4 GPU available through Google Colab.

The goal was simple:

**Which method provides the best balance of quality, memory efficiency, and training speed on limited hardware?**

---

## Experimental Setup

### Hardware

* NVIDIA T4 GPU (16 GB VRAM)
* Google Colab

### Model

* Qwen2.5-3B-Instruct

### Dataset

* Alpaca Cleaned Dataset
* 300 training samples
* 80 validation samples

### Training Configuration

* Rank (r): 16
* LoRA Alpha: 32
* Dropout: 0.05
* Learning Rate: 2e-4
* Max Sequence Length: 64
* Training Steps: 80
* Gradient Accumulation: 8

### Methods Compared

#### LoRA

Low-Rank Adaptation freezes the original model weights and learns low-rank update matrices.

#### DoRA

Weight-Decomposed Low-Rank Adaptation extends LoRA by separately modeling weight magnitude and direction, aiming to better mimic full fine-tuning.

#### QLoRA

Combines LoRA with a 4-bit NF4 quantized base model, dramatically reducing memory requirements.

#### QDoRA

Applies DoRA on top of a quantized NF4 backbone.

---

## Results

| Method | Eval Loss ↓ | Perplexity ↓ | Peak VRAM ↓ | Training Time ↓ |
| ------ | ----------- | ------------ | ----------- | --------------- |
| LoRA   | 1.3384      | 3.81         | 9201 MB     | 419.5 s         |
| DoRA   | 1.3401      | 3.82         | 9935 MB     | 1425.6 s        |
| QLoRA  | 1.3645      | 3.91         | 6482 MB     | 603.0 s         |
| QDoRA  | 1.3670      | 3.92         | 6624 MB     | 1697.0 s        |

---

## Analysis

### 1. LoRA Delivered the Best Overall Quality

LoRA achieved the lowest validation loss and perplexity across all methods tested.

Although DoRA was expected to outperform LoRA based on published literature, the observed difference was extremely small:

* LoRA: 1.3384
* DoRA: 1.3401

The gap of only 0.0017 is effectively negligible for this experiment.

---

### 2. QLoRA Preserved Most of LoRA's Performance

QLoRA reduced memory consumption significantly while maintaining comparable validation performance.

Compared to LoRA:

* Validation loss increased by only ~1.9%
* VRAM usage dropped from 9.2 GB to 6.5 GB

This demonstrates why QLoRA has become the industry standard for fine-tuning models on consumer GPUs.

---

### 3. DoRA Introduced Significant Training Overhead

While DoRA matched LoRA's quality, training time increased substantially.

| Method | Time   |
| ------ | ------ |
| LoRA   | 419 s  |
| DoRA   | 1426 s |

DoRA required over 3× more training time while providing no measurable improvement under these training conditions.

---

### 4. QDoRA Was the Most Expensive Configuration

QDoRA combined the computational overhead of DoRA with quantized training.

Although memory usage remained low, training time became the highest among all methods tested.

For this experiment, QDoRA offered neither a quality advantage nor a meaningful efficiency benefit.

---

## Memory Efficiency

The most significant finding emerged from memory usage.

| Method | Peak VRAM |
| ------ | --------- |
| LoRA   | 9.2 GB    |
| DoRA   | 9.9 GB    |
| QLoRA  | 6.5 GB    |
| QDoRA  | 6.6 GB    |

QLoRA reduced memory requirements by approximately:

29.5%

while preserving nearly all of LoRA's adaptation quality.

For researchers and builders working with T4s, RTX 3060s, RTX 4060s, or laptop GPUs, this reduction can be the difference between a successful training run and an out-of-memory crash.

---

## Limitations

This benchmark intentionally used a small-scale configuration:

* 300 training samples
* 80 validation samples
* 80 optimization steps
* Sequence length of 64 tokens

As a result, the benchmark primarily measures:

* Memory efficiency
* Training speed
* Early-stage adaptation behavior

rather than the full convergence characteristics of each method.

Previous DoRA research suggests that its advantages become more visible with:

* Larger datasets
* Longer training schedules
* More challenging reasoning tasks

Future experiments with 2,000+ samples and 500+ training steps may reveal stronger differences between LoRA and DoRA.

---

## Conclusion

For fine-tuning Qwen2.5-3B on a T4 GPU:

**Best Quality:** LoRA

**Best Memory Efficiency:** QLoRA

**Best Overall Trade-off:** QLoRA

**Most Computationally Expensive:** QDoRA

The key takeaway is that QLoRA preserved nearly all of LoRA's performance while reducing VRAM usage by roughly 30%, making it the most practical choice for resource-constrained environments.

While DoRA and QDoRA remain promising techniques, their benefits were not evident under short-run training conditions.

For most developers fine-tuning models on commodity hardware today, QLoRA continues to offer the strongest balance between quality, speed, and memory efficiency.
