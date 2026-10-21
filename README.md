# Oxford-IIIT Pet Breed Classification 

A convolutional neural network trained **from scratch** (no pretrained weights) to classify images
from the [Oxford-IIIT Pet dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) into 37 cat and dog breeds.

Module: Intelligent Systems: Machine Learning and Optimisation (COM00026I), University of York.

## Results

| Split | Accuracy |
|-------|----------|
| Train | 100.00 % |
| Test (official `test` split, 3669 images) | 65.79 % |


Train accuracy is measured by `train.py` with eval transforms (no random augmentation)
on the training set described below.

## Model: `PetResNet`

A ResNet-18-style network written from scratch in PyTorch, inspired by
He et al., 2015, *Deep Residual Learning for Image Recognition*.

- **Stem:** Conv 7×7 (64, stride 2) → BatchNorm → ReLU → MaxPool 3×3 (stride 2)
- **4 stages × 2 residual blocks:** 64 → 128 → 256 → 512 channels
  - each block: Conv 3×3 → BN → ReLU → Conv 3×3 → BN, plus a skip connection
    (1×1 Conv + BN when the shape changes), then ReLU
- **Head:** Global average pooling → Flatten → Dropout(0.2) → Linear(512 → 37)
- Kaiming-normal initialisation for convolutions
- **Total trainable parameters: 11,195,493**
- Input: 224×224 RGB

## Training setup

| Setting | Value |
|---------|-------|
| Loss | CrossEntropyLoss (label smoothing 0.1) |
| Optimiser | SGD, momentum 0.9, Nesterov, weight decay 1e-4 |
| Learning rate | 1e-1; 3-epoch linear warmup, then cosine decay to 0 |
| Epochs | 30 |
| Batch size | 64 |
| Random seed | 42 |

### Data augmentation

Training uses the official `trainval` split (3680 images) only.

**Offline step.** Each original image is saved in 3 versions in `./augmented_data/`,
giving 11,040 training images:
1. original, resized to 224×224
2. Resize + RandomHorizontalFlip + ColorJitter (brightness/contrast/saturation 0.4, hue 0.1)
3. Resize + RandomHorizontalFlip + RandomRotation(15) + RandomAffine (translate 0.1, scale 0.85–1.15)

**Online step (every epoch).** RandomHorizontalFlip, ColorJitter (0.2), Normalize,
RandomErasing (p = 0.2).

**Test-time preprocessing:** only `Resize((224, 224))`, `ToTensor()` and `Normalize`
(standard ImageNet mean/std values; no pretrained weights are used). No cropping or test-time augmentation.

## Repository structure

```
.
├── train.py          # builds the augmented training set, trains, saves model.pth
├── test.py           # loads model.pth, reports accuracy on the official test split
├── model.pth         # trained weights (~45 MB)
├── environment.yml   # conda environment
└── README.md
```

## How to run

```bash
conda env create -f environment.yml
conda activate imlo-coursework

python train.py   # downloads the dataset to ./data, creates ./augmented_data, trains 30 epochs
python test.py    # evaluates model.pth on the test split
```

The dataset is downloaded automatically via `torchvision.datasets.OxfordIIITPet`.
A GPU is strongly recommended: CPU training takes several hours.

## Notes

- Only the official Oxford-IIIT Pet data is used: no external images and no pretrained weights.
- Results can vary by about ±2–3% between training runs because of randomness.
