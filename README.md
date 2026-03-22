# CS 189 Final Project — Bacteria Classifier

Binary classification of E. coli vs. S. aureus microscopy images using ResNet18 and a custom CNN, evaluated across two different lab-generated datasets. Refer to `CS189 Final Project Report, XU MARK.pdf` for full details and results.

## Models

- `Final_Roboflow.py`: ResNet18 (pretrained) with the Roboflow dataset
- `Final_DIBaS.py`: ResNet18 (pretrained) with the Augmented DIBaS dataset
- `simple_model.py`: Custom 2-layer CNN with both datasets

## Datasets

- Roboflow (`Microbes/`): 243 E. coli + 309 S. aureus images from a single consistent lab setting
- DIBaS (not included, ~4GB): Originally 20 images/class, augmented to 520/class. Original dataset no longer publically available. Download with [DIBaS](https://drive.google.com/drive/folders/1TGokBwYZmpb2lw_ukeRWrR7Q49MNeWdO?usp=drive_link), then move it into the project root:
  ```bash
  mv DIBaS/ cs189-bacteria-classifier/
  ```

## Setup

```bash
pip install torch torchvision scikit-learn matplotlib seaborn opencv-python pillow
```

## Usage

Run from the project root directory:

```bash
python Final_Roboflow.py   # evaluate ResNet18 trained on Roboflow
python Final_DIBaS.py      # evaluate ResNet18 trained on DIBaS
python simple_model.py     # evaluate Simple CNN
```
