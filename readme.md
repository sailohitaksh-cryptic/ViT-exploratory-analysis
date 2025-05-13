# Mosquito Species Classification Project

## 1. Project Overview
This project focuses on developing a deep learning model to accurately classify mosquito species from images. The primary goal is to leverage computer vision techniques to aid in entomological studies and vector control efforts, given that different mosquito species are vectors for different diseases. The project involved extensive data preprocessing, exploration of various image augmentation and class imbalance strategies, and experimentation with several state-of-the-art vision transformer models.

The project was developed across two primary Jupyter notebooks, allowing for different experimental paths:
1.  **Data Preprocessing & Initial Model Exploration Notebook:** Focused on initial model training with standard preprocessing. Contains all necessary data loading and cleaning steps.
2.  **Advanced Preprocessing & Model Training Notebook:** This notebook specifically implements and tests more advanced image preprocessing techniques, such as CLAHE + Otsu cropping for mosquito isolation, alongside model training and evaluation. Also contains all necessary data loading and cleaning steps to run independently.
This structure allowed for comparative analysis of different approaches.

## 2. Dataset
* **Source:** The initial dataset was provided as a CSV file (`data_export.csv`) containing image URLs and metadata.
* **Initial State:** The dataset comprised approximately 2000+ image records with details like `specimenId`, `morphSpecies`, `species`, `title`, and `country`.
* **Target Classes:** The goal was to classify mosquitoes into the following 6 categories:
    1.  Anopheles Funestus
    2.  Anopheles Gambiae
    3.  Anopheles Other
    4.  Culex
    5.  Mansonia
    6.  Other (miscellaneous species)

## 3. Data Preprocessing and Cleaning (Foundational steps in both Notebooks)
Essential data loading and initial preprocessing steps are performed in both notebooks to ensure they can function independently up to the point of model-specific preparations. These common steps include:
1.  **Loading Data:** `data_export.csv` loaded into a pandas DataFrame.
2.  **Handling Missing Values:**
    * `morphSpecies` values were filled using the `species` column where `morphSpecies` was initially empty.
    * NaN values in `title` and `country` were filled with empty strings to prevent errors during string operations.
3.  **Filtering:**
    * Rows with "test" in the `title` were removed.
    * Samples from "USA" and "India" were removed (India samples were noted to not always be mosquitoes).
4.  **Image Downloading:**
    * A script (typically run once and then images are accessed locally) was used to download images from URLs specified in the dataset, saving them locally using their `specimenId`.
5.  **Label Standardization:**
    * The `morphSpecies` column was standardized (e.g., different casings of "Mansonia" were mapped to a single "Mansonia" label).
6.  **Filtering Rare Classes:**
    * The class "Anopheles Stephensi" was dropped due to having only a single sample, which would cause issues during stratified data splitting.
7.  **Final DataFrame Preparation:** A `final` DataFrame was created containing `specimenId` and the cleaned `morphSpecies`. Subsequent steps in each notebook would then filter this DataFrame for images with valid local paths before model training.

## 4. Methodology for Model Training (Notebook 2 Summary, incorporating advanced techniques)

### 4.1. Data Splitting
The dataset was split into training, validation, and test sets to ensure robust evaluation. The approximate split was:
* Training set: ~70%
* Validation set: ~10%
* Test set: ~20%
Stratified splitting was used to maintain class proportions across the sets where possible.

### 4.2. Image Preprocessing for Model Input
* **Standard Approach (Notebook 1 / Baseline in Notebook 2):** Images were typically resized and normalized as per the requirements of the pre-trained models.
* **CLAHE + Otsu Cropping (Implemented and tested, primarily in Notebook 2):** To help the model focus on the mosquito, a specific preprocessing step was implemented:
    1.  Images were converted to grayscale.
    2.  Contrast Limited Adaptive Histogram Equalization (CLAHE) was applied to enhance local contrast.
    3.  Otsu's thresholding on the CLAHE-enhanced image was used to create a binary mask.
    4.  Contours were detected, and the largest contour (assumed to be the mosquito) was used to determine a bounding box.
    5.  The original color image was cropped to this bounding box with some padding.
    This cropped image was then used as input for augmentations and the model in comparative experiments.

### 4.3. Data Augmentation
* **Standard Augmentations:** Applied to all training images, including `RandomResizedCrop` (applied after any initial mosquito crop), `RandomHorizontalFlip`, `ColorJitter`, and `RandomRotation`.
* **Targeted Aggressive Augmentations:** More aggressive transformations (e.g., `RandomVerticalFlip`, stronger `ColorJitter`, wider `RandomRotation`, `RandomAffine` transforms, `GaussianBlur`) were applied specifically to the training images of underrepresented classes (`Anopheles gambiae`, `Anopheles other`).

### 4.4. Handling Class Imbalance
Several techniques were explored:
* **Targeted Data Augmentation:** (As described above).
* **Weighted Loss Function:** `nn.CrossEntropyLoss` was configured with class weights inversely proportional to class frequencies in the training set.
* **WeightedRandomSampler:** Used with the `DataLoader` to oversample instances from underrepresented classes in the training batches.

### 4.5. Models Explored
Various Vision Transformer architectures were fine-tuned:
* **DINOv2 (`facebook/dinov2-base`):** A powerful self-supervised model. Achieved good baseline results but struggled significantly with highly underrepresented classes.
* **Swin Transformer (`microsoft/swin-base-patch4-window7-224`):** Another state-of-the-art transformer. Showed similar overall performance to DINOv2, also facing challenges with the rarest classes.
* **MobileViTV2 (`apple/mobilevitv2-1.0-imagenet1k-256`):** Explored as an efficient yet capable alternative.

### 4.6. Final Model Choice: SwinV2 Transformer
After iterative experimentation with the above models and techniques, the project settled on further developing and optimizing a **SwinV2 Transformer** model. While initial experiments were conducted with Swin-Base (a SwinV1 variant), the decision was made to leverage the advancements in SwinV2 for potentially better performance and scalability, inspired by recent literature showing high accuracy for mosquito classification with Swin Transformers.

### 4.7. Training Setup
* **Device:** Utilized MPS for Apple Silicon, with fallbacks to CUDA or CPU.
* **Optimizer:** AdamW (`torch.optim.AdamW`) with weight decay.
* **Loss Function:** `torch.nn.CrossEntropyLoss` (with and without class weights).
* **Learning Rate:** Started around `1e-5`.
* **Learning Rate Scheduler:** `torch.optim.lr_scheduler.ReduceLROnPlateau` was used, monitoring validation loss.
* **Epochs:** Typically trained for 10 epochs per experiment.
* **Batch Size:** Generally 8, adjusted based on model and memory.

## 5. Key Findings & Iterative Process
* Initial models (DINOv2, Swin-Base) achieved respectable overall accuracy (~70-85%) but performed poorly on severely underrepresented classes like `Anopheles gambiae` and `Anopheles other`, often misclassifying them as the dominant `Anopheles funestus`.
* Targeted data augmentation provided some improvement for minority classes.
* Class weighting in the loss function showed mixed results, improving some classes but not significantly resolving the issues for the rarest ones.
* Oversampling with `WeightedRandomSampler` also showed some benefits but didn't fully overcome the challenge for the most imbalanced classes when used with Swin-Base.
* The CLAHE + Otsu cropping preprocessing was implemented and tested as a promising direction to help the model focus on mosquito-specific features. The impact of this technique was evaluated against models trained without this specific cropping.
* The decision to focus on SwinV2 for the final model is based on its strong performance in vision tasks and the need for an architecture that can potentially learn subtle inter-species differences more effectively, especially when combined with robust data handling techniques.
* The best test accuracy achieved during these experiments was approximately **87.07%** (with Swinv2). The core challenge remained the classification of the rarest Anopheles species.

