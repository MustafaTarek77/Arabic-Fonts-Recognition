# Arabic Font Classification using LPQ Features and SVM Classifier

This project aims to classify Arabic fonts using Local Phase Quantization (LPQ) features for extraction and Support Vector Machine (SVM) as the classifier. This README provides detailed instructions on how to set up and run the code.

## Prerequisites

- Python 3.x
- Required Python libraries: `numpy`, `opencv-python`, `scikit-learn`, `Flask`, `pytesseract`

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/MustafaTarek77/Arabic-Fonts-Recognition.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Arabic-Fonts-Recognition
   ```

3. Install the required Python libraries:

   ```bash
   pip install -r requirements.txt
   ```

## Dataset

- The dataset used for training the model should be organized in the following structure:

  ```
  fonts-dataset/
  ├── 0/
  ├── 1/
  ├── 2/
  └── 3/

  ```

- The testset used for testing the model should be organized in the following structure:
  ```
  data/
  ├── 0/
  ├── 1/
  ├── 2/
  └── 3/
  ```
  Each font folder should contain images of Arabic text samples for the corresponding font.

## Training

- Train the SVM classifier using the extracted LPQ features:

  ```bash
  python train.py
  ```

## Testing

- Test the trained classifier on the test dataset:

  ```bash
  python main.py
  ```

## Results

- The accuracy and other evaluation metrics of the classifier will be displayed in the console upon running the testing script.
