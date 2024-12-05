# Comparing Audio Features for Instrument Recognition: MFCCs vs. Latent Space from the Music2Latent
## Project Overview

**This project explores two types of audio features for musical instrument classification:**

1.	MFCC-Based Features: Traditional audio features derived from signal processing.
2.	Latent Space Features: Extracted using the Music2Latent autoencoder, a recent model presented at ISMIR.

The dataset used for this project is Medley-Solos-DB, which provides labeled samples of various instrument classes. The aim is to evaluate the performance of these features across different machine learning models.


-------------------------------------------------------------

## Workflows

**Part1: Data Loading and Preprocessing:**

- Dataset: Medley-Solos-DB, which includes over 20,000 labeled audio samples.
- Further split into training, validation and testing set.
- Check the balance of each dataset.

The dataset is loaded using the mirdata library, which ensures structured access and metadata handling.

**Part2: Feature Extrac:**

MFCC Features:
1. Computed using the Mel-Frequency Cepstral Coefficients algorithm.
2. Statistical features such as mean and standard deviation are calculated for classification tasks.
Latent Space Features:
Generated using Music2Latent, which encodes audio signals into a highly compressed representation that captures timbral characteristics.


**Part3: Normalize the Data**

Normalize features with the mean and standard deviation of the training set to ensure consistency and generalization across subsets.

**Part4: Model Training and Validation**

Models Used:
1. k-Nearest Neighbors (KNN): A simple and interpretable baseline.
2. Random Forest: Handles non-linear separability and provides feature importance.
3. Support Vector Machines (SVM): Effective in high-dimensional feature spaces.
4. Neural Networks (NN): Captures complex, non-linear relationships.
- Hyperparameter Tuning:
Performed using grid search to optimize the performance of each model.

**Part5: Evaluation and Analysis**

Metrics:
1. F1 Score: Macro-averaged F1 score to measure classification performance.
2. Confusion Matrix: Visualizes misclassification patterns.
- Error Analysis:
Identifies the best and worst-performing classes and common misclassification trends.


-------------------------------------------------------------

## Outputs

- Data Distribution:
Displays the number of tracks in training, validation, and testing sets.
- Feature Visualization:
1. ummary statistics (mean and standard deviation) for MFCCs.
2. Feature distribution for latent space representations.
- Model Performance:
1. F1 scores for each feature type and model.
2. onfusion matrices for both feature types.
- Insights:
1. Comparison of feature effectiveness for instrument classification.
2. Identification of common misclassification patterns.


-------------------------------------------------------------

## Example Results

1.	Best Performing Features: Latent space features generally outperform MFCCs in capturing timbral differences.
2.	Best Model: Support Vector Machines (SVM) achieve the highest accuracy with latent space features.
3.	Error Analysis:
- Wind instruments (e.g., flute, saxophone) often confuse the classifier due to similar timbral qualities.
- Distinctive instruments like piano and guitar achieve the best results.


-------------------------------------------------------------

## Acknowledgments

- Dataset: Medley-Solos-DB.
- References:
1. Music2Latent autoencoder (presented at ISMIR).
2. Deep convolutional networks for musical instrument recognition (Lostanlen & Cella, 2016).
- Tools: mirdata, scikit-learn, and torch.