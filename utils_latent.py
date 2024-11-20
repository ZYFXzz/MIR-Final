import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from music2latent import EncoderDecoder

# Initialize Music2Latent EncoderDecoder
encdec = EncoderDecoder()

def compute_latent_features(y, sr):
    """
    Compute latent features for an audio file using Music2Latent.

    Parameters
    ----------
    y : np.array
        Mono audio signal
    sr : int
        Audio sample rate

    Returns
    -------
    latents: np.array (sequence_length, n_latent_features)
        Matrix of latent features
    """
    # Encode to latent space
    latents = encdec.encode(y)
    # print(latents)
    # print(latents.size())
    
    # Convert to numpy, ignoring batch dimension if present (i.e., shape [1, 64, time_steps])
    if latents.shape[0] == 1:
        latents = latents.squeeze(0)
    # print(latents)
    # print(latents.size())
    # latent has shape (batch_size/audio_channels, dim (64), sequence_length)
    return latents.T  # Transpose for consistency with MFCC format, (times frames, mfcc coefficients), and latent.T will give us (time frames/sequence length, latent channels)


def get_stats(features):
    """
    Compute summary statistics (mean and standard deviation) over a matrix of MFCCs.
    Make sure the statistics are computed across time (i.e. over all examples,
    compute the mean of each feature).

    Parameters
    ----------
    features: np.array (n_examples, n_features)
              Matrix of features

    Returns
    -------
    features_mean: np.array (n_features)
                   The mean of the features
    features_std: np.array (n_features)
                   The standard deviation of the features

    """
    # Hint: use numpy mean and std functions, and watch out for the axis.
    # YOUR CODE HERE
    features_mean = np.mean(features, axis=0)

    features_std = np.std(features, axis=0)

    return features_mean, features_std

def get_features_and_labels_latent(track_list):
    """
    Extract latent space features and labels for each track in the dataset.

    Parameters
    ----------
    track_list : list
                 list of dataset.track objects from Medley_solos_DB dataset

    Returns
    -------
    feature_matrix: np.array (len(track_list), 2 * n_latent_features)
        The latent features for each track, stacked into a matrix.
    label_array: np.array (len(track_list))
        The label for each track, represented as integers
    """
    feature_list = []
    label_list = []

    for track in track_list:
        y, sr = track.audio
        latents = compute_latent_features(y, sr)
        #print(latents)
        
        #change tensor to numpy 2D array
        latents_cpu = latents.cpu()
        latents = latents_cpu.numpy()
        #print(latents)

        # Compute statistics: mean and std across time (axis=0)
        latent_mean, latent_std = get_stats(latents)

        # Concatenate mean and std for feature vector
        feature_vector = np.hstack([latent_mean, latent_std])

        # Get the label
        label = track.instrument_id

        feature_list.append(feature_vector)
        label_list.append(label)

    feature_matrix = np.array(feature_list)
    label_array = np.array(label_list)

    return feature_matrix, label_array


def fit_knn_latent(train_features, train_labels, validation_features, validation_labels, ks=[1,2,3,4,5,6,7,8,9,10,20,30,40,50]):
    """
    Fit a k-nearest neighbor classifier on latent features and choose the k which maximizes the f-measure.

    Parameters
    ----------
    train_features : np.array (n_train_examples, n_features)
        training feature matrix
    train_labels : np.array (n_train_examples)
        training label array
    validation_features : np.array (n_validation_examples, n_features)
        validation feature matrix
    validation_labels : np.array (n_validation_examples)
        validation label array
    ks: list of int
        k values to evaluate using the validation set

    Returns
    -------
    knn_clf : scikit learn classifier
        Trained k-nearest neighbor classifier with the best k
    best_k : int
        The k which gave the best performance
    """
    best_f1 = 0
    best_k = 0
    f1_scores = []

    for k in ks:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_features, train_labels)
        val_predictions = knn.predict(validation_features)

        # Compute F1 score on the validation set
        f1 = f1_score(validation_labels, val_predictions, average="macro")
        f1_scores.append(f1)
        
        if f1 > best_f1:
            best_f1 = f1
            best_k = k
            best_knn_clf = knn

    plt.plot(ks, f1_scores, marker="o")
    plt.xlabel("k")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs k for KNN on Latent Features")
    plt.show()

    return best_knn_clf, best_k