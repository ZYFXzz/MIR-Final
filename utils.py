import mirdata
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from scikeras.wrappers import KerasClassifier
from sklearn.svm import SVC


def load_data(data_home):
    """
    Load the mini-Medley-Solos-DB dataset.

    Parameters
    ----------
    data_home : str
                Path to where the dataset is located

    Returns
    -------
    dataset: mirdata.Dataset
             The mirdata Dataset object correspondong to Medley-Solos-DB
    """

    # YOUR CODE HERE
    # Hints:
    # Look at the mirdata tutorial on how to initialize a dataset.
    # Define the correct path using the data_home argument.
    dataset = mirdata.initialize("medley_solos_db", data_home=data_home)
    return dataset


def split_data(tracks):
    """
    Splits the provided dataset into training, validation, and test subsets based on the 'subset'
    attribute of each track.

    Parameters
    ----------
    track_list : list
                 list of dataset.track objects from Medley_solos_DB dataset

    Returns
    -------
    tracks_train : list
        List of tracks belonging to the 'training' subset.
    tracks_validate : list
        List of tracks belonging to the 'validation' subset.
    tracks_test : list
        List of tracks belonging to the 'test' subset.
    """
    # YOUR CODE HERE
    tracks_train = []
    tracks_validate = []
    tracks_test = []

    for key, track in tracks.items():
        if track.subset == "training":
            tracks_train.append(track)
        elif track.subset == "validation":
            tracks_validate.append(track)
        elif track.subset == "test":
            tracks_test.append(track)
    return tracks_train, tracks_validate, tracks_test


def check_balance(y, dataset_name):
    y_df = pd.DataFrame(y, columns=['instrument'])
    
    # count the number of samples in each class
    class_counts = y_df['instrument'].value_counts().sort_index()
    
    # plot the class counts
    plt.figure(figsize=(8, 4))
    class_counts.plot(kind='bar')
    plt.title(f"{dataset_name} class counts")
    plt.xlabel('instruments')
    plt.ylabel('samples')
    plt.show()



def compute_mfccs(y, sr, n_fft=2048, hop_length=512, n_mels=128, n_mfcc=20):
    """
    Compute mfccs for an audio file using librosa, removing the 0th MFCC coefficient.

    Parameters
    ----------
    y : np.array
        Mono audio signal
    sr : int
        Audio sample rate
    n_fft : int
        Number of points for computing the fft
    hop_length : int
        Number of samples to advance between frames
    n_mels : int
        Number of mel frequency bands to use
    n_mfcc : int
        Number of mfcc's to compute

    Returns
    -------
    mfccs: np.array (t, n_mfcc - 1)
        Matrix of mfccs

    """
    # YOUR CODE HERE
    mfccs = librosa.feature.mfcc(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc
    )
    #print(mfccs)
    #print("")
    #print(mfccs.T[:, 1:])

    return mfccs.T[:, 1:]


def get_stats(features):
    """
    Compute summary statistics (mean and standard deviation) over a matrix of MFCCs.
    Make sure the statitics are computed across time (i.e. over all examples,
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


def normalize(features, features_mean, features_std):
    """
    Normalize (standardize) a set of features using the given mean and standard deviation.

    Parameters
    ----------
    features: np.array (n_examples, n_features)
              Matrix of features
    features_mean: np.array (n_features)
              The mean of the features
    features_std: np.array (n_features)
              The standard deviation of the features

    Returns
    -------
    features_norm: np.array (n_examples, n_features)
                   Standardized features

    """

    # YOUR CODE HERE
    # print(features, features_mean)
    features_norm = (features - features_mean) / features_std
    # print("features_ norm is " + str(features_norm))

    return features_norm


def get_features_and_labels(
    track_list, n_fft=2048, hop_length=512, n_mels=128, n_mfcc=20
):
    """
    Our features are going to be the `mean` and `std` MFCC values of a track concatenated
    into a single vector of size `2*n_mfcss`.

    Create a function `get_features_and_labels()` such that extracts the features
    and labels for all tracks in the dataset, such that for each audio file it obtains a
    single feature vector. This function should do the following:

    For each track in the collection (e.g. training split),
        1. Compute the MFCCs of the input audio, and remove the first (0th) coeficient.
        2. Compute the summary statistics of the MFCCs over time:
            1. Find the mean and standard deviation for each MFCC feature (2 values for each)
            2. Stack these statistics into single 1-d vector of length ( 2 * (n_mfccs - 1) )
        3. Get the labels. The label of a track can be accessed by calling `track.instrument_id`.
    Return the labels and features as `np.arrays`.

    Parameters
    ----------
    track_list : list
                 list of dataset.track objects from Medley_solos_DB dataset
    n_fft : int
                 Number of points for computing the fft
    hop_length : int
                 Number of samples to advance between frames
    n_mels : int
             Number of mel frequency bands to use
    n_mfcc : int
             Number of mfcc's to compute

    Returns
    -------
    feature_matrix: np.array (len(track_list), 2*(n_mfcc - 1))
        The features for each track, stacked into a matrix.
    label_array: np.array (len(track_list))
        The label for each track, represented as integers
    """

    # Hint: re-use functions from previous parts (e.g. compute_mfcss and get_stats)
    # YOUR CODE HERE
    feature_list = []
    label_list = []

    # Loop through each track in the dataset
    for track in track_list:
        # Load the audio for the track
        audio = track.audio
        y, sr = audio[0], audio[1]
        # step 1
        mfccs = compute_mfccs(
            y, sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, n_mfcc=n_mfcc
        )

        # step 2
        mfcc_mean, mfcc_std = get_stats(mfccs)
        # stack
        feature_vector = np.hstack([mfcc_mean, mfcc_std])

        # step 3 labels
        label = track.instrument_id

        # Append the feature vector and label to their respective lists
        feature_list.append(feature_vector)
        label_list.append(label)

    feature_matrix = np.array(feature_list)
    label_array = np.array(label_list)

    return feature_matrix, label_array


def fit_knn(
    train_features,
    train_labels,
    validation_features,
    validation_labels,
    ks=[1,2,3,4,5,6,7,8,9,10,20,30,40,50],
):
    """
    Fit a k-nearest neighbor classifier and choose the k which maximizes the
    *f-measure* on the validation set.

    Plot the f-measure on the validation set as a function of k.

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

    # Hint: for simplicity you can search over k = 1, 5, 10, 50.
    # Use KNeighborsClassifier from sklearn.
    # YOUR CODE HERE
    best_f1 = 0
    best_k = 0
    f1_scores = []

    # Iterate over the values of k
    for k in ks:
        knn = KNeighborsClassifier(n_neighbors=k)

        knn.fit(train_features, train_labels)

        val_predictions = knn.predict(validation_features)

        # Compute the F1 score on the validation set
        f1 = f1_score(validation_labels, val_predictions, average="macro")
        f1_scores.append(f1)
        # update f1
        if f1 > best_f1:
            best_f1 = f1
            best_k = k
            best_knn_clf = knn

    # Plot the graph
    plt.plot(ks, f1_scores, marker="o")
    plt.xlabel("k")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs k for KNN")
    plt.show()

    return best_knn_clf, best_k


def fit_random_forest(X_train, y_train, X_val, y_val):
    """
    Fit a random forest classifier and choose the max_depth which maximizes the
    *f-measure* on the validation set.
    """
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    # hyperparameters
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10, 20, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # best model
    best_rf = grid_search.best_estimator_
    print("best hyperparameters:", grid_search.best_params_)

    # combine train and val data and train the best model
    X_final_train = np.concatenate((X_train, X_val))
    y_final_train = np.concatenate((y_train, y_val))
    best_rf.fit(X_final_train, y_final_train)

    return best_rf


def best_NN(X_latent_train, Y_latent_train, X_latent_validate, Y_latent_validate):
    """
    Train a neural network classifier.
    """
    # preprocess
    num_classes = len(np.unique(Y_latent_train))

    # build model
    def create_model(hidden_layer_1=64, hidden_layer_2=32, dropout_rate=0.2, optimizer='adam'):
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(X_latent_train.shape[1],)),
            tf.keras.layers.Dense(hidden_layer_1, activation='relu'),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(hidden_layer_2, activation='relu'),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=optimizer,
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        return model

    # use KerasClassifier to find the best hyperparameters
    model = KerasClassifier(model=create_model, hidden_layer_1=64, hidden_layer_2=32, dropout_rate=0.2, optimizer='adam', verbose=0)

    # define hyperparameters
    param_grid = {
        'hidden_layer_1': [32, 64],
        'hidden_layer_2': [16, 32],
        'dropout_rate': [0.1, 0.2],
        'optimizer': ['adam'],
        'epochs': [20, 30],
        'batch_size': [16, 32]
    }

    # use GridSearchCV to find the best hyperparameters
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1)
    grid_result = grid.fit(X_latent_train, Y_latent_train, validation_data=(X_latent_validate, Y_latent_validate))
    best_model = grid_result.best_estimator_
    print("the best paprameters are:", grid_result.best_params_)

    return best_model


def best_SVM(X_latent_train, Y_latent_train, X_latent_validate, Y_latent_validate):
    # combine train and val data
    X_train = np.concatenate((X_latent_train, X_latent_validate), axis=0)
    Y_train = np.concatenate((Y_latent_train, Y_latent_validate), axis=0)

    # define hyperparameters and model
    svm = SVC(probability=True)
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }

    # use GridSearchCV to find the best hyperparameters
    grid = GridSearchCV(estimator=svm, param_grid=param_grid, cv=3, n_jobs=-1)
    grid_result = grid.fit(X_train, Y_train)
    best_model = grid_result.best_estimator_

    # print the best paprameters
    print(f"the best paprameters are: {grid_result.best_params_}")

    return best_model