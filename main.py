import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix as cm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time

# Notes:
# Normalization: necessary
# Learning rate: 0.1 and 0.2 -> good
# Iterations: 2000 or less and 1500 or less -> good

# Dataset selection
data = pd.read_csv("ELTeC-eng-dataset_2000tok-2000mfw.csv", sep=";")
#data = pd.read_csv("ELTeC-eng-dataset_5000tok-2000mfw.csv", sep=";")

# Label selection (case 1: author, case 2: gender)
data.drop(columns=['Unnamed: 0', 'idno', 'gender'], axis=1, inplace=True)  # drop unwanted columns (case 1: author)
#data.drop(columns=['Unnamed: 0', 'idno', 'author'], axis=1, inplace=True)  # drop unwanted columns (case 2: gender)

data_copy = data
data_copy = data_copy.drop('author', axis=1) # case 1
#data_copy = data_copy.drop('gender', axis=1) # case 2


data = data.iloc[:, :501]  # only include the label column and the first 500 features
author_names = data["author"].unique()  # case 1: save author names for heatmap visualization
#gender_names = data["gender"].unique()  # case 2: save gender for heatmap visualization

# Mapping of the authors (case 1)
unique_authors = data['author'].unique()
author_to_int_mapping = {author: i for i, author in enumerate(unique_authors)}
data['author'] = data['author'].map(author_to_int_mapping)  # replace the author name with his mapping integer value
# Mapping of the genders (case 2)
#unique_genders = data['gender'].unique()
#gender_to_int_mapping = {author: i for i, author in enumerate(unique_genders)}
#data['gender'] = data['gender'].map(gender_to_int_mapping)  # replace the gender with its mapping integer value

# Preprocess data
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)  # shuffle before splitting into dev and training sets

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = scaler.fit_transform(X_dev)
# X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = scaler.fit_transform(X_train)
# X_train = X_train / 255.
_, m_train = X_train.shape

#print(Y_train) # Debugging

def init_params(num_words):
    # Case 1: authors
    W1 = np.random.rand(10, num_words) - 0.5  # hidden layer has dim = 10, the number of words is n
    b1 = np.random.rand(10, 1) - 0.5  # each node has a bias
    W2 = np.random.rand(10, 10) - 0.5  # results layer: 10 goal categories (0-9)
    b2 = np.random.rand(10, 1) - 0.5  # see b1
    # Case 2: gender
    #W1 = np.random.rand(2, num_words) - 0.5 # Modifikation: (100, n)
    #b1 = np.random.rand(2, 1) - 0.5 # -"- (100, 1)
    #W2 = np.random.rand(2, 2) - 0.5 # -"- (2, 100)
    #b2 = np.random.rand(2, 1) - 0.5
    return W1, b1, W2, b2


def ReLU(Z):
    return np.maximum(Z, 0)


def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def ReLU_deriv(Z):
    return Z > 0


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


def get_predictions(A2):
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    # Include/Don't include predicated labels
    #print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def gradient_descent(num_words, X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params(num_words)
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            #print("Iteration: ", i)
            predictions = get_predictions(A2)
            #print("Train data (Accuracy): " + str(get_accuracy(predictions, Y)))
    return W1, b1, W2, b2


def backward_elimination(X_train, Y_train, alpha, iterations, filename_prefix='results'):
    print("Start: Backward Elimination\n")
    n_features = X_train.shape[0]  # Get the number of features
    selected_features = list(range(n_features))  # Start with all features

    # Train the network with all features to obtain a baseline accuracy
    W1, b1, W2, b2 = gradient_descent(X_train.shape[0], X_train, Y_train, alpha, iterations)
    dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
    baseline_accuracy = get_accuracy(dev_predictions, Y_dev)
    print(f"The baseline accuracy of the neural network with all features is: {baseline_accuracy}\n")

    # Store the accuracy for each step
    accuracies = [baseline_accuracy]
    times = []  # List to track time taken for each step
    removed_features = []
    selected_features_history = []  # List to save the best features from each iteration
    dev_predictions_history = dev_predictions

    # Initial confusion matrix
    conf_matrix(dev_predictions, Y_dev, f"{filename_prefix}_confusion_matrix-baseline")

    iteration_num = 0
    while len(selected_features) > 0:
        iteration_num += 1
        print(f"Iteration: {iteration_num}")
        feature_accuracies = []

        # Start timing the current iteration
        start_time = time.time()

        # Keep history of the features before an iteration
        selected_features_history.append(
            [data_copy.columns[idx] for idx in selected_features.copy()])

        # Evaluate accuracy for each feature (remove one at a time)
        for feature_idx in selected_features:
            features_to_use = [f for f in selected_features if f != feature_idx]  # Remove feature at index from list of features
            X_train_reduced = X_train[features_to_use, :] # Adjust X_train
            X_dev_reduced = X_dev[features_to_use, :] # ... and X_dev

            # Train the network with the remaining features
            W1, b1, W2, b2 = gradient_descent(X_train_reduced.shape[0], X_train_reduced, Y_train, alpha, iterations)
            dev_predictions = make_predictions(X_dev_reduced, W1, b1, W2, b2)
            accuracy = get_accuracy(dev_predictions, Y_dev)
            feature_accuracies.append((accuracy, feature_idx))

        # Sort the features based on the accuracy
        feature_accuracies.sort(reverse=True, key=lambda x: x[0])  # Sort by accuracy, highest first
        best_accuracy, best_feature_idx = feature_accuracies[0]  # Get the feature with the best accuracy
        print(f"The feature '{data_copy.columns[best_feature_idx]}' led to the highest accuracy of {best_accuracy} out of all features.")

        # Check whether the worst feature's accuracy is higher than the (initial) baseline accuracy
        if best_accuracy > baseline_accuracy:
            temp_selected_features = selected_features.copy()
            temp_selected_features.remove(best_feature_idx)  # Remove the worst feature (with the highest accuracy)

            X_train_reduced = X_train[temp_selected_features, :]  # Adjust X_train
            X_dev_reduced = X_dev[temp_selected_features, :]  # ... and X_dev

            # Train the network with the remaining features
            W1, b1, W2, b2 = gradient_descent(X_train_reduced.shape[0], X_train_reduced, Y_train, alpha, iterations)
            dev_predictions = make_predictions(X_dev_reduced, W1, b1, W2, b2)
            new_accuracy = get_accuracy(dev_predictions, Y_dev)
            print(f"Model accuracy after re-training: {new_accuracy}")

            # If the new accuracy is higher, update the baseline accuracy
            if new_accuracy > baseline_accuracy:
                baseline_accuracy = new_accuracy
                dev_predictions_history = dev_predictions
                selected_features.remove(best_feature_idx)
                removed_features.append(data_copy.columns[best_feature_idx])
                print(
                    f"The feature '{data_copy.columns[best_feature_idx]}' (Index: {best_feature_idx}) was permanently removed.")

            else:
                print("No accuracy improvement after re-training the model. Stop process!")
                print(f"The final accuracy is: {baseline_accuracy}\n")

                # End timing the current iteration and store the time
                end_time = time.time()
                times.append(end_time - start_time)

                break  # No improvement, stop elimination

        else:
            print("No accuracy improvement after stepwise feature removal. Stop process!")
            print(f"The final accuracy is: {baseline_accuracy}\n")

            # End timing the current iteration and store the time
            end_time = time.time()
            times.append(end_time - start_time)

            break  # No improvement, stop elimination

        # Add the new accuracy to the list for later visualization
        accuracies.append(baseline_accuracy)

        # End timing the current iteration and store the time
        end_time = time.time()
        times.append(end_time - start_time)

        print(f"The new accuracy is: {baseline_accuracy}. There are {len(selected_features)} features remaining.\n")

    # Create statistics
    # Save the final confusion matrix
    conf_matrix(dev_predictions_history, Y_dev, f"{filename_prefix}_confusion_matrix-final")

    # Plot the accuracy progression
    plt.figure(figsize=(10, 5))

    # Plot the accuracy over steps
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o', label='Accuracy')
    plt.xlabel('Steps of Backward Elimination')
    plt.ylabel('Accuracy')
    plt.title('Progression of Model Accuracy During Backward Elimination')

    # Plot the time taken per step
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(times) + 1), times, 'bo-', label='Time per Step')  # Use 'bo-' for dots
    plt.xlabel('Steps of Backward Elimination')
    plt.ylabel('Time (seconds)')
    plt.title('Time Taken for Each Iteration')

    plt.tight_layout()

    # Save the plots to a file
    plt.savefig(f"{filename_prefix}_elimination_plots.png")

    plt.show()

    # Convert selected feature indices to their corresponding names and print final selected feature names
    final_selected_features = [data_copy.columns[idx] for idx in selected_features]
    print(f"Final selected features after Backward Elimination: {final_selected_features}")

    print(f"Final model accuracy: {baseline_accuracy}")

    # Create CSV file (summary of results)
    # Summarize the results and outputs in a CSV file
    output_data = []

    # Initialize the final selected features at each step
    for i, accuracy in enumerate(accuracies):
        # Get the list of selected features at the current step from the history
        selected_features_at_step = selected_features_history[i]
        # Convert the selected features list into a comma-separated string
        selected_features_str = ', '.join(selected_features_at_step)
        # If no time is recorded, set as 'N/A'
        time_taken = times[i] if i < len(times) else 'N/A'
        # If no features were removed, set as None
        removed_features_str = removed_features[i - 1] if i > 0 else None

        iteration_data = {
            'Step': i + 1,
            'Accuracy': accuracy,
            'Time (seconds)': time_taken,  # Use 'N/A' if no time is available
            'Removed Features': removed_features_str,  # Use None if no features were removed
            'Selected Features at Step': selected_features_str  # Store as comma-separated string
        }
        output_data.append(iteration_data)

    df_output = pd.DataFrame(output_data)
    df_output.to_csv(f'{filename_prefix}_backward_elimination_output.csv', index=False, sep=';')

    return selected_features


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions


def test_prediction(index, W1, b1, W2, b2, X_train_reduced, optimized=False):
    if not optimized:
        # current_image = X_train[:, index, None]
        prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
        label = Y_train[index]
        print("Prediction: ", prediction)
        print("Label: ", label)
    else:
        #current_image = X_train[:, index, None]
        prediction = make_predictions(X_train_reduced[:, index, None], W1, b1, W2, b2)
        label = Y_train[index]
        print("Prediction: ", prediction)
        print("Label: ", label)

    #current_image = current_image.reshape((28, 28)) * 255
    #plt.gray()
    #plt.imshow(current_image, interpolation='nearest')
    #plt.show()


def conf_matrix(dev_predications, Y_dev, filename):
    cmatrix = cm(dev_predications, Y_dev, normalize='true')
    sns.heatmap(cmatrix, annot=True, linewidths=0.5, fmt='.2f', cmap='cividis', annot_kws={'size': 8})

    # Case 1: authors
    plt.xticks(np.arange(len(author_names)) + 0.5, author_names, rotation=45)
    plt.yticks(np.arange(len(author_names)) + 0.5, author_names, rotation=0)
    # Case 2: gender
    #plt.xticks(np.arange(len(gender_names)) + 0.5, gender_names, rotation=45)
    #plt.yticks(np.arange(len(gender_names)) + 0.5, gender_names, rotation=0)
    plt.savefig(filename)
    plt.show() # Uncomment to show the confusion matrix

'''
# Analyze and visualize word count distribution
def visualize_wordcount_distribution(file_path, batch_size, num_features):
    # Read the file
    data = pd.read_csv(file_path, sep=";")
    data.drop(columns=['Unnamed: 0', 'idno', 'gender', 'author'], axis=1, inplace=True)
    # Select the first `num_features` columns
    data = data.iloc[:, :num_features]

    # Calculate number of batches
    num_batches = data.shape[1] // batch_size + (data.shape[1] % batch_size > 0)

    # Loop through batches and create violin plots
    for i in range(num_batches):
        plt.figure(figsize=(15, 6))
        sns.violinplot(data=data.iloc[:, i * batch_size: (i + 1) * batch_size], inner="quartile")
        plt.title(
            f"Violin Plot of Word Frequencies (Features {i * batch_size + 1}-{min((i + 1) * batch_size, data.shape[1])})")
        plt.xticks(rotation=90)
        plt.show()
'''

def main():
    learning_rate = 0.1
    iterations = 500

    #visualize_wordcount_distribution(file_path="ELTeC-eng-dataset_2000tok-2000mfw.csv", batch_size=50, num_features=500)

    #W1, b1, W2, b2 = gradient_descent(X_train.shape[0], X_train, Y_train, alpha=learning_rate, iterations=iterations)
    #test_prediction(0, W1, b1, W2, b2)
    #test_prediction(1, W1, b1, W2, b2)
    #test_prediction(2, W1, b1, W2, b2)
    #test_prediction(3, W1, b1, W2, b2)

    #dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
    #get_accuracy(dev_predictions, Y_dev)
    # Get performance on Y_dev (i.e. test data)
    #print("Test data (Accuracy): " + str(get_accuracy(dev_predictions, Y_dev)))
    #conf_matrix(dev_predictions, Y_dev)

    backward_elimination(X_train, Y_train, learning_rate, iterations=iterations,filename_prefix=f"test_author_500feat_{iterations}iter")


if __name__ == '__main__':
    main()

