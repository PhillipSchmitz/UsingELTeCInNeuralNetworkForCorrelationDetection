import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix as cm
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# Notes:
# Normalization: necessary
# Learning rate: 0.1 and 0.2 -> good
# Iterations: 2000 or less and 1500 or less -> good

def init_params():
    # Case 1: authors
    # W1 = np.random.rand(10, 2000) - 0.5  # hidden layer has dim = 10, the number of words is 2000
    # b1 = np.random.rand(10, 1) - 0.5  # each node has a bias
    # W2 = np.random.rand(10, 10) - 0.5  # output layer: 10 goal categories (0-9)
    # b2 = np.random.rand(10, 1) - 0.5  # see b1
    # Case 2: gender
    W1 = np.random.rand(2, 2000) - 0.5  # Modifikation: (100, 2000)
    b1 = np.random.rand(2, 1) - 0.5  # -"- (100, 1)
    W2 = np.random.rand(2, 2) - 0.5  # -"- (2, 100)
    b2 = np.random.rand(2, 1) - 0.5
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


def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y, m, n):
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
    # print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, m, n, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    train_acc = 0
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y, m, n)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            # print("Iteration: ", i)
            predictions = get_predictions(A2)
            # print("Train data (Accuracy): " + str(get_accuracy(predictions, Y)))
            train_acc = get_accuracy(predictions, Y)
    return W1, b1, W2, b2, train_acc


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions


'''
def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    #current_image = current_image.reshape((28, 28)) * 255
    #plt.gray()
    #plt.imshow(current_image, interpolation='nearest')
    #plt.show()


def conf_matrix(dev_predications, Y_dev):
    cmatrix = cm(dev_predications, Y_dev, normalize='true')
    sns.heatmap(cmatrix, annot=True, linewidths=0.5, fmt='.2f', cmap='cividis', annot_kws={'size': 8})

    # Case 1: authors
    plt.xticks(np.arange(len(author_names)) + 0.5, author_names, rotation=45)
    plt.yticks(np.arange(len(author_names)) + 0.5, author_names, rotation=0)
    # Case 2: gender
    #plt.xticks(np.arange(len(gender_names)) + 0.5, gender_names, rotation=45)
    #plt.yticks(np.arange(len(gender_names)) + 0.5, gender_names, rotation=0)
    plt.show()
'''


def rank_words(ranking: list, new_string, new_integer, max_length=10):
    # Check if the new integer value is higher than any existing values
    for i, (_, existing_integer) in enumerate(ranking):
        if new_integer > existing_integer:
            # Insert the new element before the existing one
            ranking.insert(i, (new_string, new_integer))
            break
    else:
        # If not higher than any existing values, append to the end
        ranking.append((new_string, new_integer))

    # Remove the first element if the data structure exceeds the specified length
    if len(ranking) > max_length:
        ranking.pop(0)  # Remove the first element (closest to the lowest integer value)

    return ranking


def write_results_into_txt_file(final_list: list):
    # Open a new text file for writing
    with open('top10_words_2k2k_gender.txt', 'w') as f:
        f.write("The Top 10 most important words in the dataset are as follows:\n")
        # Write each word and accuracy to the file
        for word, accuracy in final_list[::-1]:
            f.write(f"{word}: {accuracy}\n")

    print("The results have been saved in the 'top10_words_2k2k_gender.txt' file.")


def main():
    data = pd.read_csv("ELTeC-eng-dataset_2000tok-2000mfw.csv", sep=";")

    # Preprocess the data
    # data.drop(columns=['Unnamed: 0', 'idno', 'gender'], axis=1, inplace=True)  # drop unwanted cols (case 1: author)
    data.drop(columns=['Unnamed: 0', 'idno', 'author'], axis=1, inplace=True)  # drop unwanted cols (case 2: gender)

    data = data.iloc[:, :2001]  # only include the author column and the first 784 words (now 2001 columns)
    # author_names = data["author"].unique()  # save author names for heatmap visualization
    gender_names = data["gender"].unique()  # save gender for heatmap visualization

    # Mapping of the authors
    #unique_authors = data['author'].unique()
    #author_to_int_mapping = {author: i for i, author in enumerate(unique_authors)}
    #data['author'] = data['author'].map(author_to_int_mapping)  # replace the author name with his mapping integer value
    # Mapping of the genders
    unique_authors = data['gender'].unique()
    author_to_int_mapping = {author: i for i, author in enumerate(unique_authors)}
    data['gender'] = data['gender'].map(author_to_int_mapping)  # replace the gender with its mapping integer value

    default_data = data
    top10_words = []
    for column_index in range(1, len(data.columns)):  # We don't count the first column (dependent variable)
        column_name = data.columns[column_index]
        if column_index % 10 == 0:  # set milestones to view progress
            print(column_index, column_name)
        data.drop(column_name, axis=1)
        avg_train_acc = 0
        avg_test_acc = 0
        limit = 5  # freely adaptable
        for i in range(0, limit):  # repeat process five times (for debugging reasons)
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

            learning_rate = 0.1
            iterations = 500
            W1, b1, W2, b2, train_acc = gradient_descent(X_train, Y_train, m, n, learning_rate, iterations=iterations)
            # test_prediction(0, W1, b1, W2, b2)
            # test_prediction(1, W1, b1, W2, b2)
            # test_prediction(2, W1, b1, W2, b2)
            # test_prediction(3, W1, b1, W2, b2)

            dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
            get_accuracy(dev_predictions, Y_dev)
            # Get performance on Y_dev (i.e. test data)
            # print("Test data (Accuracy): " + str(get_accuracy(dev_predictions, Y_dev)))
            # conf_matrix(dev_predictions, Y_dev) # create confusion matrix (heatmap) and show predictions
            avg_train_acc = (avg_train_acc + train_acc)
            avg_test_acc = (avg_test_acc + get_accuracy(dev_predictions, Y_dev))
        avg_train_acc /= limit
        avg_test_acc /= limit
        top10_words = rank_words(top10_words, column_name, avg_train_acc)  # add word and its accuracy to ranking list
        # print("AVG Train Accuracy: " + str(avg_train_acc))
        # print("AVG Test Accuracy: " + str(avg_test_acc))
        data = default_data

    write_results_into_txt_file(top10_words)


if __name__ == '__main__':
    main()
