import numpy as np
import pandas as pd


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def encode_labels(y):
    classes = sorted(list(set(y)))
    for i in range(len(classes)):
        y[y == classes[i]] = i
    return y.astype(int), classes


def decode_labels(y, classes):
    return np.array(classes)[y]


def mle_cost_function(X, y, theta):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    epsilon = 1e-7
    cost = (1 / m) * ((-y).T @ np.log(h + epsilon) - (1 - y).T @ np.log(1 - h + epsilon))
    return cost


def gradient_descent(X, y, theta):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    grad = np.dot(X.T, (h - y)) / m
    return grad


def logistic_regression(X, y, k, maxSteps, alpha, eps):
    m, n = X.shape[0], X.shape[1]
    theta = np.zeros((n, k))
    cost_history = []
    for i in range(k):
        y_binary = np.where(y == i, 1, 0)
        while True:
            cost = mle_cost_function(X, y_binary, theta[:, i])
            # print(f'Cost: {cost}')
            cost_history.append(cost)
            grad = gradient_descent(X, y_binary, theta[:, i])
            theta[:, i] -= alpha * grad

            if len(cost_history) > 1 and np.abs(cost_history[-1] - cost_history[-2]) < eps:
                break
            if len(cost_history) > maxSteps:
                break
    return theta


def predict(X, theta):
    h = sigmoid(np.dot(X, theta))
    return np.argmax(h, axis=1)


def calculate_metrics(y_true, y_pred):
    num_classes = len(np.unique(y_true))
    confusion_matrix = np.zeros((num_classes, num_classes))

    for true_label, pred_label in zip(y_true, y_pred):
        confusion_matrix[int(true_label), int(pred_label)] += 1

    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)

    accuracy = np.sum(TP) / confusion_matrix.sum()

    print(f'\nAccuracy: {accuracy}')

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)
    f1_score = 2 * precision * recall / (precision + recall)

    micro_avg_precision = TP.sum() / (TP.sum() + FP.sum())
    micro_avg_recall = TP.sum() / (TP.sum() + FN.sum())
    micro_avg_specificity = TN.sum() / (TN.sum() + FP.sum())
    micro_avg_f1_score = 2 * micro_avg_precision * micro_avg_recall / (micro_avg_precision + micro_avg_recall)

    print(f'Micro-average precision: {micro_avg_precision}')
    print(f'Micro-average recall: {micro_avg_recall}')
    print(f'Micro-average specifity: {micro_avg_specificity}')
    print(f'Micro-average F1: {micro_avg_f1_score}')

    print(f'Macro-average precision: {np.mean(precision)}')
    print(f'Macro-average recall: {np.mean(recall)}')
    print(f'Macro-average specifity: {np.mean(specificity)}')
    print(f'Macro-average F1: {np.mean(f1_score)}')


def main():
    train_data = pd.read_csv('multi_classes/train/train.csv', sep=';', header=None)
    X_train = train_data.iloc[:, :-1].values

    y_train, encoder = encode_labels(train_data.iloc[:, -1].values)

    theta = logistic_regression(X_train, y_train, len(np.unique(y_train)), 10000, 0.01, 1e-5)

    in_data = pd.read_csv('multi_classes/dev/in.csv', sep=';', header=None)
    X_in = in_data.values

    y_out = predict(X_in, theta)

    expected_values = pd.read_csv('multi_classes/dev/expected.csv', sep=';', header=None)
    y_true, _ = encode_labels(expected_values.values.ravel())
    calculate_metrics(y_true, y_out)

    y_out = decode_labels(y_out, encoder)
    pd.DataFrame(y_out).to_csv('multi_classes/dev/out.csv', index=False, header=False)


main()
