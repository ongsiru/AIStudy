import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
from autosklearn.classification import AutoSklearnClassifier
import matplotlib.pyplot as plt

# Load Cifar10 data
(x_train_orig, y_train_orig), (x_test_orig, y_test_orig) = datasets.cifar10.load_data()

def preprocess_data(x_train, x_test, y_train, y_test, num_classes, imbalance=False):
    # Preprocess data
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    if imbalance:
        # Create imbalanced dataset
        x_train, _, y_train, _ = train_test_split(x_train, y_train, test_size=0.5, stratify=y_train)

    return x_train, x_test, y_train, y_test

def create_model(input_shape, num_classes):
    # Create a simple CNN model
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def train_and_evaluate(x_train, x_test, y_train, y_test, num_classes, model=None):
    if model is None:
        # Create and compile the model
        model = create_model(x_train.shape[1:], num_classes)

    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

    # Evaluate the model on test data
    _, accuracy = model.evaluate(x_test, y_test)
    print(f'Top-1 Accuracy: {accuracy * 100:.2f}%')

    return accuracy, model

def autosklearn_train_and_evaluate(x_train, x_test, y_train, y_test, num_classes):
    # Flatten the input data
    x_train_flatten = x_train.reshape(x_train.shape[0], -1)
    x_test_flatten = x_test.reshape(x_test.shape[0], -1)

    # Create and fit AutoSklearn classifier
    automl = AutoSklearnClassifier(time_left_for_this_task=120, per_run_time_limit=30, ensemble_size=1)
    automl.fit(x_train_flatten, y_train.argmax(axis=1))

    # Get the best model from AutoSklearn
    best_model = automl.show_models()[0]

    # Evaluate the best model on test data
    accuracy = best_model.score(x_test_flatten, y_test.argmax(axis=1))
    print(f'Top-1 Accuracy (AutoSklearn): {accuracy * 100:.2f}%')

    return accuracy, best_model

# A. Imbalanced data samples for Cifar10
x_train_imbalanced, x_test_imbalanced, y_train_imbalanced, y_test_imbalanced = preprocess_data(
    x_train_orig, x_test_orig, y_train_orig, y_test_orig, num_classes=10, imbalance=True
)
print("Cifar10 - Imbalanced Data:")
baseline_acc_imbalanced, baseline_model_imbalanced = train_and_evaluate(
    x_train_imbalanced, x_test_imbalanced, y_train_imbalanced, y_test_imbalanced, num_classes=10
)
autosklearn_acc_imbalanced, autosklearn_model_imbalanced = autosklearn_train_and_evaluate(
    x_train_imbalanced, x_test_imbalanced, y_train_imbalanced, y_test_imbalanced, num_classes=10
)

# B. Balanced data samples for Cifar10
x_train_balanced, x_test_balanced, y_train_balanced, y_test_balanced = preprocess_data(
    x_train_orig, x_test_orig, y_train_orig, y_test_orig, num_classes=10, imbalance=False
)
print("Cifar10 - Balanced Data:")
baseline_acc_balanced, baseline_model_balanced = train_and_evaluate(
    x_train_balanced, x_test_balanced, y_train_balanced, y_test_balanced, num_classes=10
)
autosklearn_acc_balanced, autosklearn_model_balanced = autosklearn_train_and_evaluate(
    x_train_balanced, x_test_balanced, y_train_balanced, y_test_balanced, num_classes=10
)

# C. Cifar100 with 10 selected categories
selected_categories = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
x_train_100, x_test_100, y_train_100, y_test_100 = preprocess_data(
    x_train_100_orig, x_test_100_orig, y_train_100_orig, y_test_100_orig, num_classes=100
)
# Filter only selected categories
train_mask = np.isin(y_train_100.argmax(axis=1), selected_categories)
test_mask = np.isin(y_test_100.argmax(axis=1), selected_categories)
x_train_100 = x_train_100[train_mask]
y_train_100 = y_train_100[train_mask]
x_test_100 = x_test_100[test_mask]
y_test_100 = y_test_100[test_mask]
print("Cifar100 - 10 Selected Categories (Balanced Data):")
baseline_acc_cifar100, baseline_model_cifar100 = train_and_evaluate(
    x_train_100, x_test_100, y_train_100, y_test_100, num_classes=100
)
autosklearn_acc_cifar100, autosklearn_model_cifar100 = autosklearn_train_and_e
