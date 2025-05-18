
from scratch_net.data import train_test_split
from scratch_net.activation import ReLu, Sigmoid, Tanh
from scratch_net.net import Net, Layer
from scratch_net.optimizer import SGD,Adam
from scratch_net.loss import MSE

import numpy as np

np.random.seed(42)

# Generate training data
num_samples = 1000
x_coords = np.random.uniform(-10, 10, size=(num_samples, 1))
y_coords = np.random.uniform(-10, 10, size=(num_samples, 1))

x = np.hstack((x_coords, y_coords))  # shape: (2, 1000)

# Define the target line
# Label = 1 if point is above the line, 0 if below
line_y = 0.5 * x_coords + 2
labels = (y_coords > line_y).astype(float)

# Initialize and train network
net = Net(
    layers=[
        Layer(1, Sigmoid(), input_layer_size=2)
    ],
    loss_function=MSE()
)

net.train(x, labels, optimizer=None, epochs=1000, learning_rate=0.05, verbose=True)

# Test on new unseen data
np.random.seed(99)  
x_test_coords = np.random.uniform(-10, 10, size=(num_samples, 1))
y_test_coords = np.random.uniform(-10, 10, size=(num_samples, 1))
x_test = np.hstack((x_test_coords, y_test_coords))
line_y_test = 0.5 * x_test_coords + 2
y_test_labels = (y_test_coords > line_y_test).astype(float)


y_pred = net.predict(x_test)
y_pred_labels = (y_pred > 0.5).astype(float)

# Evaluation
accuracy = np.mean(y_pred_labels == y_test_labels)
print(f"\nAccuracy on test set: {accuracy * 100:.2f}%")
