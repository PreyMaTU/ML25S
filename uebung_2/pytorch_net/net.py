import torch.nn as nn
import torch
import torch.optim as optim

class Net(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_layers):
        print(f"Initializing Net with input_dims={input_dims}, output_dims={output_dims}, hidden_layers={hidden_layers}")
        super(Net, self).__init__()

        layers = []
        prev_dim = input_dims

        # Hidden layers
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dims))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
    @torch.no_grad()
    def predict(self, x):
        return self.model(x)

    
def train_model(model, train_loader, epochs, learning_rate=0.01):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for x, y_true in train_loader:
            outputs = model(x)  # take output of last layer
            loss = loss_fn(outputs,y_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        if epoch % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}")


def test_model(model, test_loader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y_true in test_loader:
            outputs = model(x)  # take output of last layer
            y_pred = torch.argmax(outputs,1)

            total += y_true.size(0)
            correct += (y_pred == y_true).sum().item()

    print(f"\nAccuracy on test set: {100 * correct / total:.2f}%")