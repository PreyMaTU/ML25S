import numpy as np

class LossFunction:
  def reset(self):
    pass

  # Loss
  def forward(self, y_true, y_pred):
    return 0

  # Derivative
  def backward(self, y_true, y_pred):
    return 0

# Mean Squared Error
class MSE(LossFunction):
  def forward(self, y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)
  
  def backward(self, y_true, y_pred):
    return 2.0 * (y_pred - y_true)



class CrossEntropyLoss(LossFunction):
  def forward(self, y_true, y_pred):

    # Subtract max logits per sample -> improves numerical stability, 
    logits_stable = y_pred - np.max(y_pred, axis=0, keepdims=True)

    # Compute log softmax
    log_probs = logits_stable - np.log(np.sum(np.exp(logits_stable), axis=0, keepdims=True))

    # Compute cross entropy over samples
    loss = -np.sum(y_true * log_probs) / y_true.shape[1]
    return loss

  def backward(self, y_true, y_pred):
    logits_stable = y_pred - np.max(y_pred, axis=0, keepdims=True)

  	# Compute gradient
    exp_scores = np.exp(logits_stable)
    probs = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)

    # Gradient of CE loss over samples
    grad = (probs - y_true) / y_true.shape[1]
    return grad