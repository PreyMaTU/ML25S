
import re
from numbers import Number
from matplotlib import pyplot as plt

class TrainingTrace:
  def __init__(self, name: str):
    self.name= name
    self.epochs= []
    self.loss= None
    self.accuracy= None

  def set(self, epochs: list|None, loss: list|None, accuracy: list|None):
    if loss and accuracy and len(loss) != len(accuracy):
      raise ValueError(f'Got different number of loss and accuracy entries ({len(loss)} vs {len(accuracy)})')
    
    if not loss and not accuracy:
      return
    
    if not epochs:
      count= len(loss) if loss else len(accuracy)
      epochs= [i+1 for i in range(count)]

    self.epochs= epochs
    self.loss= loss
    self.accuracy= accuracy

  def append(self, epoch: Number, loss: Number, accuracy: Number):
    if not self.loss:
      self.loss= []
      self.accuracy= []

    if epoch is None:
      epoch= len(self.loss)+ 1

    self.epochs.append(epoch)
    self.loss.append(loss)
    self.accuracy.append(accuracy)

only_word_chars_pattern = re.compile('[^\\w ]')
def snakeify( text, sep= '_' ):
  return re.sub(only_word_chars_pattern, '', text.lower()).replace(' ', sep)

def export_current_plot_with_title_name( title ):
  path= f"./out/{snakeify(title)}"

  print('Exporting plot:', path)
  plt.savefig(path)
  plt.clf()

def plot_loss(title: str, traces: list[TrainingTrace], filter: list[str]= None):
  for trace in traces:
    if not filter or trace.name in filter:
      line, = plt.plot(trace.epochs, trace.loss)
      line.set_label( trace.name )

  plt.title(title)
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend(framealpha = 0.6)

  export_current_plot_with_title_name( title )

def plot_accuracy(title: str, traces: list[TrainingTrace], filter: list[str]= None):
  for trace in traces:
    if not filter or trace.name in filter:
      line, = plt.plot(trace.epochs, trace.accuracy)
      line.set_label( trace.name )

  plt.title(title)
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend(framealpha = 0.6)

  export_current_plot_with_title_name( title )

