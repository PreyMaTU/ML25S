import matplotlib.pyplot as plt
import re
import math
from learning import TrainingInfo

only_word_chars_pattern = re.compile('[^\\w ]')
def snakeify( text, sep = '_' ):
  return re.sub(only_word_chars_pattern, '', text.lower()).replace(' ', sep)

def export_current_plot_with_title_name( title ):
  path = f"./out/{snakeify(title)}"

  print('Exporting plot:', path)
  plt.savefig(path)
  plt.clf()

def plot_win_percentage(policy_name: str, training_info: TrainingInfo, batch_size = 100):
  percent_batches = []
  # Create batches of max batch_size episodes
  for win_batch, _ in training_info.batches( batch_size ):
    win_count = 0
    for w in win_batch:
      win_count += 1 if w else 0

    percent = 100* win_count / len(win_batch)
    percent_batches.append( percent )

  line, = plt.plot(percent_batches)
  line.set_label( 'Wins [%]' )

  title = 'Episode Wins per Batch'
  plt.title( title )
  plt.xlabel(f'Batch (Size = {batch_size})')
  plt.ylabel('Win Percentage')
  plt.legend(framealpha = 0.6)

  export_current_plot_with_title_name( f'{policy_name} {title}' )

def plot_step_count_per_win(policy_name: str, training_info: TrainingInfo, batch_size = 100):
  avg_steps_batches = []
  # Create batches of max batch_size episodes
  for win_batch, step_count_batch in training_info.batches( batch_size ):
    steps_accu= 0
    win_count= 0
    for w, s in zip(win_batch, step_count_batch):
      if w:
        win_count += 1
        steps_accu += s

    average_steps= steps_accu / win_count if win_count > 0 else 0
    avg_steps_batches.append( average_steps )

  line, = plt.plot(avg_steps_batches)
  line.set_label( 'Avg. Step Count' )

  title= 'Average Step Count per Batch'
  plt.title( title )
  plt.xlabel(f'Batch (Size = {batch_size})')
  plt.ylabel('Step Count')
  plt.legend(framealpha = 0.6)

  export_current_plot_with_title_name( f'{policy_name} {title}' )
