import matplotlib.pyplot as plt
import re
import math
import numpy as np
from game import PADDLE_WIDTH, PADDLE_HEIGHT, BRICK_WIDTH, BRICK_HEIGHT
from learning import TrainingInfo, FIELD_HEIGHT, FIELD_WIDTH
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Rectangle

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


def plot_game_trajectory(policy_name: str, trajectory: list[tuple[int,int]], game_won: bool, layout: list[tuple[int,int]], layout_letter: str, direction: tuple[int,int]):
  # Convert list of tuples to numpy array of shape (N, 2)
  ball_positions = np.array(trajectory, dtype= np.float64)
  indices = np.arange(len(ball_positions))
  ball_positions[:, 0] += 0.35 + indices * (0.3/indices[-1]) # offset for x values
  ball_positions[:, 1] += 0.5 # offset for y values

  # Create segments between consecutive points
  points = ball_positions.reshape(-1, 1, 2)
  segments = np.concatenate([points[:-1], points[1:]], axis=1)

  # Set up color gradient (blue to red) using 'plasma' colormap
  cmap = plt.get_cmap('plasma')
  norm = plt.Normalize(0, len(segments))  # Normalize over time

  # Create a line collection for the segments
  lc = LineCollection(segments, cmap=cmap, norm=norm, alpha= 0.6)
  lc.set_array(np.arange(len(segments)))  # Color by segment index
  lc.set_linewidth(2)

  # Plot
  fig, ax = plt.subplots(figsize=(7.5, 5))
  ax.add_collection(lc)

  rect = Rectangle((5, 9), PADDLE_WIDTH, PADDLE_HEIGHT, linewidth=1, edgecolor=('black', 0.3), facecolor=('black', 0.15))
  ax.add_patch(rect)

  for (x,y) in layout:
    rect = Rectangle((x, y), BRICK_WIDTH, BRICK_HEIGHT, linewidth=2, edgecolor=('green', 0.4), facecolor=('green', 0.3))
    ax.add_patch(rect)

  ax.xaxis.set_major_locator(MultipleLocator(1)) # draw full grid
  ax.yaxis.set_major_locator(MultipleLocator(1))
  ax.set_xlim(-0.5, FIELD_WIDTH + 0.5)
  ax.set_ylim(-0.5, FIELD_HEIGHT - 0.5)
  ax.invert_yaxis()  # Y=0 is at the top in Atari screens
  ax.set_xlabel("X-Position")
  ax.set_ylabel("Y-Position")
  title= f"Ball Trajectory with Map-Layout {layout_letter} and Direction {direction}"
  plt.suptitle(title)
  ax.set_title(f"Game {"won" if game_won else "lost" } after {len(trajectory)} steps")
  plt.colorbar(lc, ax=ax, label="Time Step")
  plt.grid(True)
  plt.tight_layout()
  # plt.show()

  export_current_plot_with_title_name( f'{policy_name} {title}' )
