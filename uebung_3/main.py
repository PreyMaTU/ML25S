
from game import *
from learning import *
from plotting import *
import pygame
import random
import pathlib

GRID_PIXELS= 30
FIELD_WIDTH= 15
FIELD_HEIGHT= 10

FPS= 3

BRICK_LAYOUT_A = [(0,0), (6,0), (12,0), (0,2), (12,2)]
BRICK_LAYOUT_B = [(0,0), (3,0), (6,0), (9,0), (12,0), (0,2), (6,2), (12,2)]
BRICK_LAYOUT_C = [(0,0), (3,0), (6,0), (9,0), (12,0), (0,2), (3,2), (6,2), (9,2), (12,2)]

class PyGameRenderer( Renderer ):
  def __init__(self):
    self.screen= None
    self.clock= None

  def init(self):
    pygame.init()
    self.screen= pygame.display.set_mode((FIELD_WIDTH * GRID_PIXELS, FIELD_HEIGHT * GRID_PIXELS))
    self.clock= pygame.time.Clock()

  def draw_rect(self, x, y, w, h, color):
    rect= pygame.Rect(x * GRID_PIXELS, y* GRID_PIXELS, w* GRID_PIXELS, h* GRID_PIXELS)
    pygame.draw.rect(self.screen, color, rect)

  def begin_frame(self):
    self.screen.fill( (0,0,0) )

  def end_frame(self):
    pygame.display.flip()
    self.clock.tick( FPS )

  def handle_events(self):
    pressed_key= ''

    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        return 'quit'
      
      if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_ESCAPE:
          return 'quit'
        if event.key == pygame.K_LEFT:
          pressed_key= 'left'
        if event.key == pygame.K_RIGHT:
          pressed_key= 'right'
      
    return pressed_key
  

# Play with keyboard
def play_game_interactively():
  renderer= PyGameRenderer()
  renderer.init()

  game= Game(FIELD_WIDTH, FIELD_HEIGHT, BRICK_LAYOUT_B, renderer)
  game.reset()
  game.draw()

  while True:
    event= renderer.handle_events()
    if event == 'quit':
      break

    if event == 'left':
      game.move_paddle_left()
    if event == 'right':
      game.move_paddle_right()

    game.update()
    
    game.draw()

    if game.has_won():
      print('You won!')
      break

  pygame.quit()

# Bot plays and game is shown
def play_game_automatically( policy: Policy, layout ):
  renderer= PyGameRenderer()
  renderer.init()

  print(f'############################\nPlay game')

  game= Game(FIELD_WIDTH, FIELD_HEIGHT, layout, renderer)
  game.reset()
  game.draw()

  step= 0
  while True:
    event= renderer.handle_events()
    if event == 'quit':
      pygame.quit()
      exit()

    action= policy.play_step( game )
    
    print(f'  Step {step}: {action}')

    game.update()
    game.draw()

    if game.has_won():
      print('Game won!')
      break

    step+= 1

  pygame.quit()
  

def play_game_and_plot( policy_name, policy: Policy, layout, layout_id, direction):
  layout_letter= chr(65+ layout_id)
  print(f'Play game on map {layout_letter}, starting direction {direction}')

  game= Game(FIELD_WIDTH, FIELD_HEIGHT, layout, EmptyRenderer())
  game.reset(direction)

  trajectory= [ (game.ball.x, game.ball.y) ]
  still_playing= True
  while still_playing:
    action= policy.play_step( game )

    still_playing, _ = game.update()

    if game.has_won():
      break

    trajectory.append( (game.ball.x, game.ball.y) )

  plot_game_trajectory( policy_name, trajectory, game.has_won(), layout, layout_letter, direction)


def main():

  pathlib.Path('./out').mkdir(parents=True, exist_ok=True)

  random.seed(42)

  layouts= [
    BRICK_LAYOUT_A,
    BRICK_LAYOUT_B,
    BRICK_LAYOUT_C
  ]

  # policy_name= 'policy_cooldown_250k'
  policy_name= 'policy_fixed_250k'
  policy_path= f'./out/{policy_name}.pickle'

  if False:
    policy= Policy()

    training_info= TrainingInfo()
    policy.train(layouts, 250000, training_info= training_info, epsilon_start= 0.1, epsilon_min= 0.1)

    policy.store_to_file( policy_path )

    plot_win_percentage(policy_name, training_info, batch_size=3000)
    plot_step_count_per_win(policy_name, training_info, batch_size=3000)

  else:

    policy = Policy.load_from_file( policy_path )

    # Play each of the 15 possible start (3 layouts * 5 starting directions)
    for layout_id in range( len(layouts) ) :
      for direction in Paddle.Reflections:
        play_game_and_plot(policy_name, policy, layouts[layout_id], layout_id, direction )

        # play_game_automatically( policy, layouts[layout_id] )



if __name__ == '__main__':
  main()

