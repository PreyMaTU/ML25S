
from game import *
from learning import *
import pygame
import random

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

def main():
  random.seed(42)

  layouts= [
    BRICK_LAYOUT_A,
    #BRICK_LAYOUT_B,
    #BRICK_LAYOUT_C
  ]

  policy= Policy()

  policy.train(layouts, 8000)

  renderer= PyGameRenderer()
  renderer.init()

  for i in range(100):
    print(f'############################\nPlay game {i}')

    game= Game(FIELD_WIDTH, FIELD_HEIGHT, BRICK_LAYOUT_A, renderer)
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

if __name__ == '__main__':
  main()

