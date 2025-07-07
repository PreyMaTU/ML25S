import random
import math

PADDLE_HEIGHT= 1
PADDLE_WIDTH= 5
PADDLE_COLOR = (0, 150, 255)

BRICK_HEIGHT= 1
BRICK_WIDTH= 3
BRICK_COLOR = (0, 163, 108)

BALL_COLOR= (255, 255, 255)

class Renderer:
  def init(self):
    pass

  def draw_rect(self, x, y, w, h, color):
    pass

  def begin_frame(self):
    pass

  def end_frame(self):
    pass


class Ball:
  def __init__(self):
    self.x= 0
    self.y= 0
    self.dx= 0
    self.dy= 0

  def reset_randomly(self, width, height):
    self.y= height - PADDLE_HEIGHT - 1
    self.x= math.floor( width / 2 )

    (dx, dy)= random.choice( Paddle.Reflections )
    self.dx= dx
    self.dy= dy

  def reflect_y(self):
    self.dy *= -1

  def reflect_x(self):
    self.dx *= -1

  def update(self):
    self.x += self.dx
    self.y += self.dy

  def draw(self, renderer: Renderer):
    renderer.draw_rect( self.x, self.y, 1, 1, BALL_COLOR)


class Paddle:
  Reflections = [(-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1)]

  def __init__(self):
    self.x= 0
    self.y= 0
    self.dx= 0

  def reset(self, width, height):
    self.y= height - PADDLE_HEIGHT
    self.x= round( width / 2 - PADDLE_WIDTH / 2 )

    self.dx= 0

  def move(self, direction):
    # Check maximum speed of paddle
    self.dx = min( 2, max( -2, self.dx + direction ) )

  def reflect_ball(self, ball: Ball):
    if self.x <= ball.x < self.x + PADDLE_WIDTH and ball.y == self.y - 1:
      offset= min( ball.x - self.x, len(Paddle.Reflections) )
      (dx, dy)= Paddle.Reflections[offset]
      ball.dx= dx
      ball.dy= dy

  def update(self, width):
    # Check gamefield boundaries
    self.x= min( width - PADDLE_WIDTH, max( 0, self.x + self.dx ) )

    if self.x <= 0 or self.x + PADDLE_WIDTH >= width:
      self.dx= 0


  def draw(self, renderer: Renderer):
    renderer.draw_rect(self.x, self.y, PADDLE_WIDTH, PADDLE_HEIGHT, PADDLE_COLOR)

class Game:
  def __init__(self, width: int, height: int, brick_layout: list[(int,int)], renderer: Renderer):
    self.width= width
    self.height= height

    self.ball= Ball()
    self.paddle= Paddle()
    self.renderer= renderer

    self.brick_layout= brick_layout
    self.bricks= []

  def reset(self):
    self.bricks= self.brick_layout.copy()

    self.ball.reset_randomly(self.width, self.height)
    self.paddle.reset(self.width, self.height)

  def move_paddle_right(self):
    self.paddle.move( +1 )

  def move_paddle_left(self):
    self.paddle.move( -1 )

  def update(self):
    self.paddle.update( self.width )
    self.ball.update()

    # Player looses when the ball moves past the paddle
    if self.ball.y >= self.height-1:
      self.reset()
      return

    # Check paddle collision
    self.paddle.reflect_ball( self.ball )

    # Check block collisions
    for i in range(len(self.bricks)):
      (x,y)= self.bricks[i]
      if y <= self.ball.y < y + BRICK_HEIGHT and x <= self.ball.x < x+ BRICK_WIDTH:
        self.ball.reflect_y()
        self.bricks.pop( i )
        break
        
    # Ball gets reflected on all other walls
    if self.ball.y + self.ball.dy < 0:
      self.ball.reflect_y()

    if not (0 <= self.ball.x + self.ball.dx <= self.width - 1):
      self.ball.reflect_x()
    
    
  def has_won(self):
    return len(self.bricks) < 1
      


  def draw(self):
    self.renderer.begin_frame()

    for (x, y) in self.bricks:
      self.renderer.draw_rect(x, y, BRICK_WIDTH, BRICK_HEIGHT, BRICK_COLOR)

    self.paddle.draw( self.renderer )
    self.ball.draw( self.renderer )

    self.renderer.end_frame()
