from game import *

FIELD_WIDTH= 15
FIELD_HEIGHT= 10

class EmptyRenderer(Renderer):
  pass

class IncrementalAverage:
  def __init__(self, init: float= 0):
    self.sum= init
    self.count= 0

  def append(self, x: float):
    self.sum += x
    self.count += 1

  def get(self, default: float= 0):
    return self.sum / self.count if self.count else default
  
  def __repr__(self):
    return f'(avg {self.get()})'

class Action:
  NOOP= 0
  MOVE_LEFT= 1
  MOVE_RIGHT= 2

  def __init__(self, x: int= 0):
    self.action= x

  def make_random():
    return Action( random.choice([
      Action.NOOP,
      Action.MOVE_LEFT,
      Action.MOVE_RIGHT
    ]) )
  
  def apply(self, game: Game):
    if self.action == Action.NOOP:
      pass
    elif self.action == Action.MOVE_LEFT:
      game.move_paddle_left()
    elif self.action == Action.MOVE_RIGHT:
      game.move_paddle_right()

  def __hash__(self):
    return hash( self.action )

  def __eq__(self, other):
    return self.action == other.action
  
  def __repr__(self):
    if self.action == Action.NOOP:
      return 'NOOP'
    elif self.action == Action.MOVE_LEFT:
      return 'MOVE_LEFT'
    elif self.action == Action.MOVE_RIGHT:
      return 'MOVE_RIGHT'
    else:
      return f'<Invalid action: {self.action}>'

# In each state we have 3 possible actions to take
class StateActionSpace:
  def __init__(self):
    self.actions= [ IncrementalAverage(), IncrementalAverage(), IncrementalAverage()]
    self.policy= Action.make_random()

  def update(self, action: Action, gain: float):
    self.actions[action.action].append( gain )

    avg= self.actions[0].get( -math.inf )
    act= 0
    for i in range(len(self.actions)):
      new_avg= self.actions[i].get( -math.inf )
      if new_avg > avg:
        avg= new_avg
        act= i

    # print( self.actions, '-> picked:', act )

    self.policy= Action( act )

  def get_policy(self, epsilon: float = 0):
    if epsilon == 0 or random.random() < (1- epsilon):
      return self.policy
    
    return Action.make_random()

class Policy:
  def __init__(self):
    self.value_map= dict()

  def reset(self):
    self.value_map= dict()

  def get_action_space(self, state: State):
    x= self.value_map.get( state )
    if x:
      return x
    
    x= StateActionSpace()
    self.value_map[ state ]= x
    return x

  def select_action(self, state: State, epsilon: float):
    action_space= self.get_action_space( state )
    return action_space.get_policy( epsilon )

  def generate_episode(self, game: Game, epsilon: float=0.1, max_length= 2500):
    game.reset()
    state= game.to_state()

    states= []
    actions= []
    rewards= []

    for i in range(max_length):
      action= self.select_action( state, epsilon )

      # print('Do action:', action)

      states.append( state )
      actions.append( action )
      rewards.append( -i-1 )

      action.apply( game )

      still_playing, state= game.update()

      #TODO: Maybe break game/episode more intelligently?
      #if not still_playing:
        # Assign negative reward
        #break

      if game.has_won():
        break

    if not game.has_won():
      rewards[ -1 ] *= 1.1

    return states, actions, rewards

  def train(self, brick_layouts: list[list[(int,int)]], episode_count: int, gamma: float= 0.9, verbose= True):
    self.reset()

    for i in range(episode_count):
      brick_layout= brick_layouts[ i % len(brick_layouts) ]
      game= Game(FIELD_WIDTH, FIELD_HEIGHT, brick_layout, EmptyRenderer())

      states, actions, rewards= self.generate_episode( game )

      if verbose and (i % 20 == 0 or i == episode_count - 1):
        if not game.has_won():
          print(f'Episode {i}: Game lost after {len(states)} steps (state map: {len(self.value_map.keys())})')

        else:
          print(f'Episode {i}: Game won after {len(states)} steps (state map: {len(self.value_map.keys())})')

      gain= 0
      seen_state_actions= set()

      for state, action, reward in zip(reversed(states), reversed(actions), reversed(rewards)):
        gain= gain * gamma + reward

        if not (state, action) in seen_state_actions:
          seen_state_actions.add( (state, action) )

          action_space= self.get_action_space( state )
          action_space.update( action, gain )

          # print(f'Updating state, new policy:', action_space.policy)

  def play_step(self, game: Game):
    state= game.to_state()
    action= self.select_action( state, 0 )
    action.apply( game )
    
    return action
