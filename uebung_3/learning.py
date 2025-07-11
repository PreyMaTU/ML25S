from game import *

import pickle

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
    # Update the action's average value gain
    self.actions[action.action].append( gain )

    # Select the action with the best average value gain as the policy
    avg= self.actions[0].get( -math.inf )
    act= 0
    for i in range(len(self.actions)):
      new_avg= self.actions[i].get( -math.inf )
      if new_avg > avg:
        avg= new_avg
        act= i

    self.policy= Action( act )

  def get_policy(self, epsilon: float = 0):
    if epsilon == 0 or random.random() < (1- epsilon):
      return self.policy
    
    return Action.make_random()

class TrainingInfo:
  def __init__(self):
    self.episode_wins= []
    self.episode_step_count= []

  def append(self, did_win, step_count):
    self.episode_wins.append( did_win )
    self.episode_step_count.append( step_count )

  def batches(self, batch_size):
    return TrainingInfoBatcher( self, batch_size )

class TrainingInfoBatcher:
  def __init__(self, training_info: TrainingInfo, batch_size: int):
    self.training_info= training_info
    self.batch_size= batch_size
    self.batch_count= math.ceil( len(training_info.episode_wins) / batch_size )
    self.batch_index= 0

  def __iter__(self):
    self.batch_index = 0
    return self
  
  def __next__(self):
    if self.batch_index >= self.batch_count:
      raise StopIteration

    start= self.batch_index * self.batch_size
    end= (self.batch_index + 1) * self.batch_size
    batch_wins= self.training_info.episode_wins[start:end]
    batch_step_count= self.training_info.episode_step_count[start:end]

    self.batch_index += 1

    return batch_wins, batch_step_count


class Policy:
  def __init__(self):
    self.value_map= dict()

  def load_from_file( path: str ) -> 'Policy':
    with open(path, 'rb') as file:
      return pickle.load( file )

  def store_to_file(self, path: str):
    with open(path, 'wb') as file:
      pickle.dump( self, file )

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

  def generate_episode(self, game: Game, epsilon: float, max_length= 2500):
    game.reset()
    state= game.to_state()

    states= []
    actions= []
    rewards= []

    # Limit runtime of game (prevent getting stuck with no progress)
    for i in range(max_length):
      action= self.select_action( state, epsilon )

      states.append( state )
      actions.append( action )
      rewards.append( -i-1 )

      action.apply( game )

      still_playing, state= game.update()

      # When game is 'lost', give agent a big punishment and stop episode
      if not still_playing:
        rewards[ -1 ] -= max_length 
        break

      if game.has_won():
        break
    
    # Add additional punishment if game got stuck
    if not game.has_won():
      rewards[ -1 ] *= 1.1

    return states, actions, rewards

  def train(self, brick_layouts: list[list[tuple[int,int]]], episode_count: int, gamma: float= 0.92, epsilon_start: float= 0.5, epsilon_min: float= 0.02, verbose= True, training_info: TrainingInfo|None= None):
    self.reset()

    episodes_with_win= 0

    epsilon_decay = 0
    if epsilon_start != epsilon_min:
      epsilon_decay= (1.0 / episode_count) * math.log((epsilon_start - epsilon_min) / (0.0001))

    for i in range(episode_count):
      # Run the game with the current policy
      brick_layout= brick_layouts[ i % len(brick_layouts) ]
      game= Game(FIELD_WIDTH, FIELD_HEIGHT, brick_layout, EmptyRenderer())

      epsilon= epsilon_min + (epsilon_start - epsilon_min) * math.exp(-epsilon_decay * i)
      states, actions, rewards= self.generate_episode( game, epsilon )

      if training_info:
        training_info.append( game.has_won(), len(states) )

      # Print progress
      episodes_with_win += 1 if game.has_won() else 0
      if verbose and (i % 100 == 0 or i == episode_count - 1):
        episodes= 100 if i % 100 == 0 else i % 100
        print(f'Episode {i}: {episodes_with_win}/{episodes} games won (last steps: {len(states)}, epsilon: {epsilon}, state map: {len(self.value_map.keys())})')
        episodes_with_win= 0

      gain= 0
      seen_state_actions= set()

      # Update the policy map based on this game run
      for state, action, reward in zip(reversed(states), reversed(actions), reversed(rewards)):
        gain= gain * gamma + reward

        # Only update each state-action-pair once per run even if it occurs multiple times
        if not (state, action) in seen_state_actions:
          seen_state_actions.add( (state, action) )

          action_space= self.get_action_space( state )
          action_space.update( action, gain )

  # Play the game with the best action (=policy, no exploration)
  def play_step(self, game: Game):
    state= game.to_state()
    action= self.select_action( state, 0 )
    action.apply( game )
    
    return action
