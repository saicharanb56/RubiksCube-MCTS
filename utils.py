import numpy as np
import torch
import os
from rubikscube import Cube

class SaveBestModel:
    """
    Class to save the best policy model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_loss=float('inf')
    ):
        self.best_loss = best_loss
        
    def __call__(
        self, args, current_loss, 
        epoch, model, model_target, optimizer, ce_losses, mse_losses
    ):
        if current_loss < self.best_loss:
            for file in os.listdir(args.save_path):
                if file.startswith('best_policy'):
                    os.remove(os.path.join(args.save_path, file))
            
            self.best_loss = current_loss
            # print(f"Best loss: {self.best_loss}")
            print(f"Saving best model for epoch: {epoch+1}\n")
            state = {'state_dict': model.state_dict(), 'target_state_dict': model_target.state_dict(),
                     'optimizer': optimizer.state_dict(), 'ce_losses': ce_losses, 'mse_losses': mse_losses, 'epoch': epoch}   
            torch.save(state, os.path.join(args.save_path, 'best_policy_checkpoint_' + str(epoch+1) + '.pt'))

def generate_input_states(cube, scrambled_states):
    '''
    Generate input states to network
    Input is of shape (batch_size,) tuples
    Output should be of shape (batch_size, 480) float
    '''
    input_states = torch.empty((len(scrambled_states), 480))

    for i, state in enumerate(scrambled_states):
        cube.set_state(state)
        input_states[i, :] = torch.tensor(cube.representation(),
                                          dtype=torch.float32)

    return input_states


def generate_child_states(args, cube, n_actions, states):
    # generate child states
    child_states = torch.empty((len(states), n_actions, 480),
                                  device=args.device)
    rewards = torch.full((len(states), n_actions, 1),
            fill_value=-1.0, device=args.device)

    for batch in range(len(states)):
        # set state
        cur_state = states[batch]
        cube.set_state(cur_state)

        # for each action in n_actions, we will generate the next_state
        for action in range(n_actions):
            # perform action
            cube.turn(action)

            # define next state, reward
            child_states[batch, action, :] = torch.from_numpy(
                cube.representation()).float()
            
            if cube.solved():
                rewards[batch, action, :] = 1.0

            # set state back to parent state
            cube.set_state(cur_state)
    
    return child_states, rewards


def validate(args, model, ncubes_per_depth=10, nscrambles=30, max_nmoves=50):
    '''
    Validate performance of model
    '''
    cube = Cube.cube_qtm()
    states = []
    # generate ncubes_per_depth states for each depth level
    for l in range(ncubes_per_depth):
        for k in range(nscrambles):
            cube.scramble(k + 1)
            cur_state = cube.get_state()  # parent state for this iteration
            states.append(cur_state)

            # set cube back to solved state
            cube = Cube.cube_qtm()
    
    # count number of times a state gets solved
    solved_count = 0
    total_count = len(states)

    # identify optimal action for each state
    for _ in range(max_nmoves):
        
        # child_states, rewards = generate_child_states(args, cube, n_actions, states) # shape is (batch_size, n_actions, 480)

        input_states = generate_input_states(cube, states)
        input_states = input_states.to(args.device)
        with torch.no_grad():
            # v_out = model.values(child_states)
            # optimal_actions = torch.argmax(rewards + v_out, dim=1)
            
            optimal_actions = torch.argmax(model.prob(input_states), dim=1)
            optimal_actions = optimal_actions.squeeze(-1)
            optimal_actions = optimal_actions.cpu().numpy()

        # make actions and generate new states
        next_states = []
        for i, action in enumerate(optimal_actions):
            cube.set_state(states[i])
            cube.turn(action)

            if cube.solved():
                solved_count += 1
            else:
                next_states.append(cube.get_state())

        # if all states are solved
        if len(next_states) == 0:
            break
        
        states = next_states

    return solved_count/total_count