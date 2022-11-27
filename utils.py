import os

import numpy as np
import torch
from rubikscube import Cube


class SaveBestModel:
    """
    Class to save the best policy model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(self, best_loss=float('inf')):
        self.best_loss = best_loss

    def __call__(self, args, current_loss, epoch, model, model_target,
                 optimizer, ce_losses, mse_losses):
        if current_loss < self.best_loss:
            for file in os.listdir(args.save_path):
                if file.startswith('best_policy'):
                    os.remove(os.path.join(args.save_path, file))

            self.best_loss = current_loss
            # print(f"Best loss: {self.best_loss}")
            print(f"Saving best model for epoch: {epoch+1}\n")
            state = {
                'state_dict': model.state_dict(),
                'target_state_dict': model_target.state_dict(),
                'optimizer': optimizer.state_dict(),
                'ce_losses': ce_losses,
                'mse_losses': mse_losses,
                'epoch': epoch
            }
            torch.save(
                state,
                os.path.join(
                    args.save_path,
                    'best_policy_checkpoint_' + str(epoch + 1) + '.pt'))


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
                         fill_value=-1.0,
                         device=args.device)

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
    solved_count = np.zeros(nscrambles)
    # generate ncubes_per_depth states for each depth level
    for k in range(1, nscrambles + 1):
        for _ in range(ncubes_per_depth):
            cube = Cube.cube_qtm()
            cube.scramble(k)

            for _ in range(max_nmoves):
                repr = cube.representation()
                repr_tensor = torch.tensor(repr,
                                           dtype=torch.float32).to(args.device)
                best_action = torch.argmax(model.logits(repr_tensor)).item()

                cube.turn(best_action)

                if cube.solved():
                    solved_count[k] += 1
                    break

    return solved_count / ncubes_per_depth
