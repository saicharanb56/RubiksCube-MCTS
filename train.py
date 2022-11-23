import numpy as np
from torch import nn, optim
import torch
from model import NNet
import argparse
from rubikscube import Cube

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', default=10000, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--niter', default=1000, type=int, help='Number of ADI iterations M')
parser.add_argument('--nscrambles', default=100, type=int, help='Number of scrambles of cube')
parser.add_argument('--nstates', default=100, type=int, help='Number of scrambled cubes N')
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--wd', default=0, type=int, help="Weight decay")
parser.add_argument('--momentum', default=0, type=int, help="Momentum")
parser.add_argument('--device', default="cuda", type=str)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)

def soft_update(local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        =======
            local model (PyTorch model): weights will be copied from
            target model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(),
                                           local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)

def adi(args, model, model_target, cube, lossfn_prob, lossfn_val, optimizer, n_actions=12):
    '''
    Performs Autodidactic iteration and trains the neural network
    Args:
    args:           from argparser
    model:          neural network model
    model_target:   target neural network model
    cube:            Rubiks cube environment
    lossfn_prob:    Loss function of probs
    lossfn_val:     Loss function of value
    optimizer:      Optimizer (RMSProp)
    n_actions:      Quarter-turn metric, hence, n_actions is 12 as default
    '''
    assert n_actions == 12 or n_actions == 18
    assert cube.solved()

    # save solved state
    solved_state = cube.get_state()

    for iter in range(args.niter):
        # initialize list of labels. This will be a list of dictionaries.
        # each dict will have keys "value" and "probs" to be used during training 
        labels = []

        # generate N scrambled states
        scrambled_states = []

        for j in range(args.nstates):
            # intialize probs, values for all child states of state
            p_all_actions = torch.zeros((n_actions, n_actions))
            v_all_actions = torch.zeros(n_actions)
            
            # initialize rewards array
            rewards_all_actions = torch.zeros(n_actions)

            # define state, current_state is stored in env instance
            cube.scramble(args.nscrambles)
            cur_state = cube.get_state()
            scrambled_states.append(cur_state)

            # for each action in n_actions, we will generate the next_state
            model.eval()
            for action in range(n_actions):
                # perform action
                cube.turn(action)
                
                # define next state, reward
                next_state = torch.tensor(cube.representation(), dtype=torch.float32, device=args.device)
                reward = 1 if cube.solved() else -1

                # forward pass
                p_action, v_action = model(next_state)

                # update array elements
                p_all_actions[:,action] = p_action
                v_all_actions[action] = v_action
                rewards_all_actions[action] = reward

                # set state back to initial state
                cube.set_state(cur_state)

            # set labels to be maximal value from each children state
            v_label = torch.max(rewards_all_actions + v_all_actions)
            idx = torch.argmax(rewards_all_actions + v_all_actions)
            p_label = p_all_actions[:,idx]

            labels.append({"value": v_label, "probs": p_label})

            # set cube back to solved state
            cube.set_state(solved_state)

        # initialize weights using Glorot/Xavier initialization
        init_weights(model)

        # training loop
        model.train()
        for i in range(args.nstates):

            optimizer.zero_grad()

            state = scrambled_states[i]
            cube.set_state(state)
            input_state = torch.tensor(cube.representation(), dtype=torch.float32, device=args.device)
            value, probs = labels[i]["value"], labels[i]["probs"]

            value, probs = value.to(args.device), probs.to(args.device)

            probs_pred, val_pred = model(input_state)
            loss_prob = lossfn_prob(probs_pred, probs)
            loss_val = lossfn_val(val_pred, value)

            loss_prob.backward()
            loss_val.backward()

            optimizer.step()

        soft_update(model, model_target, tau=0.001)

    return model


if __name__ == "__main__":

    args = parser.parse_args()
    device = torch.device(args.device)

    # Instantiate model, optimizer, loss functions
    model = NNet()
    model = model.to(device)

    model_target = NNet()
    model_target = model_target.to(device)

    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
    lossfn_val = nn.MSELoss()
    lossfn_prob = nn.CrossEntropyLoss()

    #Instantiate env
    cube = Cube.cube_qtm()

    print('ADI started')
    model = adi(args, model, model_target, cube, lossfn_prob, lossfn_val, optimizer)
    print('ADI done')

    # run the MCTS tree search here
