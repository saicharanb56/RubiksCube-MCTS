import numpy as np
from torch import nn, optim
import torch
from model import NNet
import argparse
import os
import time
from rubikscube import Cube

from utils import SaveBestPolicyModel, SaveBestValueModel

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', default=10000, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--nepochs', default=200, type=int, help='Number of ADI epochs (M)')
parser.add_argument('--nscrambles', default=25, type=int, help='Number of scrambles of cube (k)')
parser.add_argument('--nsequences', default=4000, type=int, help='Number of sequences of scrambles (l)')
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--wd', default=0, type=int, help="Weight decay")
parser.add_argument('--momentum', default=0, type=int, help="Momentum")
parser.add_argument('--tau', default=0.1, type=float, help="Interpolation parameter in soft update")
parser.add_argument('--device', default="cuda", type=str)

parser.add_argument('--resume_path', default=None, type=str, help="Path of model dict to resume from")
parser.add_argument('--save_path', default="/home/saicharanb56/RubiksCube-MCTS/results/", type=str, help="Folder in which results are stored")

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

def generate_scrambled_states(args, cube):
    '''
    Generate and return N = K*L scrambled states as tuples
    '''
    if not cube.solved():
        cube = Cube.cube_qtm()

    scrambled_states = []
    # generate list of scrambled_states
    for _ in range(args.nsequences):
        # define state, current_state is stored in env instance
        for _ in range(args.nscrambles):
            cube.scramble(1)
            cur_state = cube.get_state() # parent state for this iteration
            # scrambled_states[idx, :] = torch.from_numpy(cube.representation()).float()
            scrambled_states.append(cur_state)

        # set cube back to solved state
        cube = Cube.cube_qtm()

    return scrambled_states

def generate_input_states(cube, scrambled_states):
    '''
    Generate input states to network
    Input is of shape (batch_size,) tuples
    Output should be of shape (batch_size, 480) float
    '''
    input_states = torch.empty((len(scrambled_states), 480))

    for i in range(len(scrambled_states)):
        cube.set_state(scrambled_states[i])
        input_states[i, :] = torch.tensor(cube.representation(), dtype=torch.float32)

    return input_states

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

    # instantiate saveBestModels
    saveBestPolicyModel = SaveBestPolicyModel()
    saveBestValModel = SaveBestValueModel()

    # initialize weights using Glorot/Xavier initialization
    if args.resume_path:
        resume_state = torch.load(args.resume_path, map_location=args.device)
        model.load_state_dict(resume_state['state_dict'])
        optimizer.load_state_dict(resume_state['optimizer'])
        losses_ce = resume_state['ce_losses']
        losses_mse = resume_state['mse_losses']
        startEpoch = resume_state['epoch'] + 1
    else:
        init_weights(model)
        init_weights(model_target)
        losses_ce = []
        losses_mse = []
        startEpoch = 0

    for epoch in range(startEpoch, args.nepochs):
        tic = time.time()

        batch_size = args.nscrambles*args.nsequences

        # generate N = K*L scrambled states
        scrambled_states = generate_scrambled_states(args, cube)
        
        next_states = torch.empty((batch_size, n_actions, 480), device=args.device)
        for l in range(batch_size):
            # set state
            cur_state = scrambled_states[l]
            cube.set_state(cur_state)

            # initialize probs, values, rewards for all child states (next_states) of state
            # p_all_actions = torch.empty((batch_size, n_actions, n_actions))
            # v_all_actions = torch.empty(batch_size, n_actions)
            rewards_all_actions = torch.zeros(batch_size, n_actions, 1, device=args.device)

            # for each action in n_actions, we will generate the next_state
            for action in range(n_actions):
                # perform action
                cube.turn(action)
                
                # define next state, reward
                next_states[l, action, :] = torch.from_numpy(cube.representation()).float()
                rewards_all_actions[l, action] = 1 if cube.solved() else -1

                # set state back to parent state
                cube.set_state(cur_state)

        # forward pass
        with torch.no_grad():
            p_out, v_out = model(next_states)
            # p_out shape is (batch_size, n_actions, n_actions)
            # v_out shape is (batch_size, n_actions, 1)

            # set labels to be maximal value from each children state
            v_label, idx = torch.max(rewards_all_actions + v_out, dim=1)
            idx = idx.repeat(1,n_actions)
            idx = idx.unsqueeze(1) # shape is (batch_size, 1, n_actions)
            p_label = torch.gather(p_out, dim=1, index=idx)
            p_label = p_label.squeeze(1)

        # training
        input_states = generate_input_states(cube, scrambled_states)
        input_states = input_states.to(args.device)

        optimizer.zero_grad()
        
        probs_pred, val_pred = model_target(input_states)

        loss_prob = lossfn_prob(probs_pred, p_label)
        loss_val = lossfn_val(val_pred, v_label)

        loss = loss_prob + loss_val
        loss.backward()

        optimizer.step()

        losses_ce.append(loss_prob.data.item())
        losses_mse.append(loss_val.data.item())

        print('Epoch: [{0}]\t'
                'CE Loss {1:.8f}\t'
                'MSE Loss: {2:.8f}  T: {3:.2f}\n'.format(
                    epoch+1, losses_ce[-1],
                    losses_mse[-1], time.time() - tic 
                ))


        soft_update(model, model_target, tau=args.tau)

        # save best models
        saveBestPolicyModel(args, losses_ce[-1], epoch, model, optimizer, losses_ce, losses_mse)
        saveBestValModel(args, losses_mse[-1], epoch, model, optimizer, losses_ce, losses_mse)

        # save this epoch's model and delete previous epoch's model
        state = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'ce_losses': losses_ce, 'mse_losses': losses_mse, 'epoch': epoch}
        torch.save(state, os.path.join(args.save_path, 'checkpoint_' + str(epoch+1) + '.pt'))
        if os.path.exists(os.path.join(args.save_path, 'checkpoint_' + str(epoch) + '.pt')):
            os.remove(os.path.join(args.save_path, 'checkpoint_' + str(epoch) + '.pt'))

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
