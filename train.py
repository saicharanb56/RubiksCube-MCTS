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
parser.add_argument('--nepochs', default=1000, type=int, help='Number of ADI epochs (M)')
parser.add_argument('--nscrambles', default=100, type=int, help='Number of scrambles of cube (k)')
parser.add_argument('--nsequences', default=100, type=int, help='Number of sequences of scrambles (l)')
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
        ce_loss = resume_state['ce_losses']
        mse_loss = resume_state['mse_losses']
        startEpoch = resume_state['epoch'] + 1
    else:
        init_weights(model)
        ce_loss = []
        mse_loss = []
        startEpoch = 0

    for epoch in range(startEpoch, args.nepochs):
        tic = time.time()

        # initialize list of labels. This will be a list of dictionaries.
        # each dict will have keys "value" and "probs" to be used during training 
        labels = []

        # generate N scrambled states
        scrambled_states = []

        for j in range(args.nsequences):
            # intialize probs, values for all child states of state
            p_all_actions = torch.zeros((n_actions, n_actions))
            v_all_actions = torch.zeros(n_actions)
            
            # initialize rewards array
            rewards_all_actions = torch.zeros(n_actions)

            # define state, current_state is stored in env instance
            for k in range(args.nscrambles):
                cube.scramble(1)
                cur_state = cube.get_state()
                scrambled_states.append(cur_state)

                # for each action in n_actions, we will generate the next_state
                for action in range(n_actions):
                    # perform action
                    cube.turn(action)
                    
                    # define next state, reward
                    next_state = torch.tensor(cube.representation(), dtype=torch.float32, device=args.device)
                    reward = 1 if cube.solved() else -1

                    # forward pass
                    with torch.no_grad():
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
            cube = Cube.cube_qtm()

        # training loop
        for i in range(len(scrambled_states)):

            optimizer.zero_grad()

            state = scrambled_states[i]
            cube.set_state(state)
            input_state = torch.tensor(cube.representation(), dtype=torch.float32, device=args.device)
            value, probs = labels[i]["value"], labels[i]["probs"]

            value, probs = value.to(args.device), probs.to(args.device)

            probs_pred, val_pred = model_target(input_state)
            loss_prob = lossfn_prob(probs_pred, probs)
            
            loss_val = lossfn_val(val_pred[0], value)

            loss_prob.backward(retain_graph=True)
            loss_val.backward()

            optimizer.step()

            ce_loss.append(loss_prob.data.item())
            mse_loss.append(loss_val.data.item())

        print('Epoch: [{0}]\t'
                'CE Loss {1:.4f}\t'
                'MSE Loss: {2:.4f}  T: {3:.2f}\n'.format(
                    epoch+1, np.mean(ce_loss),
                    np.mean(mse_loss), time.time() - tic 
                ))


        soft_update(model, model_target, tau=args.tau)

        # save best models
        saveBestPolicyModel(args, ce_loss[-1], epoch, model, optimizer, ce_loss, mse_loss)
        saveBestValModel(args, mse_loss[-1], epoch, model, optimizer, ce_loss, mse_loss)

        # save this epoch's model and delete previous epoch's model
        state = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'ce_losses': ce_loss, 'mse_losses': mse_loss, 'epoch': epoch}   
        torch.save(state, os.path.join(args.save_path, 'best_policy_checkpoint_' + str(epoch+1) + '.pt'))
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
