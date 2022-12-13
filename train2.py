import numpy as np
from torch import nn, optim
import torch
from model import NNet, ResnetModel
import argparse
import os
import time
from rubikscube import Cube

from torch.utils.tensorboard import SummaryWriter

from utils import SaveBestModel, validate

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', default=10000, type=int)
parser.add_argument('--lr', default=5e-4, type=float)
parser.add_argument('--nepochs',
                    default=100000,
                    type=int,
                    help='Number of ADI epochs (M)')
parser.add_argument('--number_scrambles',
                    default=100,
                    type=int,
                    help='Number of scrambles of cube (k)')
parser.add_argument('--number_sequences',
                    default=100,
                    type=int,
                    help='Number of sequences of scrambles (l)')
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--weight_decay', default=0, type=int, help="Weight decay")
parser.add_argument('--momentum', default=0, type=int, help="Momentum")
parser.add_argument('--tau',
                    default=1.0,
                    type=float,
                    help="Interpolation parameter in soft update")
parser.add_argument('--device', default="cuda", type=str)

parser.add_argument('--resume_path',
                    default=None,
                    type=str,
                    help="Path of model dict to resume from")
parser.add_argument('--save_path',
                    default="results/",
                    type=str,
                    help="Folder in which results are stored")
parser.add_argument('--validation_frequence',
                    default=50,
                    type=int,
                    help="Frequency of validation step (per n epochs)")
parser.add_argument('--update_frequence',
                    default=1000,
                    type=int,
                    help="Frequency of soft update")
parser.add_argument('--validation_scrambles',
                    default=30,
                    type=int,
                    help='Number of scrambles of cube during validation(k)')
parser.add_argument('--run_description',
                    default='',
                    type=str,
                    help='Summary writer comment describing current run')
parser.add_argument('--sampling_method',
                    default='paper',
                    choices=['custom', 'paper'],
                    type=str,
                    help='The sampling method used in training')


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)


def soft_update(local_model, target_model, tau=1.0):
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
        target_param.data.copy_(tau * local_param.data +
                                (1 - tau) * target_param.data)


def generate_scrambled_states(args):
    '''
    Generate and return N = K*L scrambled states as tuples
    Generate weights = 1 / D(xi) where D(xi) is the number of scrambles
    '''

    cube = Cube.cube_qtm()

    scrambled_states = []
    weights = []
    if args.sampling_method == 'custom':
        # generate list of scrambled_states
        for _ in range(args.number_sequences):
            # define state, current_state is stored in env instance
            for d in range(args.number_scrambles):
                cube.scramble(d + 1)
                cur_state = cube.get_state()  # parent state for this iteration
                scrambled_states.append(cur_state)
                weights.append(1 / min((d + 1), 26))

                # set cube back to solved state
                cube = Cube.cube_qtm()

            # add a state at 100 scrambles to each sequence
            cube.scramble(100)
            cur_state = cube.get_state()  # parent state for this iteration
            scrambled_states.append(cur_state)
            weights.append(1 / 26)

            # set cube back to solved state
            cube = Cube.cube_qtm()

    else:
        # generate list of scrambled_states
        for _ in range(args.number_sequences):
            # define state, current_state is stored in env instance
            for d in range(1, args.number_scrambles + 1):
                cube.scramble(1)
                cur_state = cube.get_state()  # parent state for this iteration
                scrambled_states.append(cur_state)
                weights.append(1 / min(d, 26))

            # set cube back to solved state
            cube = Cube.cube_qtm()

    return scrambled_states, weights


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


def adi(args,
        model,
        model_target,
        cube,
        lossfn_prob,
        lossfn_val,
        optimizer,
        n_actions=12):
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

    # Instantiate writer object
    writer = SummaryWriter(comment=args.run_description)

    model = model.to(args.device)
    model_target = model_target.to(args.device)

    # initialize weights using Glorot/Xavier initialization
    if args.resume_path:
        resume_state = torch.load(args.resume_path, map_location=args.device)
        model.load_state_dict(resume_state['state_dict'])
        model_target.load_state_dict(resume_state['state_dict'])
        optimizer.load_state_dict(resume_state['optimizer'])
        losses_ce = resume_state['ce_losses']
        losses_mse = resume_state['mse_losses']
        startEpoch = resume_state['epoch'] + 1
        val_scores = resume_state['val_scores']
        # instantiate saveBestModels with best loss being min of previous losses
        # best_loss = np.min(np.array(losses_ce) + np.array(losses_mse))
        # saveBestModel = SaveBestModel(best_loss)
        amortized_score = 2 * (val_scores[-1] * np.arange(
            1, args.validation_scrambles + 1)).sum() / (
                args.validation_scrambles) / (args.validation_scrambles + 1)
        saveBestModel = SaveBestModel(-amortized_score)
    else:
        init_weights(model)
        soft_update(model, model_target, 1.0)
        losses_ce = []
        losses_mse = []
        val_scores = []
        startEpoch = 0
        # instantiate saveBestModels
        saveBestModel = SaveBestModel()

    model_target.eval()

    for epoch in range(startEpoch, args.nepochs):
        tic = time.time()

        batch_size = args.number_scrambles * args.number_sequences

        # generate N = K*L scrambled states
        scrambled_states, weights = generate_scrambled_states(args)
        weights = torch.tensor(weights, device=args.device)
        # weights = weights/torch.sum(weights)

        next_states = torch.empty((batch_size, n_actions, 480),
                                  device=args.device)
        rewards_all_actions = torch.full((batch_size, n_actions, 1),
                                         fill_value=-1.0,
                                         device=args.device)

        for batch in range(batch_size):
            # set state
            cur_state = scrambled_states[batch]
            cube.set_state(cur_state)

            # for each action in n_actions, we will generate the next_state
            for action in range(n_actions):
                # perform action
                cube.turn(action)

                # define next state, reward
                next_states[batch, action, :] = torch.from_numpy(
                    cube.representation()).float()

                if cube.solved():
                    rewards_all_actions[batch, action] = 1.0

                # set state back to parent state
                cube.set_state(cur_state)

        # forward pass
        with torch.no_grad():
            v_out = model_target.values(
                next_states
            )  # next_states shape is (batchsize, n_actions, 480)

            print("Val mean: ", v_out.mean(dim=0))

            # set labels to be maximal value from each children state
            v_label, idx = torch.max(rewards_all_actions + v_out *
                                     (rewards_all_actions < 0),
                                     dim=1)
            p_label = idx.squeeze(dim=1)

        # training
        model.train()
        input_states = generate_input_states(cube, scrambled_states)
        input_states = input_states.to(args.device)

        optimizer.zero_grad()

        logits_pred, val_pred = model(input_states)

        # calculate sample weighted losses
        loss_logits = lossfn_prob(logits_pred, p_label) * weights
        loss_val = lossfn_val(val_pred, v_label) * weights

        loss = loss_logits.mean() + loss_val.mean()
        loss.backward()

        optimizer.step()

        losses_ce.append(loss_logits.mean().item())
        losses_mse.append(loss_val.mean().item())

        print('Epoch: [{0}]\t'
              'CE Loss {1:.8f}\t'
              'MSE Loss: {2:.8f}  T: {3:.2f}\n'.format(epoch + 1,
                                                       losses_ce[-1],
                                                       losses_mse[-1],
                                                       time.time() - tic))

        if (epoch + 1) % args.update_frequence == 0:
            soft_update(model, model_target, tau=args.tau)
            model_target.eval()

        # validation
        if (epoch + 1) % args.validation_frequence == 0:
            score = validate(args, model)
            print("Validation scores")
            print("\n".join([
                f"scramble depth {i+1} ::: {x:.3%}"
                for (i, x) in enumerate(score)
            ]))
            val_scores.append(score)

            # save best models
            # saveBestModel(args, losses_ce[-1] + losses_mse[-1], epoch, model,
            #               model_target, optimizer, losses_ce, losses_mse)
            amortized_score = 2 * (val_scores[-1] * np.arange(
                1, args.validation_scrambles + 1)).mean() / (
                    args.validation_scrambles) / (args.validation_scrambles +
                                                  1)
            saveBestModel(args, -amortized_score, epoch, model, model_target,
                          optimizer, losses_ce, losses_mse, val_scores)

            for i, split in enumerate(np.split(score, 6)):
                writer.add_scalars(
                    f'Validation/Scores_{5*i+1}_{5*i+5}', {
                        f"scramble_depth {5*i + j + 1}": x
                        for (j, x) in enumerate(split)
                    }, (epoch // args.validation_frequence) + 1)

            writer.add_scalar('Validation/AmortizedScore', amortized_score,
                              (epoch // args.validation_frequence) + 1)

        # save this epoch's model and delete previous epoch's model
        state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'ce_losses': losses_ce,
            'mse_losses': losses_mse,
            'val_scores': val_scores,
            'epoch': epoch
        }
        torch.save(
            state,
            os.path.join(args.save_path,
                         'checkpoint_' + str(epoch + 1) + '.pt'))
        if os.path.exists(
                os.path.join(args.save_path,
                             'checkpoint_' + str(epoch) + '.pt')):
            if (epoch + 1) % 1000 != 0:
                os.remove(
                    os.path.join(args.save_path,
                                 'checkpoint_' + str(epoch) + '.pt'))

        # send stuff to Tensorboard
        writer.add_scalar('TrainLoss/CrossEntropy', losses_ce[-1], epoch + 1)
        writer.add_scalar('TrainLoss/MeanSquareError', losses_mse[-1],
                          epoch + 1)
        writer.add_scalar('TotalLoss', losses_ce[-1] + losses_mse[-1],
                          epoch + 1)

        # print gradient norms
        total_norm = 0.0
        for p in model.parameters():
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item()**2

        total_norm = total_norm**0.5
        print("Gradient norm = ", total_norm)

        writer.add_scalar('TotalLoss/GradientNorm', total_norm, epoch + 1)

    return model


if __name__ == "__main__":

    args = parser.parse_args()
    device = torch.device(args.device)

    # Instantiate model, optimizer, loss functions
    model = ResnetModel(batch_norm=False)
    model_target = ResnetModel(batch_norm=False)

    optimizer = optim.Adam(model.parameters(),
                              lr=args.lr)
    lossfn_val = nn.MSELoss(reduction='none')
    lossfn_prob = nn.CrossEntropyLoss(reduction='none')

    # Instantiate env
    cube = Cube.cube_qtm()

    print('ADI started')
    model = adi(args, model, model_target, cube, lossfn_prob, lossfn_val,
                optimizer)
    print('ADI done')
