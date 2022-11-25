import numpy as np
import torch
import os

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
            print(f"Saving best policy model for epoch: {epoch+1}\n")
            state = {'state_dict': model.state_dict(), 'target_state_dict': model_target.state_dict(),
                     'optimizer': optimizer.state_dict(), 'ce_losses': ce_losses, 'mse_losses': mse_losses, 'epoch': epoch}   
            torch.save(state, os.path.join(args.save_path, 'best_policy_checkpoint_' + str(epoch+1) + '.pt'))
