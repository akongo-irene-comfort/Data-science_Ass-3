import torch

def save_checkpoint(path, model, optimizer, episode, global_step, reward, epsilon):
    checkpoint = {
        'episode': episode,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'reward': reward,
        'epsilon': epsilon
    }
    torch.save(checkpoint, path)