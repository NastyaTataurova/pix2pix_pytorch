import torch

def save_model_optimazer(model, optimizer, filename):
    model_opt_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(model_opt_dict, filename)


def load_model_optimazer(filename, model, optimizer, lr, device):
    model_opt_dict = torch.load(filename, map_location=device)
    model.load_state_dict(model_opt_dict['model'])
    optimizer.load_state_dict(model_opt_dict['optimizer'])

    for param in optimizer.param_groups:
        param['lr'] = lr