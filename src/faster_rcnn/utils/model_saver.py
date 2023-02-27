import torch

class ModelSaver:
    def __init__(self, path_to_save, name):
        self.path_to_save = path_to_save
        self.path_to_save.mkdir(parents=True, exist_ok=True)
        self.best_valid_loss = torch.inf
        self.name = name

    def __call__(self, model, cur_valid_loss):
        if self.best_valid_loss > cur_valid_loss:
            print(f"\nSaving new best model")
            self.best_valid_loss = cur_valid_loss
            bestpath = self.path_to_save.joinpath(self.name + '_best' + '.pth')
            self.save(model, bestpath)
        print(f"\nSaving last model")
        lastpath = self.path_to_save.joinpath(self.name + '_last' + '.pth')
        self.save(model, lastpath)

    def save(self, model, save_path):
        torch.save(model.state_dict(), save_path)
