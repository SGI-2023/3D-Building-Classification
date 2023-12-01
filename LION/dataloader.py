import datasets.pointflow_datasets as pf
from default_config import cfg as config

class FakeArgs:
    def __init__(self):
        self.distributed = False
        self.local_rank = 0
        self.eval_trainnll = False

def dataloaders_from_config(config):
    args = FakeArgs()
    loaders = pf.get_data_loaders(config, args)
    train_loader = loaders['train_loader']
    test_loader = loaders['test_loader']
    return train_loader, test_loader

def encode_vae()
# convert to latent representation
# compute_loss_vae function in train_2prior.py
# check out vae_adain.py -- decompose_eps function
# add noise according to timestep t
# should be in training code


if __name__=="__main__":
    model_config = './config/car_prior_cfg.yml'
    config.merge_from_file(model_config)
    train_loader, test_loader = dataloaders_from_config(config.data)
    breakpoint()
    print("train_loader", train_loader)
    
    # get the first batch
    batch = next(iter(train_loader))
    # get the input point cloud - shape: (B, Npoints, 3)
    # B = 20, number of points = 2048
    tr_pts = batch['tr_points']
    