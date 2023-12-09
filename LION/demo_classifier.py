import datasets.pointflow_datasets as pf
from default_config import cfg as config
import importlib
import torch
from models.lion_classifier import LION_Classifier
import datasets.pointflow_datasets as pf
import copy
import clip


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

def build_model(cfg):
    model_lib = importlib.import_module(cfg.shapelatent.model)
    model = model_lib.Model(cfg)
    return model

def make_4d(x): 
    return x.unsqueeze(-1).unsqueeze(-1) if len(x.shape) == 2 else x.unsqueeze(-1)

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = './lion_ckpt/text2shape/chair/checkpoints/model.pt'
    model_config = './lion_ckpt/text2shape/chair/cfg.yml'

    config.merge_from_file(model_config)
    lion = LION_Classifier(config)
    lion.load_model(model_path)
    
    data_config = copy.deepcopy(config)
    data_config.merge_from_file("./config/car_prior_cfg.yml")
    train_loader, test_loader = dataloaders_from_config(data_config.data)
    
    B = 20
    diffusion_args = config.sde
    
    if config.clipforge.enable:
        input_t = ["a swivel chair, five wheels"] 
        device_str = 'cuda'
        clip_model, clip_preprocess = clip.load(
                            config.clipforge.clip_model, device=device_str)    
        text = clip.tokenize(input_t).to(device_str)
        clip_feat = []
        clip_feat.append(clip_model.encode_text(text).float())
        clip_feat = torch.cat(clip_feat, dim=0)
        print('clip_feat', clip_feat.shape)
    else:
        clip_feat = None
    
    global_prior, local_prior = lion.priors[0], lion.priors[1]
    
    while next(iter(train_loader)) is not None:
        batch = next(iter(train_loader))
        # get the input point cloud - shape: (B, Npoints, 3)
        # B = 20, number of points = 2048
        tr_pts = batch['tr_points'].to(device)
        # encode the points to latent representation
        all_eps, _, _ = lion.vae.encode(tr_pts)
        eps = make_4d(all_eps) #just reshapes it
        # z_global lion.vae.encode(tr_pts)[2][0][0].shape --> torch.Size([20, 128])
        # z_local lion.vae.encode(tr_pts)[2][1][0].shape torch.Size([20, 8192])
        decomposed_eps = lion.vae.decompose_eps(eps)
        # global style-->
        eps = decomposed_eps[0]
        # decomposed_eps[1] --> local style
        # generate timestep encoding

        t = 768 # change this to something random
        t_tensor = torch.ones(B, dtype=torch.int64, device='cuda') * (t+1)
        breakpoint()
        t_p, var_t_p, m_t_p, _, _, _ = lion.diffusion.iw_quantities_t(B, t_tensor, \
                                diffusion_args.time_eps, diffusion_args.iw_sample_p, diffusion_args.iw_subvp_like_vp_sde)
        #error ^ with the timestep shape
        # breakpoint() RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 20 but got size 1 for tensor number 1 in the list.
        
        # add noise to the encoded points
        noise_p = torch.randn(size=eps.size(), device=device)
        eps_t_p = lion.diffusion.sample_q(eps, noise_p, var_t_p, m_t_p)
        # eps_t_p.shape torch.Size([20, 128, 1, 1]          
        # run denoising for that timestep -> get the noise_pred

        condition_input = None        
        # PriorSEClip
        noise_pred = global_prior(x=eps_t_p, t=t_tensor.float(), condition_input=condition_input, clip_feat=clip_feat)
        
        
        # compute some loss between noise_pred and noise_p
        # give you the estimate 
        # N samples
        # for the same timestep t, add noise to each sample
        # run denoising for that timestep -> get the noise_pred
        # for the actual class the loss should be low

        # unconditional/ conditional model
        
        # multi-class model
        # given one sample
        # add noise to it , sample some random noise, for some t
        # for timesteps [1,T]     randomly sample 10 timesteps from [0.4T-0.6T]-- take average loss
        # clip features for every class [ K ]
        # is you use the right clip features --loss would be low
        # wrong clip features -- loss would be high
    

    # t1 = time.time()
    # output = lion.sample(1 if clip_feat is None else clip_feat.shape[0], clip_feat=clip_feat)
    # t2 = time.time()
    # print("Execution time: ", t2-t1, "seconds")