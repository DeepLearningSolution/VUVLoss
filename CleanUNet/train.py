# Adapted from https://github.com/NVIDIA/waveglow under the BSD 3-Clause License.

# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import os
import time
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import random
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)


from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor

from dataset import load_CleanNoisyPairDataset
from stft_loss import MultiResolutionSTFTLoss
from util import rescale, find_max_epoch, print_size
from util import LinearWarmupCosineDecay, loss_fn

from network import CleanUNet


def train(num_gpus, rank, group_name, 
          exp_path, log, optimization, loss_config):

    # setup local experiment path
    if rank == 0:
        print('exp_path:', exp_path)
    
    # Create tensorboard logger.
    log_directory = os.path.join(log["directory"], exp_path)
    if rank == 0:
        tb = SummaryWriter(os.path.join(log_directory, 'tensorboard'))

    # distributed running initialization
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_config)

    # Get shared ckpt_directory ready
    ckpt_directory = os.path.join(log_directory, 'checkpoint')
    if rank == 0:
        if not os.path.isdir(ckpt_directory):
            os.makedirs(ckpt_directory)
            os.chmod(ckpt_directory, 0o775)
        print("ckpt_directory: ", ckpt_directory, flush=True)

    # load training data
    load_vuv = (loss_config.get('use_vuv_loss', False) or 
                loss_config.get('use_adaptive_vuv', False) or
                loss_config.get('time_vuv_loss', False) or
                loss_config.get('freq_vuv_loss', False))
    trainloader = load_CleanNoisyPairDataset(**trainset_config, 
                            subset='training',
                            batch_size=optimization["batch_size_per_gpu"], 
                            num_gpus=num_gpus,
                            load_vuv=load_vuv)
    print('Data loaded', f'(VUV: {load_vuv})')
    
    # predefine model
    net = CleanUNet(**network_config).cuda()
    print_size(net)
    
    # Initialize adaptive VUV importance module if needed (NEW)
    importance_module = None
    if loss_config.get('use_adaptive_vuv', False):
        from adaptive_vuv_loss import AdaptiveVUVWeightingModule
        
        adaptive_config = loss_config.get('adaptive_vuv_config', {})
        importance_module = AdaptiveVUVWeightingModule(
            n_fft=adaptive_config.get('n_fft', 512),
            n_freq_bands=adaptive_config.get('n_freq_bands', 8)
        ).cuda()
        
        total_params = sum(p.numel() for p in importance_module.parameters())
        print(f'Importance module initialized with {total_params:,} parameters')

    # apply gradient all reduce
    if num_gpus > 1:
        net = apply_gradient_allreduce(net)
        if importance_module is not None:
            importance_module = apply_gradient_allreduce(importance_module)

    # define optimizer (include importance_module parameters if present)
    if importance_module is not None:
        optimizer = torch.optim.Adam(
            list(net.parameters()) + list(importance_module.parameters()),
            lr=optimization["learning_rate"]
        )
        print('Optimizer includes importance module parameters')
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=optimization["learning_rate"])

    # load checkpoint
    time0 = time.time()
    if log["ckpt_iter"] == 'max':
        ckpt_iter = find_max_epoch(ckpt_directory)
    else:
        ckpt_iter = log["ckpt_iter"]
    if ckpt_iter >= 0:
        try:
            # load checkpoint file
            model_path = os.path.join(ckpt_directory, '{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # feed model dict and optimizer state
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load importance module state dict if present (NEW)
            if importance_module is not None and 'importance_module_state_dict' in checkpoint:
                importance_module.load_state_dict(checkpoint['importance_module_state_dict'])
                print('Importance module state loaded from checkpoint')

            # record training time based on elapsed time
            time0 -= checkpoint['training_time_seconds']
            print('Model at iteration %s has been trained for %s seconds' % (ckpt_iter, checkpoint['training_time_seconds']))
            print('checkpoint model loaded successfully')
        except:
            ckpt_iter = -1
            print('No valid checkpoint model found, start training from initialization.')
    else:
        ckpt_iter = -1
        print('No valid checkpoint model found, start training from initialization.')

    # training
    n_iter = ckpt_iter + 1

    # define learning rate scheduler and stft-loss
    scheduler = LinearWarmupCosineDecay(
                    optimizer,
                    lr_max=optimization["learning_rate"],
                    n_iter=optimization["n_iters"],
                    iteration=n_iter,
                    divider=25,
                    warmup_proportion=0.05,
                    phase=('linear', 'cosine'),
                )

    if loss_config["stft_lambda"] > 0:
        mrstftloss = MultiResolutionSTFTLoss(**loss_config["stft_config"]).cuda()
    else:
        mrstftloss = None

    while n_iter < optimization["n_iters"] + 1:
        # for each epoch
        for batch_data in trainloader:
            if len(batch_data) == 3:
                clean_audio, noisy_audio, fileid = batch_data
                unvoiced_audio, voiced_audio = None, None
            elif len(batch_data) == 5:
                clean_audio, noisy_audio, unvoiced_audio, voiced_audio, fileid = batch_data
            else:
                raise ValueError(f"Unexpected batch data length: {len(batch_data)}")
            
            clean_audio = clean_audio.cuda()
            noisy_audio = noisy_audio.cuda()
            if unvoiced_audio is not None:
                unvoiced_audio = unvoiced_audio.cuda()
                voiced_audio = voiced_audio.cuda()

            # If you have a data augmentation function augment()
            # noise = noisy_audio - clean_audio
            # noise, clean_audio = augment((noise, clean_audio))
            # noisy_audio = noise + clean_audio

            # back-propagation
            optimizer.zero_grad()
            X = (clean_audio, noisy_audio, unvoiced_audio, voiced_audio)
            loss, loss_dic = loss_fn(net, X, **loss_config, mrstftloss=mrstftloss,
                                     importance_module=importance_module)  # NEW
            if num_gpus > 1:
                reduced_loss = reduce_tensor(loss.data, num_gpus).item()
            else:
                reduced_loss = loss.item()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(net.parameters(), 1e9)
            
            # Also clip importance module gradients if present (NEW)
            if importance_module is not None:
                importance_grad_norm = nn.utils.clip_grad_norm_(importance_module.parameters(), 0.5)
            
            scheduler.step()
            optimizer.step()

            # output to log
            if n_iter % log["iters_per_valid"] == 0:
                print("iteration: {} \treduced loss: {:.7f} \tloss: {:.7f}".format(
                    n_iter, reduced_loss, loss.item()), flush=True)
                
                if rank == 0:
                    # save to tensorboard
                    tb.add_scalar("Train/Train-Loss", loss.item(), n_iter)
                    tb.add_scalar("Train/Train-Reduced-Loss", reduced_loss, n_iter)
                    
                    # Log individual loss components (NEW)
                    for key, value in loss_dic.items():
                        if key != 'attention_weights' and isinstance(value, (int, float, torch.Tensor)):
                            if isinstance(value, torch.Tensor):
                                value = value.item()
                            tb.add_scalar(f"Train/{key}", value, n_iter)
                    
                    # Visualize attention weights if using adaptive VUV (NEW)
                    if 'attention_weights' in loss_dic and n_iter % 5000 == 0:
                        try:
                            from adaptive_vuv_loss import visualize_attention_weights
                            visualize_attention_weights(
                                loss_dic['attention_weights'], 
                                n_iter, 
                                save_dir=os.path.join(log_directory, 'vis')
                            )
                        except Exception as e:
                            print(f"Failed to visualize attention weights: {e}")
                    tb.add_scalar("Train/Gradient-Norm", grad_norm, n_iter)
                    tb.add_scalar("Train/learning-rate", optimizer.param_groups[0]["lr"], n_iter)

            # save checkpoint
            if n_iter > 0 and n_iter % log["iters_per_ckpt"] == 0 and rank == 0:
                checkpoint_name = '{}.pkl'.format(n_iter)
                checkpoint = {
                    'iter': n_iter,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                    'training_time_seconds': int(time.time()-time0)
                }
                
                # Save importance module state if present (NEW)
                if importance_module is not None:
                    checkpoint['importance_module_state_dict'] = importance_module.state_dict()
                
                torch.save(checkpoint, os.path.join(ckpt_directory, checkpoint_name))
                print('model at iteration %s is saved' % n_iter)

            n_iter += 1

    # After training, close TensorBoard.
    if rank == 0:
        tb.close()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.json', 
                        help='JSON file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='',
                        help='name of group for distributed')
    args = parser.parse_args()

    # Parse configs. Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    train_config            = config["train_config"]        # training parameters
    global dist_config
    dist_config             = config["dist_config"]         # to initialize distributed training
    global network_config
    network_config          = config["network_config"]      # to define network
    global trainset_config
    trainset_config         = config["trainset_config"]     # to load trainset

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        if args.group_name == '':
            print("WARNING: Multiple GPUs detected but no distributed group set")
            print("Only running 1 GPU. Use distributed.py for multiple GPUs")
            num_gpus = 1

    if num_gpus == 1 and args.rank != 0:
        raise Exception("Doing single GPU training on rank > 0")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    train(num_gpus, args.rank, args.group_name, **train_config)
