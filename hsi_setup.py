import torch
import torch.optim as optim
import models
from math import inf
from tqdm import tqdm
import  scipy.io as scio

import os
import argparse

from os.path import join
from utility import *
from utility.ssim import SSIMLoss
from utility import dataloaders_hsi_test
from utility.read_HSI import read_HSI
from utility.refold import refold 
import torch.nn.functional as F

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


class MultipleLoss(nn.Module):
    def __init__(self, losses, weight=None):
        super(MultipleLoss, self).__init__()
        self.losses = nn.ModuleList(losses)
        self.weight = weight or [1/len(self.losses)] * len(self.losses)
    
    def forward(self, predict, target):
        total_loss = 0
        for weight, loss in zip(self.weight, self.losses):
            total_loss += loss(predict, target) * weight
        return total_loss

    def extra_repr(self):
        return 'weight={}'.format(self.weight)


def train_options(parser):
    def _parse_str_args(args):
        str_args = args.split(',')
        parsed_args = []
        for str_arg in str_args:
            arg = int(str_arg)
            if arg >= 0:
                parsed_args.append(arg)
        return parsed_args    
    parser.add_argument('--prefix', '-p', type=str, default='sru3d_nobn_test',
                        help='prefix')
    parser.add_argument('--arch', '-a', metavar='ARCH',
                        default='mscnet',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names))
    parser.add_argument('--batchSize', '-b', type=int,
                        default=1, help='training batch size. default=16')         
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate. default=1e-3.')
    parser.add_argument('--wd', type=float, default=0,
                        help='weight decay. default=0')
    parser.add_argument('--loss', type=str, default='l2',
                        help='which loss to choose.', choices=['l1', 'l2', 'smooth_l1', 'ssim', 'l2_ssim'])
    parser.add_argument('--init', type=str, default='kn',
                        help='which init scheme to choose.', choices=['kn', 'ku', 'xn', 'xu', 'edsr'])
    parser.add_argument('--no-cuda', action='store_true', help='disable cuda?')
    parser.add_argument('--no-log', action='store_true',
                        help='disable logger?')
    parser.add_argument('--threads', type=int, default=8,
                        help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=2021,
                        help='random seed to use. default=2021')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--pretrain', '-pre', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--no-ropt', '-nro', action='store_true',
                            help='not resume optimizer')          
    parser.add_argument('--chop', action='store_true',
                            help='forward chop')                                      
    parser.add_argument('--resumePath', '-rp', type=str,
                        default=None, help='checkpoint to use.')
    parser.add_argument('--dataroot', '-d', type=str,
                        default='datasets/ICVL64_31.db', help='data root')
    parser.add_argument('--clip', type=float, default=1e6)
    parser.add_argument('--gpu-ids', type=str, default='1,3', help='gpu ids')
    parser.add_argument('--noise_level', type=str, default=15)
    parser.add_argument('--outdir', type=str, default='./')
    parser.add_argument('--stride_test', type=int, default=12)
    parser.add_argument('--phantom', type=bool, default=True)
    

    ###MSCNet
    parser.add_argument('--kernel_size_unique', type=int, default=3)
    parser.add_argument('--num_filters_unique', type=int, default=3)
    parser.add_argument('--kernel_size_share', type=int, default=3)
    parser.add_argument('--num_filters_share', type=int, default=3)
    parser.add_argument('--unfoldings', type=int, default=9)
    parser.add_argument('--num_filters', type=int, default=8)
    parser.add_argument('--stride_share', type=int, default=1)
    parser.add_argument('--stride_unique', type=int, default=1)
    parser.add_argument('--data_stride', type=int, default=1)
    parser.add_argument('--noise_est', type=bool, default=True)
    parser.add_argument('--multi_theta', type=int, default=1)
    parser.add_argument('--testroot', '-tr', type=str,default= '/home/xxx/HDD/xxx/ICVL/test')
    parser.add_argument('--gtroot', '-gr', type=str,default= '/home/xxx/HDD/xxx/ICVL/test/test_crop/')
    parser.add_argument('--jac_loss_freq', type=float, default=0.0,
                    help='the frequency of applying the jacobian regularization (default to 0)')
    parser.add_argument('--jac_incremental', type=int, default=0,
                    help='if positive, increase jac_loss_weight by 0.1 after this many steps')
    parser.add_argument('--pretrain_steps', type=int, default=0,
                    help='number of pretrain steps (default to 0')
    parser.add_argument('--jac_loss_weight', type=float, default=0.0,
                    help='jacobian regularization loss weight (default to 0)')
    opt = parser.parse_args()
    opt.gpu_ids = _parse_str_args(opt.gpu_ids)
    return opt


def make_dataset(opt, train_transform, target_transform, common_transform, batch_size=None, repeat=1):
    dataset = LMDBDataset(opt.dataroot, repeat=repeat)
    # dataset.length -= 1000
    # dataset.length = size or dataset.length

    """Split patches dataset into training, validation parts"""
    dataset = TransformDataset(dataset, common_transform)

    train_dataset = ImageTransformDataset(dataset, train_transform, target_transform)

    #train_loader = DataLoader(train_dataset,
    #                          batch_size=batch_size or opt.batchSize, shuffle=True,
    #                          num_workers=opt.threads, pin_memory=not opt.no_cuda, worker_init_fn=worker_init_fn)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size or opt.batchSize, shuffle=True,
                              num_workers=opt.threads, pin_memory=not opt.no_cuda, worker_init_fn=worker_init_fn)
    return train_loader


class Engine(object):
    def __init__(self, opt):
        self.prefix = opt.prefix
        self.opt = opt
        self.net = None
        self.optimizer = None
        self.criterion = None
        self.basedir = None
        self.iteration = None
        self.epoch = None
        self.best_psnr = None
        self.best_loss = None
        self.writer = None

        self.__setup()

    def __setup(self):
        self.basedir = join(self.opt.outdir,'checkpoints', self.opt.arch)
        if not os.path.exists(self.basedir):
            os.makedirs(self.basedir)

        self.best_psnr = 0
        self.best_loss = 1e6
        self.epoch = 0  # start from epoch 0 or last checkpoint epoch
        self.iteration = 0
        '''
        cuda_list = str(self.opt.gpu_ids)[1:-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_list
        gpus_list = []
        for gpus in range(len(self.opt.gpu_ids)):
            gpus_list.append(gpus)
        self.opt.gpu_ids = gpus_list
        '''
        cuda = not self.opt.no_cuda
        self.device = 'cuda:'+str(self.opt.gpu_ids[0]) if cuda else 'cpu'
        #if self.device!='cpu':
        torch.cuda.set_device('cuda:{}'.format(self.opt.gpu_ids[0]))
        print('Cuda Acess: %d' % cuda)
        if cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")

        torch.manual_seed(self.opt.seed)
        if cuda:
            torch.cuda.manual_seed(self.opt.seed)

        """Model"""
        print("=> creating model '{}'".format(self.opt.arch))

        from models.CSCNet_T_grad_DEQ import ListaParams
        params = ListaParams(kernel_size_unique=self.opt.kernel_size_unique, num_filters_unique=self.opt.num_filters_unique, stride_share=self.opt.stride_share,stride_unique=self.opt.stride_unique, unfoldings=self.opt.unfoldings,multi_lmbda=self.opt.multi_theta,
               kernel_size_share=self.opt.kernel_size_share, num_filters_share=self.opt.num_filters_share,noise_est=self.opt.noise_est,threshold=0.01, data_stride=self.opt.data_stride, phantom=self.opt.phantom
                             )


        # params = ListaParams(in_channels=1, channels=self.opt.channels,
        #                  num_half_layer=self.opt.num_half_layer,unfolding=self.opt.unfolding,bn=self.opt.bn)


        with torch.cuda.device(self.opt.gpu_ids[0]):
            self.net = models.__dict__[self.opt.arch](params)
        # initialize parameters
        
        init_params(self.net, init_type=self.opt.init) # disable for default initialization

        if len(self.opt.gpu_ids) > 1:
            from models.sync_batchnorm import DataParallelWithCallback
            self.net = DataParallelWithCallback(self.net, device_ids=self.opt.gpu_ids)
        
        if self.opt.loss == 'l2':
            self.criterion = nn.MSELoss()
        if self.opt.loss == 'l1':
            self.criterion = nn.L1Loss()
        if self.opt.loss == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss()
        if self.opt.loss == 'ssim':
            self.criterion = SSIMLoss(data_range=1, channel=31)
        if self.opt.loss == 'l2_ssim':
            self.criterion = MultipleLoss([nn.MSELoss(), SSIMLoss(data_range=1, channel=31)], weight=[1, 2.5e-3])
        
        print(self.criterion)

        if cuda:
            #torch.cuda.set_device('cuda:{}'.format(self.device))
            #torch.cuda.is_available()
            #torch.cuda.clear_memory_allocated()
            self.net.to(self.device)
            self.criterion = self.criterion.to(self.device)

        """Logger Setup"""
        log = not self.opt.no_log
        if log:
            self.writer = get_summary_writer(os.path.join(self.basedir, 'logs'), self.opt.prefix)

        """Optimization Setup"""
        self.optimizer = optim.Adam(
            self.net.parameters(), lr=self.opt.lr, weight_decay=self.opt.wd, amsgrad=False)

        """Resume previous model"""
        if self.opt.resume:
            # Load checkpoint.
            self.load(self.opt.resumePath, not self.opt.no_ropt)
        elif self.opt.pretrain:
            self.load_params(self.opt.resumePath, not self.opt.no_ropt)
        else:
            print('==> Building model..')
            print(self.net)

    def forward(self, inputs):        
        if self.opt.chop:            
            output = self.forward_chop(inputs)
        else:
            output = self.net(inputs)
        
        return output

    def forward_chop(self, x, base=16):        
        n, c, b, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        
        shave_h = np.ceil(h_half / base) * base - h_half
        shave_w = np.ceil(w_half / base) * base - w_half

        shave_h = shave_h if shave_h >= 10 else shave_h + base
        shave_w = shave_w if shave_w >= 10 else shave_w + base

        h_size, w_size = int(h_half + shave_h), int(w_half + shave_w)        
        
        inputs = [
            x[..., 0:h_size, 0:w_size],
            x[..., 0:h_size, (w - w_size):w],
            x[..., (h - h_size):h, 0:w_size],
            x[..., (h - h_size):h, (w - w_size):w]
        ]

        outputs = [self.net(input_i) for input_i in inputs]

        output = torch.zeros_like(x)
        output_w = torch.zeros_like(x)
        
        output[..., 0:h_half, 0:w_half] += outputs[0][..., 0:h_half, 0:w_half]
        output_w[..., 0:h_half, 0:w_half] += 1
        output[..., 0:h_half, w_half:w] += outputs[1][..., 0:h_half, (w_size - w + w_half):w_size]
        output_w[..., 0:h_half, w_half:w] += 1
        output[..., h_half:h, 0:w_half] += outputs[2][..., (h_size - h + h_half):h_size, 0:w_half]
        output_w[..., h_half:h, 0:w_half] += 1
        output[..., h_half:h, w_half:w] += outputs[3][..., (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]
        output_w[..., h_half:h, w_half:w] += 1
        
        output /= output_w

        return output

    def __step(self, train, inputs, targets, noise_level):
        if train:
            self.optimizer.zero_grad()
        loss_data = 0
        total_norm = None
        if self.get_net().bandwise:
            O = []
            for _, (i, t,noise_level) in enumerate(zip(inputs.split(1, 1),noise_level.split(1,1), targets.split(1, 1))):
                o = self.net(i,noise_level)
                O.append(o)
                loss = self.criterion(o, t)
                if train:
                    loss.backward()
                loss_data += loss.item()
            outputs = torch.cat(O, dim=1)
        else:
            # outputs, jac_loss, sradius = self.net(inputs)
            compute_jac_loss = np.random.uniform(0,1) < self.opt.jac_loss_freq
            outputs, jac_loss = self.net(inputs, compute_jac_loss)
            # print('psnr:',np.mean(cal_bwpsnr(outputs, targets)))
            # outputs = torch.clamp(self.net(inputs), 0, 1)
            # loss = self.criterion(outputs, targets)
            
            # if outputs.ndimension() == 5:
            #     loss = self.criterion(outputs[:,0,...], torch.clamp(targets[:,0,...], 0, 1))
            # else:
            #     loss = self.criterion(outputs, torch.clamp(targets, 0, 1))
            loss = self.criterion(outputs, targets)
            jac_loss = jac_loss.float().mean().type_as(loss)
            if compute_jac_loss:
                loss = loss + self.opt.jac_loss_weight * jac_loss
            
            if train:
                loss.backward()
            loss_data += loss.item()
        if train:
            total_norm = nn.utils.clip_grad_norm_(self.net.parameters(), self.opt.clip)
            self.optimizer.step()

        return outputs, loss_data, total_norm

    def load(self, resumePath=None, load_opt=True):
        model_best_path = join(self.basedir, self.prefix, 'model_latest.pth')
        if os.path.exists(model_best_path):
            best_model = torch.load(model_best_path,map_location={'cuda:4': 'cuda:0','cuda:1': 'cuda:0'})

        print('==> Resuming from checkpoint %s..' % resumePath)
        assert os.path.isdir('checkpoints'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(resumePath or model_best_path,map_location=torch.device(self.device))
        #### comment when using memnet
        # self.epoch = checkpoint['epoch'] 
        self.epoch = 0 
        self.iteration = checkpoint['iteration']
        if load_opt and self.iteration!=196770:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.optimizer.param_groups[0]['capturable'] = True

        ####
        pytorch_total_params = sum(p.numel() for p in  self.get_net().parameters())
        print('Nb tensors: ', len(list(self.get_net().named_parameters())), "; Trainable Params: ", pytorch_total_params)
        self.get_net().load_state_dict(checkpoint['net'],strict=False)

    def load_params(self, resumePath=None, load_opt=True):
        model_best_path = join(self.basedir, self.prefix, 'model_latest.pth')
        if os.path.exists(model_best_path):
            best_model = torch.load(model_best_path)

        print('==> Resuming from checkpoint %s..' % resumePath)
        assert os.path.isdir('checkpoints'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(resumePath or model_best_path,map_location=torch.device(self.device))
        ####
        pytorch_total_params = sum(p.numel() for p in  self.get_net().parameters())
        print('Nb tensors: ', len(list(self.get_net().named_parameters())), "; Trainable Params: ", pytorch_total_params)
        self.get_net().load_state_dict(checkpoint['net'])

    def train(self, train_loader):
        print('\nEpoch: %d' % self.epoch)
        self.net.train()
        train_loss = 0

        for batch_idx, (inputs, targets, noise_level) in enumerate(train_loader):
            if not self.opt.no_cuda:
                # print(type(inputs))
                # print((targets[0].shape))
                inputs, targets, noise_level = inputs.to(self.device), targets.to(self.device),noise_level.to(self.device)
            outputs, loss_data, total_norm = self.__step(True, inputs, targets,noise_level)
            train_loss += loss_data
            avg_loss = train_loss / (batch_idx+1)

            if not self.opt.no_log:
                self.writer.add_scalar(
                    join(self.prefix, 'train_loss'), loss_data, self.iteration)
                self.writer.add_scalar(
                    join(self.prefix, 'train_avg_loss'), avg_loss, self.iteration)

            self.iteration += 1

            progress_bar(batch_idx, len(train_loader), 'AvgLoss: %.4e | Loss: %.4e | Norm: %.4e' 
                         % (avg_loss, loss_data, total_norm))

        self.epoch += 1
        if self.epoch > max(0, self.opt.pretrain_steps) and self.opt.jac_loss_weight > 0 and self.opt.jac_loss_freq > 0 and \
           self.opt.jac_incremental > 0 and self.epoch % self.opt.jac_incremental == 0:
            # logging(f"Adding 0.1 to jac. regularization weight after {self.epoch} steps")
            self.opt.jac_loss_weight += 0.02
        if not self.opt.no_log:
            self.writer.add_scalar(
                join(self.prefix, 'train_loss_epoch'), avg_loss, self.epoch)

    def validate(self, valid_loader, name):
        self.net.eval()
        validate_loss = 0
        total_psnr = 0
        total_ssim=0
        total_sam=0
        print('\n[i] Eval dataset {}...'.format(name))
        #print(torch.cuda.device_count())
        torch.cuda.empty_cache()
        res_arr = np.zeros((len(valid_loader), 3))
        with torch.no_grad():
            with torch.cuda.device(self.opt.gpu_ids[0]):
                for batch_idx,(inputs,noise_level,fname) in enumerate(tqdm(valid_loader,disable=True)):
                    fname=fname[0]
                    targets=dataloaders_hsi_test.get_gt(self.opt.gtroot,fname)
                    # inputs = inputs.unsqueeze(0)
                    # targets = targets.unsqueeze(0).unsqueeze(0)
                    targets = targets.unsqueeze(0)
                    inputs, targets=inputs[:,:,:,:], targets[:,:,:,:]
                    # noise_level=torch.from_numpy(noise_level/255)
                    # noise_level = torch.tensor(noise_level / 255, dtype=inputs.dtype)
                    if not self.opt.no_cuda:
                        self.device = 'cuda:'+str(self.opt.gpu_ids[0]) if not self.opt.no_cuda else 'cpu'
                        inputs, targets,noise_level = inputs.to(self.device), targets.to(self.device),noise_level.to(self.device)
        ###modified###
                    outputs, loss_data, _ = self.__step(False, inputs, targets, noise_level)
                    scio.savemat('./iter_res/'+fname + '_a5.mat', {'deqcscnet': outputs.cpu().detach().numpy()})
                    psnr = np.mean(cal_bwpsnr(outputs, targets))
                    # ssim = np.mean(cal_bwssim(outputs, targets))
                    # sam = cal_sam(outputs, targets)
                    #print(outputs.shape)
                    res_arr[batch_idx, :] = MSIQA(outputs, targets)
                    validate_loss += loss_data
                    avg_loss = validate_loss / (batch_idx+1)
                    # total_ssim +=ssim
                    # total_sam  +=sam
                    total_psnr += psnr
                    avg_psnr = total_psnr / (batch_idx+1)
                    # avg_ssim =total_ssim/(batch_idx+1)
                    avg_sam= total_sam/(batch_idx+1)
 					# progress_bar(batch_idx, len(valid_loader), 'Loss: %.4e | PSNR: %.4f | SSIM: %.4f |SAM: %.4f'
      #                           % (avg_loss, avg_psnr, avg_ssim, avg_sam))
                    progress_bar(batch_idx, len(valid_loader), 'Loss: %.4e | PSNR: %.4f'
                                % (avg_loss, avg_psnr))
                   # if batch_idx == 10:###modified###
                    #    break###modified###
        if not self.opt.no_log:
            self.writer.add_scalar(
                join(self.prefix, name, 'val_loss_epoch'), avg_loss, self.epoch)
            self.writer.add_scalar(
                join(self.prefix, name, 'val_psnr_epoch'), avg_psnr, self.epoch)

        return avg_psnr, avg_loss,res_arr

    def validate_patch2(self, valid_loader, name, kernel_size, stride, pad):
        self.net.eval()
        validate_loss = 0
        total_psnr = 0
        total_ssim=0
        total_sam=0
        total_time = 0

        # kernel_size = (31,56,56)
        # stride = (31,50,50)
        # pad = (0,8,8)
        # """ ICVL """
        # kernel_size = (31,64,64)
        # stride = (31,64,64)
        # pad = (0,0,0)
        # """ Houston """
        # kernel_size = (31,64,64)
        # stride = (15,64,64)
        # pad = (0,0,0)

        print('\n[i] Eval dataset {}...'.format(name))
        torch.cuda.empty_cache()
        res_arr = np.zeros((len(valid_loader), 3))
        # filenames = []
        with torch.no_grad():
            with torch.cuda.device(self.opt.gpu_ids[0]):
                for batch_idx,(inputs,noise_level,fname) in enumerate(tqdm(valid_loader,disable=True)):
                    tic = time.time()
                    fname=fname[0]
                    # filenames.append(fname)
                    targets=dataloaders_hsi_test.get_gt(self.opt.gtroot,fname)
                    inputs = inputs.squeeze(1)
                    targets = targets.unsqueeze(0)
                    inputs, targets=inputs[:,:,:,:], targets[:,:,:,:]
                    b_size = inputs.shape[0]
                    
                    # print(inputs.shape)

                    self.device = 'cuda'
                    inputs_pad = F.pad(inputs,(pad[2]//2,pad[2]//2,pad[1]//2,pad[1]//2,pad[0]//2,pad[0]//2),mode='reflect')
                    col_data_,data_shape = read_HSI(inputs[0].cpu().numpy(),kernel_size=kernel_size,stride=stride,device=self.device)
                    col_data,data_shape_ = read_HSI(inputs_pad[0].cpu().numpy(),kernel_size=(kernel_size[0]+pad[0],kernel_size[1]+pad[1],kernel_size[2]+pad[2]),stride=stride,device=self.device)
                    col_data = col_data.to('cpu')  
                    if col_data.shape[0] ==0:
                        inputs = col_data[0,:,:,:,:].unsqueeze(0)
                    else:
                        inputs = col_data
                    outputs = torch.empty_like(inputs)

                    start_time=time.time()
                    flops_sum = 0
                    for b in range(0,inputs.shape[0],b_size):
                        print('__',b+1,'/',inputs.shape[0],end='\r')
                        # outputs[b:b+b_size,:,:,:,:] = model.forward(inputs[b:b+b_size,:,:,:,:].squeeze(1).to(self.device)).unsqueeze(1)
                        res, loss_data, _ = self.__step(False, inputs[b:b+b_size,:,:,:,:].squeeze(1).to(self.device), inputs[b:b+b_size,:,:,:,:].squeeze(1).to(self.device),noise_level.to(self.device))
                        outputs[b:b+b_size,:,:,:,:] = res.unsqueeze(1)
                        torch.cuda.empty_cache()
                    # print(flops_sum)
                    outputs = outputs[:,:,
                                      pad[0]//2:kernel_size[0]+pad[0]//2,
                                      pad[1]//2:kernel_size[1]+pad[1]//2,
                                      pad[2]//2:kernel_size[2]+pad[2]//2]
                    endtime=time.time()
                    # print('time:',endtime-start_time)
                    cost_time = endtime-start_time
                    total_time = total_time + cost_time
                    out = refold(outputs.to(inputs.device),data_shape=data_shape, kernel_size=kernel_size,stride=stride,device=self.device).unsqueeze(0).unsqueeze(0).float().to(self.device).squeeze(1)
                    # psnr = np.mean(cal_bwpsnr(outputs, targets))
                    # out = resize_back(out,resize_from).float().to(self.device).squeeze(0)
                    elapsed = time.time() - tic
            
                    # scio.savemat('/opt/data/private/aaai2024_y/DEQ-CSCNet/mat/icvl/'+fname , {'cscnet': out.cpu().detach().numpy()})
                    # print(out.shape)
                    # scio.savemat('/mnt/data_3/yejin/experiment_mat/deqcscnet/wdc/95/'+fname, {'deqcscnet': out.cpu().detach().numpy()})

                    psnr = np.mean(cal_bwpsnr(out, targets))
                    res_arr[batch_idx, :] = MSIQA(out, targets)
                    validate_loss += loss_data
                    avg_loss = validate_loss / (batch_idx+1)
                    total_psnr += psnr
                    avg_psnr = total_psnr / (batch_idx+1)
                    avg_sam= total_sam/(batch_idx+1)
                    avg_time = total_time/(batch_idx + 1)
                    progress_bar(batch_idx, len(valid_loader), 'Loss: %.4e | PSNR: %.4f'
                                % (avg_loss, avg_psnr))
        if not self.opt.no_log:
            self.writer.add_scalar(
                join(self.prefix, name, 'val_loss_epoch'), avg_loss, self.epoch)
            self.writer.add_scalar(
                join(self.prefix, name, 'val_psnr_epoch'), avg_psnr, self.epoch)

        # with open('filenames1.txt', 'w', encoding='utf-8') as file:
        #     for item in filenames:
        #         file.write(item + '\n')  
        print("avg_time",avg_time)
        return avg_psnr, avg_loss,res_arr

    def validate_patch(self, valid_loader, name):
        self.net.eval()
        validate_loss = 0
        total_psnr = 0
        total_ssim=0
        total_sam=0
        print('\n[i] Eval dataset {}...'.format(name))
        #print(torch.cuda.device_count())
        torch.cuda.empty_cache()
        res_arr = np.zeros((len(valid_loader), 3))
        with torch.no_grad():
            with torch.cuda.device(self.opt.gpu_ids[0]):
                for batch_idx,(inputs,noise_level,fname) in enumerate(tqdm(valid_loader,disable=True)):
                    fname=fname[0]
                    targets=dataloaders_hsi_test.get_gt(self.opt.gtroot,fname)
                    # inputs = inputs.unsqueeze(0)
                    # targets = targets.unsqueeze(0).unsqueeze(0)
                    targets = targets.unsqueeze(0)
                    inputs, targets=inputs[:,:,:,:], targets[:,:,:,:]
                    kernel_size = (31,64,64)
                    stride = (31,64,64)
                    col_data,data_shape = read_HSI(inputs[0].numpy(),kernel_size=kernel_size,stride=stride,device='cuda:'+str(self.opt.gpu_ids[0])) 
                    if col_data.shape[0] ==0:
                        inputs = col_data[0,:,:,:,:].unsqueeze(0)
                    else:
                        inputs = col_data
                    targets = targets.unsqueeze(0).unsqueeze(0)
                    # targets= targets/targets.max()
                    self.device = 'cuda:'+str(self.opt.gpu_ids[0]) if not self.opt.no_cuda else 'cpu'
                    if not self.opt.no_cuda:
                        inputs, targets = inputs.to(self.device), targets.to(self.device) 
                    outputs = torch.empty_like(inputs).to(inputs.device)
                    for batch in range(inputs.shape[0]):
                        print(batch,'/',inputs.shape[0],end='\r')
                        outputs[batch:batch+1,:,:,:,:], loss_data, _ = self.__step(False, inputs[batch,:,:,:,:].to(self.device), inputs[batch,:,:,:,:].to(self.device),noise_level.to(self.device))
                        torch.cuda.empty_cache()
                    outputs = refold(outputs,data_shape=data_shape, kernel_size=kernel_size,stride=stride,device='cuda:'+str(self.opt.gpu_ids[0])).unsqueeze(0).unsqueeze(0)
                    # noise_level=torch.from_numpy(noise_level/255)
                    # noise_level = torch.tensor(noise_level / 255, dtype=inputs.dtype)
                    # if not self.opt.no_cuda:
                    #     self.device = 'cuda:'+str(self.opt.gpu_ids[0]) if not self.opt.no_cuda else 'cpu'
                    #     inputs, targets,noise_level = inputs.to(self.device), targets.to(self.device),noise_level.to(self.device)
        ###modified###
                    # outputs, loss_data, _ = self.__step(False, inputs, targets, noise_level)
                    # scio.savemat(fname + 'Res.mat', {'output': outputs.cpu().detach().numpy()})
                    # print("output:",outputs.size())
                    # print("targets:",targets.size())
                    outputs = outputs.squeeze(0).squeeze(0).squeeze(0)
                    targets = targets.squeeze(0).squeeze(0).squeeze(0)
                    psnr = np.mean(cal_bwpsnr(outputs, targets))
                    # ssim = np.mean(cal_bwssim(outputs, targets))
                    # sam = cal_sam(outputs, targets)
                    #print(outputs.shape)
                    res_arr[batch_idx, :] = MSIQA(outputs, targets)
                    validate_loss += loss_data
                    avg_loss = validate_loss / (batch_idx+1)
                    # total_ssim +=ssim
                    # total_sam  +=sam
                    total_psnr += psnr
                    avg_psnr = total_psnr / (batch_idx+1)
                    # avg_ssim =total_ssim/(batch_idx+1)
                    avg_sam= total_sam/(batch_idx+1)
 					# progress_bar(batch_idx, len(valid_loader), 'Loss: %.4e | PSNR: %.4f | SSIM: %.4f |SAM: %.4f'
      #                           % (avg_loss, avg_psnr, avg_ssim, avg_sam))
                    progress_bar(batch_idx, len(valid_loader), 'Loss: %.4e | PSNR: %.4f'
                                % (avg_loss, avg_psnr))
                   # if batch_idx == 10:###modified###
                    #    break###modified###
        if not self.opt.no_log:
            self.writer.add_scalar(
                join(self.prefix, name, 'val_loss_epoch'), avg_loss, self.epoch)
            self.writer.add_scalar(
                join(self.prefix, name, 'val_psnr_epoch'), avg_psnr, self.epoch)

        return avg_psnr, avg_loss,res_arr
    # def validate(self, valid_loader, name, block, batch_size):
    #     self.net.eval()
    #     validate_loss = 0
    #     total_psnr = 0
    #     total_ssim = 0
    #     total_sam = 0
    #     print('\n[i] Eval dataset {}...'.format(name))
    #     # print(torch.cuda.device_count())
    #     # torch.cuda.empty_cache()
    #     res_arr = np.zeros((len(valid_loader), 3))
    #
    #     with torch.no_grad():
    #         with torch.cuda.device(self.opt.gpu_ids[0]):
    #             for batch_idx, (inputs,noise_level, fname) in enumerate(tqdm(valid_loader, disable=True)):
    #                 fname = fname[0]
    #                 print(fname)
    #                 targets = dataloaders_hsi_test.get_gt(self.opt.gtroot, fname)
    #                 # inputs=inputs
    #                 # print(noise_level)
    #                 noise_level=noise_level/255
    #                 noise_level=noise_level.permute([0,3,1,2])
    #                 if not self.opt.no_cuda:
    #                     self.device = 'cuda:' + str(self.opt.gpu_ids[0]) if not self.opt.no_cuda else 'cpu'
    #                     inputs, targets,noise_level = inputs.to(self.device), targets.to(self.device),noise_level.to(self.device)
    #                     # print(inputs.shape)
    #                 batch_noisy_blocks = block._make_blocks(inputs)
    #                 patch_loader = torch.utils.data.DataLoader(batch_noisy_blocks, batch_size=batch_size,
    #                                                            drop_last=False)
    #                 batch_out_blocks = torch.zeros_like(batch_noisy_blocks)
    #                 for i, inp in enumerate(patch_loader):  # if it doesnt fit in memory
    #                     id_from, id_to = i * patch_loader.batch_size, (i + 1) * patch_loader.batch_size
    #                     # print(noise_level.repeat(patch_loader.batch_size,1,1,1))
    #                     bs=inp.shape[0]
    #                     batch_out_blocks[id_from:id_to] = self.net(inp,noise_level.repeat(bs,1,1,1))
    #                 outputs = block._agregate_blocks(batch_out_blocks)
    #                 # print(outputs.shape)
    #                 scio.savemat(fname + 'Res.mat', {'avg': outputs.cpu().detach().numpy()})
    #                 targets = targets.unsqueeze(0).unsqueeze(0)
    #                 # print(targets.shape)
    #                 psnr = np.mean(cal_bwpsnr(outputs, targets))
    #                 # ssim = np.mean(cal_bwssim(outputs, targets))
    #                 # sam = cal_sam(outputs, targets)
    #                 # print(outputs.shape)
    #                 res_arr[batch_idx, :] = MSIQA(outputs, targets)
    #                 avg_loss = validate_loss / (batch_idx + 1)
    #                 # total_ssim +=ssim
    #                 # total_sam  +=sam
    #                 total_psnr += psnr
    #                 avg_psnr = total_psnr / (batch_idx + 1)
    #                 # avg_ssim =total_ssim/(batch_idx+1)
    #                 avg_sam = total_sam / (batch_idx + 1)
    #                 # progress_bar(batch_idx, len(valid_loader), 'Loss: %.4e | PSNR: %.4f | SSIM: %.4f |SAM: %.4f'
    #                 #                           % (avg_loss, avg_psnr, avg_ssim, avg_sam))
    #                 progress_bar(batch_idx, len(valid_loader), '| PSNR: %.4f'
    #                              % (avg_psnr))
    #             # if batch_idx == 10:###modified###
    #             #    break###modified###
    #     if not self.opt.no_log:
    #         self.writer.add_scalar(
    #             join(self.prefix, name, 'val_loss_epoch'), avg_loss, self.epoch)
    #         self.writer.add_scalar(
    #             join(self.prefix, name, 'val_psnr_epoch'), avg_psnr, self.epoch)
    #
    #     return avg_psnr, avg_loss, res_arr

    def save_checkpoint(self, model_out_path=None, **kwargs):
        if not model_out_path:
            model_out_path = join(self.basedir, self.prefix, "model_epoch_%d_%d.pth" % (
                self.epoch, self.iteration))

        state = {
            'net': self.get_net().state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'iteration': self.iteration,
        }
        
        state.update(kwargs)

        if not os.path.isdir(join(self.basedir, self.prefix)):
            os.makedirs(join(self.basedir, self.prefix))

        torch.save(state, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    # saving result into disk
    def test_develop(self, test_loader, savedir=None, verbose=True):
        from scipy.io import savemat
        from os.path import basename, exists

        def torch2numpy(hsi):
            if self.net.use_2dconv:
                R_hsi = hsi.data[0].cpu().numpy().transpose((1,2,0))
            else:
                R_hsi = hsi.data[0].cpu().numpy()[0,...].transpose((1,2,0))
            return R_hsi    

        self.net.eval()
        test_loss = 0
        total_psnr = 0
        dataset = test_loader.dataset.dataset

        res_arr = np.zeros((len(test_loader), 3))
        input_arr = np.zeros((len(test_loader), 3))

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                if not self.opt.no_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs, loss_data, _ = self.__step(False, inputs, targets)
                
                test_loss += loss_data
                avg_loss = test_loss / (batch_idx+1)
                
                res_arr[batch_idx, :] = MSIQA(outputs, targets)
                input_arr[batch_idx, :] = MSIQA(inputs, targets)

                """Visualization"""
                # Visualize3D(inputs.data[0].cpu().numpy())
                # Visualize3D(outputs.data[0].cpu().numpy())

                psnr = res_arr[batch_idx, 0]
                ssim = res_arr[batch_idx, 1]
                if verbose:
                    print(batch_idx, psnr, ssim)

                if savedir:
                    filedir = join(savedir, basename(dataset.filenames[batch_idx]).split('.')[0])  
                    outpath = join(filedir, '{}.mat'.format(self.opt.arch))

                    if not exists(filedir):
                        os.mkdir(filedir)

                    if not exists(outpath):
                        savemat(outpath, {'R_hsi': torch2numpy(outputs)})
                        
        return res_arr, input_arr



    def test_real(self, test_loader, savedir=None):
        self.net.eval()
        #print(torch.cuda.device_count())
        torch.cuda.empty_cache()
        with torch.no_grad():
            with torch.cuda.device(self.opt.gpu_ids[0]):
                for batch_idx,(inputs,fname) in enumerate(tqdm(test_loader,disable=True)):
                    fname=fname[0]
                    print('\n[i] Eval dataset {}...'.format(fname))
                    inputs = inputs.unsqueeze(0)
                    inputs=inputs[:,:,:,:,:]
                    if not self.opt.no_cuda:
                        self.device = 'cuda:'+str(self.opt.gpu_ids[0]) if not self.opt.no_cuda else 'cpu'
                        inputs = inputs.to(self.device)
                    outputs, loss_data, _ = self.__step(False, inputs, inputs)
                    scio.savemat(fname + 'Res.mat', {'output': outputs.cpu().detach().numpy()})

    def get_net(self):
        if len(self.opt.gpu_ids) > 1:
            return self.net.module
        else:
            return self.net           
