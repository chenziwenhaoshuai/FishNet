import torch.nn
import torch
from torch import nn
from dataset import Mydataset
# from model_resnet import mymodel
from model_darknet import mymodel
import torchvision
import os
import time
from loss import loss_func
from torchsummary import summary
from thop import profile
from tqdm import tqdm


if __name__ == '__main__':
    epoches = 300
    lr = 0.001
    batch_size = 11
    weight_path = "./weights/detect_point_pre.pt"
    pretrained = 1
    accumulation_steps = 108 # actual batch size = batch_size * accumulation_steps, 1 is the original batchsize
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Train on CUDA')
    else:
        device = torch.device('cpu')

    dataset = Mydataset()

    train_loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size, shuffle=True)

    net = mymodel().to(device)
    summary(net,[(3,416,416)])

    if pretrained:
        if os.path.exists(weight_path):
            print('Use Pretrained Model')
            net.load_state_dict(torch.load(weight_path))
        else:
            raise RuntimeError('NO pretrained model')
    net.train()

    opt = torch.optim.Adam(net.parameters(),lr=lr)
    # opt = torch.optim.SGD(net.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99)
    # scheduler = torch.optim.lr_scheduler.StepLR(opt, 200,gamma = 0.9,last_epoch=epoches)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=10, T_mult=2)

    for epoch in range(epoches):
        start = time.time()
        with tqdm(total=len(train_loader)) as pbar:
            for i,(image, target) in enumerate(train_loader):
                target_13, target_26, target_52, image = target[13].to(device).float(), target[26].to(device).float(), target[52].to(device).float(), image.to(device).float()
                output_13, output_26, output_52 = net(image)
                loss13 = loss_func(output_13, target_13, 0.6, 1)
                loss26 = loss_func(output_26, target_26, 0.6, 1)
                loss52 = loss_func(output_52, target_52, 0.6, 1)

                loss = loss13 + loss26 + loss52
                loss = loss/accumulation_steps

                loss.backward()
                if ((i + 1) % accumulation_steps) == 0:
                    # optimizer the net
                    opt.step()  # update parameters of net
                    opt.zero_grad()  # reset gradient
                pbar.set_description("Epoch %i" % int(epoch+1))
                pbar.set_postfix(loss = loss.item(),lr=scheduler.get_last_lr()[0])
                pbar.update()
            scheduler.step()





            # print(image.type)
            # print(target[13].shape)
        end = time.time()
        run_time = end-start
        # print(f'epoch：{epoch},  run time:{run_time},  loss:{loss}, lr:{scheduler.get_last_lr()[0]}\n')
        if epoch%5 == 0:
            torch.save(net.state_dict(), f'weights/checkpoint.pt')