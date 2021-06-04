import os
import torch
import torch.nn as nn
from torch.nn.modules.loss import L1Loss
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange
import cv2
import math

import time
from Pix2PixModel import Pix2PixGenerator, Pix2PixDiscriminator
from MyDataset import MyDataset
from config import *


#None data삭제 
#dataset에서 cv.imread시 CRC error 제거용
def my_collate(batch): # batch size 4 [{tensor image, tensor label},{},{},{}] could return something like G= [None, {},{},{}]
    batch= list(filter (lambda x:x is not None, batch)) # this gets rid of nones in batch. For example above it would result to G= [{},{},{}]

    return torch.utils.data.dataloader.default_collate(batch)

#weights 초기화
def init_weights(net, init_type='normal', init_gain=0.02):

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


#psnr metric 계산
def PSNR(input1, input2):

    #이미지당 PSNR계산 함수
    def cal_PSNR(img1, img2):
        MSE = np.mean(((img1-img2)**2))
        PSNR = 10* np.math.log10((255**2)/MSE)
        return PSNR
    l = []

    #배치에 있는 모든 이미지에 대하여
    for i1, i2 in zip(input1, input2):
        psnr = cal_PSNR(i1, i2)
        l.append(psnr)

    return mean(l)

#image를 -1~1로 정규화 시켰던 것을 다시 0~255로 돌림
def img_de_norm(img):
    img = img.detach().numpy()
    img += 1
    img *= 127.5
    img = np.transpose(img, (1,2,0)).astype(np.uint8)

    return img

#loss, psnr 그래프 저장
def save_plot(epoch, train_list, val_list, title):
    x = range(1,epoch+1)
    plt.figure(figsize=(10,5))
    plt.plot(x, train_list, label='train')
    plt.plot(x, val_list, label='val')
    plt.xticks(x)
    plt.title(title)
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(f'./{TRAIN_CONFIG["save_dir"]}/plots/{title}_{epoch}.png')
    plt.axis()
    plt.close()

#결과 이미지 저장 input, fake, ground_truth순으로
def save_image(input_images, fake_images, gt_images, epoch):
    plt.figure(figsize=(30, 50))
    for i in range(5):
        plt.subplot(5,3, i*3+1)
        plt.imshow(input_images[i])
        plt.subplot(5,3, i*3+2)
        plt.imshow(fake_images[i])
        plt.subplot(5,3, i*3+3)
        plt.imshow(gt_images[i])
    
    plt.savefig(f'./{TRAIN_CONFIG["save_dir"]}/images/{epoch}.png')
    plt.close()

#폴더 만들기
def make_dir(PATH):
    if not os.path.exists(PATH):
        os.mkdir(PATH)

#평균
def mean(l):
    return sum(l) / len(l)


#학습 시작
def train(device):
    #training용 데이터셋 정의
    train_dataset = MyDataset(data_path=DATA_CONFIG['data_path'],
                                list_file=DATA_CONFIG['train_list'],
                                input_dir=DATA_CONFIG['input_image_dir'],
                                gt_dir=DATA_CONFIG['ground_truth_image_dir'], 
                                shuffle=TRAIN_CONFIG['shuffle'],
                                transform='CROP')
    
    #training을 위한 dataloader
    train_dataloader = DataLoader(train_dataset, 
                                    batch_size=TRAIN_CONFIG['batch_size'],
                                    shuffle=TRAIN_CONFIG['shuffle'],
                                    num_workers=TRAIN_CONFIG['num_workers'],
                                    collate_fn=my_collate
                                    )
    start = time.time()
    for i in range(32):
        a = train_dataset[i]
    print(time.time()-start)

    #training중 데이터 저장을 위한 각종 폴더 생성
    make_dir(f'./{TRAIN_CONFIG["save_dir"]}')
    make_dir(f'./{TRAIN_CONFIG["save_dir"]}/models')
    make_dir(f'./{TRAIN_CONFIG["save_dir"]}/plots')
    make_dir(f'./{TRAIN_CONFIG["save_dir"]}/images')

    #validation용 데이터셋 정의 
    #validation 데이터셋은 batch_size 32로 고정
    #shuffle은 하지 않음
    val_dataset = MyDataset(data_path=DATA_CONFIG['data_path'],
                            list_file=DATA_CONFIG['val_list'],
                            input_dir=DATA_CONFIG['input_image_dir'],
                            gt_dir=DATA_CONFIG['ground_truth_image_dir'],
                            shuffle=False)

    val_dataloader = DataLoader(val_dataset, 
                                batch_size=32,
                                shuffle=False,
                                num_workers=TRAIN_CONFIG['num_workers'],
                                collate_fn=my_collate
                                )

    #pix2pix Generator 
    pix2pix_G = Pix2PixGenerator(in_channels=TRAIN_CONFIG['g_input_channels'],
                                out_channels=TRAIN_CONFIG['g_output_channels'],
                                nfeature=TRAIN_CONFIG['nfeatures'])
    init_weights(pix2pix_G)
    pix2pix_G.to(device)
    pix2pix_G.train()

    #pix2pix Discriminator
    pix2pix_D = Pix2PixDiscriminator(in_channels=TRAIN_CONFIG['d_input_channels'],
                                    nfeature=TRAIN_CONFIG['nfeatures'])
    init_weights(pix2pix_D)
    pix2pix_D.to(device)
    pix2pix_D.train()

    #손실함수 정의
    l1_loss = nn.L1Loss().to(device)
    BCE_loss = nn.BCELoss().to(device)

    #optimizer정의
    optim_G = torch.optim.Adam(pix2pix_G.parameters(), lr=TRAIN_CONFIG['learning_rate'], betas=(0.5,0.999)) 
    optim_D = torch.optim.Adam(pix2pix_D.parameters(), lr=TRAIN_CONFIG['learning_rate'], betas=(0.5,0.999))

    #grad 초기화
    pix2pix_G.zero_grad()
    pix2pix_D.zero_grad()
    optim_G.zero_grad()
    optim_D.zero_grad()

    #training iterator정의
    train_iterator = trange(TRAIN_CONFIG['epoch'], desc='Epoch')

    #각종 지표 저장 리스트
    loss_D_total_list = []
    loss_G_total_list = []   
    PSNR_total_list = []
    loss_D_total_list_val = []
    loss_G_total_list_val = []   
    PSNR_total_list_val = []

    #epoch당 training을 전부 한 후 validation실행
    for _epoch, _ in enumerate(train_iterator):
        #index가 0부터 시작하여 1을 더해줌
        epoch = _epoch+1
        epoch_iterator = tqdm(train_dataloader, 'Iteration')
        loss_D_list = []
        loss_G_list = []
        PSNR_list = []

        #validation때  eval모드로 변경하므로 다시 train으로 변경
        pix2pix_G.train()
        pix2pix_D.train()

        #training
        for step, batch in enumerate(epoch_iterator):            
            input_batch, gt_batch = batch
            
            input_batch = input_batch.to(device)
            gt_batch = gt_batch.to(device)

            #Generator를통해 fake_image생성
            fake_image = pix2pix_G(input_batch)
            
            #input_image를 각각 fake와 gt와 concat fake는 그래디언트 전파를 막기 위해 detach
            real_pair = torch.cat([input_batch, gt_batch], dim=1)
            fake_pair = torch.cat([input_batch, fake_image.detach()], dim=1)

            #Discriminator forward
            real_D = pix2pix_D(real_pair)
            fake_D = pix2pix_D(fake_pair)

            #real은 1, fake는 0으로 설정 후 loss 계산
            real_D_loss = BCE_loss(real_D, torch.ones_like(real_D))
            fake_D_loss = BCE_loss(fake_D, torch.zeros_like(fake_D))


            total_D_loss = (real_D_loss + fake_D_loss) / 2

            #Discriminator backward및 그래디언트 전파
            total_D_loss.backward()
            optim_D.step()

            #Generator학습시에는 Discriminator에 넣는 fake_image에 detach를 하지 않음
            fake_pair = torch.cat([input_batch, fake_image], dim=1)
            fake_D = pix2pix_D(fake_pair)
            
            #Generator의 loss계산
            loss_G = BCE_loss(fake_D, torch.ones_like(fake_D))
            loss_G += TRAIN_CONFIG['l1_lambda'] * l1_loss(fake_image, gt_batch)

            #Generator 역전파
            loss_G.backward()
            optim_G.step()

            #그래디언트 초기화
            optim_D.zero_grad()
            optim_G.zero_grad()

            #각 지표 저장
            loss_D_list.append(total_D_loss)
            loss_G_list.append(loss_G)
            fake_image = [img_de_norm(img) for img in fake_image.to('cpu')]
            gt_image = [img_de_norm(img) for img in gt_batch.to('cpu')]

            #psnr계산
            psnr = PSNR(fake_image, gt_image)
            PSNR_list.append(psnr)
        

        #validation 단계
        loss_D_list_val = []
        loss_G_list_val = []
        PSNR_list_val = []
        with torch.no_grad():
            pix2pix_G.eval()
            pix2pix_D.eval()
            epoch_iterator = tqdm(val_dataloader, 'val_Iteration')
            for step, batch in enumerate(epoch_iterator):

                #validation에서는 loss를 계산하지만 optimizer를 실행시키지는 않음
                input_batch, gt_batch = batch
                input_batch = input_batch.to(device)
                gt_batch = gt_batch.to(device)
                
                fake_image = pix2pix_G(input_batch)
                
                real_pair = torch.cat([input_batch, gt_batch], dim=1)
                fake_pair = torch.cat([input_batch, fake_image.detach()], dim=1)

                real_D = pix2pix_D(real_pair)
                fake_D = pix2pix_D(fake_pair)

                real_D_loss = BCE_loss(real_D, torch.ones_like(real_D))
                fake_D_loss = BCE_loss(fake_D, torch.zeros_like(fake_D))

                total_D_loss = (real_D_loss + fake_D_loss) / 2

                #Generator
                fake_pair = torch.cat([input_batch, fake_image], dim=1)
                fake_D = pix2pix_D(fake_pair)
                
                loss_G = BCE_loss(fake_D, torch.ones_like(fake_D))
                loss_G += TRAIN_CONFIG['l1_lambda'] * l1_loss(fake_image, gt_batch)

                input_image = [img_de_norm(img) for img in input_batch.to('cpu')]
                fake_image = [img_de_norm(img) for img in fake_image.to('cpu')]
                gt_image = [img_de_norm(img) for img in gt_batch.to('cpu')]

                loss_D_list_val.append(total_D_loss)
                loss_G_list_val.append(loss_G)
            
                psnr = PSNR(fake_image, gt_image)
                PSNR_list_val.append(psnr)
                
                #validation때 첫번째 배치의 5개이미지에 대한 결과를 저장
                #epoch마다 이미지의 변화를 확인하기 위해 저장함
                #validation은 data를 shuffle하지 않기 때문에 같은 이미지가 나옴
                if step == 0:
                    save_image(input_image[:5], fake_image[:5], gt_image[:5], epoch)


        loss_D_mean_t = mean(loss_D_list)
        loss_G_mean_t = mean(loss_G_list)
        PSNR_mean_t = mean(PSNR_list)
        loss_D_mean_v = mean(loss_D_list_val)
        loss_G_mean_v = mean(loss_G_list_val)
        PSNR_mean_v = mean(PSNR_list_val)

        loss_D_total_list.append(loss_D_mean_t)
        loss_G_total_list.append(loss_G_mean_t)
        PSNR_total_list.append(PSNR_mean_t)
        loss_D_total_list_val.append(loss_D_mean_v)
        loss_G_total_list_val.append(loss_G_mean_v)
        PSNR_total_list_val.append(PSNR_mean_v)
        

        #epoch결과 출력
        print(f'Epoch-{epoch} result', f'train loss : Generator - {loss_G_mean_t}, Discriminator - {loss_D_mean_t}, PSNR - {PSNR_mean_t}', 
        f'val loss : Generator - {loss_G_mean_v}, Discriminator - {loss_D_mean_v}, PSNR - {PSNR_mean_v}', sep='\n')

        #각 지표에 대한 그래프저장 (train vs validation)
        save_plot(epoch, loss_D_total_list, loss_D_total_list_val, 'loss_D')
        save_plot(epoch, loss_G_total_list, loss_G_total_list_val, 'loss_G')
        save_plot(epoch, PSNR_total_list, PSNR_total_list_val, 'PSNR')

        #각 지표 값 저장
        with open(f'./{TRAIN_CONFIG["save_dir"]}/result.csv', 'a') as f:
            f.write(f'{epoch},{loss_D_total_list[epoch-1]},{loss_D_total_list_val[epoch-1]},{loss_G_total_list[epoch-1]},{loss_G_total_list_val[epoch-1]},{PSNR_total_list[epoch-1]},{PSNR_total_list_val[epoch-1]}\n')


        #설정한 epoch단위로 모델 저장
        if (epoch % TRAIN_CONFIG['model_save_epoch']) == 0:
            torch.save({
                'epoch': epoch,
                'pix2pix_G_state_dict': pix2pix_G.state_dict(),
                'pix2pix_D_state_dict': pix2pix_D.state_dict(),
                'optim_G_state_dict': optim_G.state_dict(),
                'optim_D_stati_dict': optim_D.state_dict(),
            }, f'./{TRAIN_CONFIG["save_dir"]}/models/pix2pix_{epoch}.pt')

if __name__ == '__main__':
    if TRAIN_CONFIG['device'] == 'cuda':
        device = torch.device(f'cuda:{TRAIN_CONFIG["gpu_num"]}' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    train(device=device)

