import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2

import random
import re
import os
import math

def get_image_path(data_path, output_file, input_dir, gt_dir):
    #directory path
    input_image_dir_path = os.path.join(data_path, input_dir)
    gt_image_dir_path = os.path.join(data_path, gt_dir)
    #filelist in directory
    input_image_name_list = os.listdir(input_image_dir_path)
    gt_image_name_list = os.listdir(gt_image_dir_path)

    #이름_xx_xx.확장자 형태를 추출하는 정규식
    regex_string = '(\w+)_(\w+_\w+)\.(\w+)'
    regex = re.compile(regex_string)
    #파일 매칭 결과 저장 to csv
    with open(os.path.join(data_path, output_file), 'w') as f:
        #input 이미지에 대하여 이름 추출 후
        #접미사 _G_AS.png를 붙여서 ground truth 이미지 파일과 매칭
        for image_name in input_image_name_list:
            #정규식 실행 결과 
            regex_result = regex.match(image_name)
            #정규식 결과의 1,2,3번 그룹을 각 이미지이름, 중간부분, 확장자로 나눔
            name, _, ext = regex_result.group(1), regex_result.group(2), regex_result.group(3)
            #이름에 GROUND TRUTH를 나타내는 접미사 붙인 후 저장
            gt_name = name+'_G_AS.png'
            #csv이므로 구분자 ,를 사용
            f.write(f'{gt_name},{image_name}\n')

#train_data, val_data, test_data로 데이터를 전부 나눈후 파일리스트로 각각 나누어 저장
def split_data(data_path, train_size=0.8, val_size=0.1, test_size=0.1):
    with open(os.path.join(data_path, 'data_list.csv'), 'r') as f:
        temp = f.read().splitlines()
        random.shuffle(temp)

        train_idx = math.ceil(len(temp)*train_size)
        val_idx = math.ceil(len(temp)*(train_size+val_size))

        train_data = temp[:train_idx]
        val_data = temp[train_idx:val_idx]
        test_data = temp[val_idx:]

        with open(os.path.join(data_path, 'train_data.csv'), 'w') as f:
            train_data = '\n'.join(train_data)
            f.write(train_data)
        
        with open(os.path.join(data_path, 'val_data.csv'), 'w') as f:
            val_data = '\n'.join(val_data)
            f.write(val_data)

        with open(os.path.join(data_path, 'test_data.csv'), 'w') as f:
            test_data = '\n'.join(test_data)
            f.write(test_data)
        


#Custom Dataset Class
class MyDataset(Dataset):
    def __init__(self, data_path, list_file, input_dir, gt_dir, transform='RESIZE', shuffle=True):
        #file_list를 저장할 파일
        self.file_list_path = os.path.join(data_path, list_file)
        #input_image폴더 경로
        self.input_image_dir = os.path.join(data_path, input_dir)
        #ground_truth폴더 경로
        self.gt_image_dir = os.path.join(data_path, gt_dir)
        
        #file_list 데이터 load
        self.image_name_list = self.__file_list_load()

        self.transforms = transform
        #데이터 shuffle 활성화여부 저장
        self.shuffle = shuffle
        if self.shuffle:
            self.shuffle_data()


    #file_list가 적힌 파일을 가져와서 input_image, ground_truth이미지 쌍의 이름을 저장
    def __file_list_load(self):
        data_list = list()
        #file_list가 적힌 파일을 open
        with open(self.file_list_path, 'r') as f:
            #파일의 내용을 전부 읽어온 후 개행문자를 기준으로 줄단위로 리스트로 저장
            temp = f.read().splitlines()

            for t in temp:
                #csv파일이므로 ,를 기준으로 분할
                t = t.split(',')
                gt_name, input_name = t[0], t[1]
                data_list.append([input_name, gt_name])
        
        return data_list
    
    #데이터셋의 순서를 변경
    def shuffle_data(self):
        random.shuffle(self.image_name_list)

    #데이터셋 길이
    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        #인덱스에 해당하는 파일 이름 가져옴
        input_name, gt_name = self.image_name_list[idx]
        #파일 이름과 디렉터리 경로로 경로 생성 후 이미지 read
        input_image = cv2.imread(os.path.join(self.input_image_dir,input_name), cv2.IMREAD_COLOR)
        gt_image = cv2.imread(os.path.join(self.gt_image_dir, gt_name), cv2.IMREAD_COLOR)

        if (input_image is None) or (gt_image is None):
            return None
        #opencv의 image처리 형태가 BGR형태이므로 흔히 아는 RGB로 변경
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)

        #transform함수 따로 구현하지 않고 그냥 내부에서 두가지 모드만 실행
        if self.transforms == 'RESIZE':
            input_image = cv2.resize(input_image, dsize=(256,256), interpolation=cv2.INTER_AREA)
            gt_image = cv2.resize(gt_image, dsize=(256,256), interpolation=cv2.INTER_AREA)
        elif self.transforms == 'CROP':
            h, w, c = input_image.shape
            w_rand = random.randint(0, w-256)
            h_rand = random.randint(0, h-256)
            input_image = input_image[h_rand:h_rand+256, w_rand:w_rand+256]
            gt_image = gt_image[h_rand:h_rand+256 , w_rand:w_rand+256]

        
        input_image = np.transpose(input_image, (2,0,1)).astype(np.float32)
        gt_image = np.transpose(gt_image, (2,0,1)).astype(np.float32)

        #-1 ~ 1 의 값으로 정규화
        return (input_image/127.5)-1, (gt_image/127.5)-1


#각 디렉토리 경로값 지정
DATA_PATH = '/home/kmsjames/cs/data'
INPUT_IMAGE_DIR = 'Set1_input_images'
GROUND_TRUTH_IMAGE_DIR = 'Set1_ground_truth_images'
FILE_LIST_PATH = 'data_list.csv'

if __name__ == '__main__':
    #file_list 생성
    #get_image_path(data_path=DATA_PATH, output_file=FILE_LIST_PATH, 
    #    input_dir=INPUT_IMAGE_DIR, gt_dir=GROUND_TRUTH_IMAGE_DIR)
    #custom dataset 인스턴스 생성
    dataset = MyDataset(data_path=DATA_PATH, list_file=FILE_LIST_PATH,
        input_dir=INPUT_IMAGE_DIR, gt_dir=GROUND_TRUTH_IMAGE_DIR, shuffle=True)

    #dataset의 0번 데이터 가져옴
    input_image, gt_image = dataset[0]
    
    #cv로 받아온 이미지는 [Width, Height, Channel]이므로 torch.Tensor에 맞춰서 [Channel, Width, Height]로 변경
    input_tensor = torch.Tensor(np.transpose(input_image, (2,0,1)))
    gt_tensor = torch.Tensor(np.transpose(gt_image, (2,0,1)))

    #split_data(DATA_PATH)
    with open(os.path.join(DATA_PATH, 'test_data.csv')) as f:
        t = f.read().splitlines()
        print(len(t))


