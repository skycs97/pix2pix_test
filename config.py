DATA_CONFIG={
    'data_path': 'in your data path',
    'input_image_dir': 'Set1_input_images',
    'ground_truth_image_dir': 'Set1_ground_truth_images',
    'file_list': 'data_list.csv',
    'train_list': 'train_data.csv',
    'val_list': 'val_data.csv',
    'test_list': 'test_data.csv'
}

TRAIN_CONFIG={
    'g_input_channels': 3,          #generator input channel
    'g_output_channels': 3,         #generator output channel
    'd_input_channels': 6,          #decriminator input channel (이미지 두개 concat)
    'nfeatures': 64,                #conv 채널 feature 단위
    'batch_size': 32,               #배치 사이즈
    'epoch':20,                     #학습 epoch
    'model_save_epoch': 5,          #모델 저장할 epoch단위
    'device': 'cuda',               #device 선택
    'gpu_num': 1,                   #cuda 선택 시 사용할 gpu번호
    'shuffle': True,                #training중 데이터셋 shuffle여부
    'num_workers': 4,               #dataloader worker갯수
    'learning_rate':0.0002,         #learning_rate 설정
    'l1_lambda': 100,               #l1_loss에 곱해줄 람다값
    'save_dir': 'temp'              #트레이닝 결과 저장소
}
