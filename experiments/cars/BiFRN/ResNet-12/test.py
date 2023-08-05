import sys
import os
import torch
import yaml
sys.path.append('../../../../')
from models.BiFRN import BiFRN
from utils import util
from trainers.eval import meta_test


with open('../../../../config.yml', 'r') as f:
    temp = yaml.safe_load(f)
data_path = os.path.abspath(temp['data_path'])

test_path = os.path.join(data_path,'cars_196/test')
# model_path = './model_ResNet-12.pth'
#model_path = '../../../../trained_model_weights/CUB_fewshot_cropped/Proto/ResNet-12_1-shot/model.pth'
model_path = '/home/wujijie/2022/Bi-FRN/experiments/mini-ImageNet/BiFRN/model_ResNet-12.pth'


gpu = 0
torch.cuda.set_device(gpu)

model = BiFRN(resnet=True)
model.cuda()
model.load_state_dict(torch.load(model_path,map_location=util.get_device_map(gpu)),strict=True)
model.eval()

with torch.no_grad():
    way = 5
    for shot in [1, 5]:
        mean,interval = meta_test(data_path=test_path,
                                model=model,
                                way=way,
                                shot=shot,
                                pre=False,
                                transform_type=0,
                                trial=10000)
        print('%d-way-%d-shot acc: %.3f\t%.3f'%(way,shot,mean,interval))