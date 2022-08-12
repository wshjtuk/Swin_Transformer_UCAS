import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import os
from models import build_model
from config import get_config
import argparse
import numpy as np
import shutil
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer Test script', add_help=False)
    parser.add_argument('--cfg', default='configs/mytest_0703.yaml', type=str, metavar="FILE",
                        help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', default='output/swin_tiny_patch4_window7_224/default/ckpt_epoch_1.pth',
                        help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument("--local_rank", default='0', type=int, help='local rank for DistributedDataParallel')
    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_, config = parse_option()
model = build_model(config)
check = 'output/ckpt_epoch_0712.pth'
checkpoint = torch.load(
    check, map_location='cpu')
model.load_state_dict(checkpoint['model'], strict=False)
model.eval()
model.to(DEVICE)
image_path = "test/test_image"
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
labels = ["00-现代极简", "01-侘寂风格", "02-工业风风格", "03-日式风格", "04-现代简约", "05-现代轻奢", "06-MUJI风格", "07-北欧风格", "08-古典中式", "09-新中式","10-欧式风格","11-美式风格"]
accurate = 0
num = 0
Note=open('test/result/result.txt',mode='w')
for dir, folder, files in os.walk(image_path):
    for file in files:
        image_path2 = image_path + '/' + file
        img = Image.open(image_path2).convert('RGB')
        img = transform_test(img)
        img.unsqueeze_(0)
        img = Variable(img).to(DEVICE)
        out = model(img)
        _,pred = torch.max(out.data,1)
        predict = pred.data.item()
        Note.write(file + " => "+ labels[predict] + '\n')
        print(file + " => "+ labels[predict])
        shutil.copyfile(image_path2, "test/result/" + labels[predict] + "/" + file)
            
