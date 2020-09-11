import argparse
import csv
import numpy as np
import os
from subprocess import check_call
from imageio import imwrite
from torch import squeeze, unsqueeze, from_numpy, max, tensor
from utils import nhwc_to_nchw, nchw_to_nhwc
from lib import ReCoNetModel as StyleModel
from mit_semseg.models import SegmentationModel
from ffmpeg_tools import VideoReader, VideoWriter
from mit_semseg.config import cfg

def jp(*x): return os.path.expanduser(os.path.join(*x))
DEFAULT_MODEL = "./model_mosaic_2.pth"
ROOT_DIR = os.path.expanduser(jp('~','localized-style-distortion'))
TEST_DIR  =jp(jp(ROOT_DIR, 'data'), 'roadster')
LABEL_CSV = jp(ROOT_DIR,'data/object150_info.csv')

def get_mask(image, index): return unsqueeze(squeeze((image[:,index,:,:]) > 0.25).int(), -1)
def to_image(x): return squeeze(x).cpu().numpy().astype('uint8')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=TEST_DIR, help="Path to input video directory.")
    parser.add_argument("--input", default=f'{TEST_DIR}\\input.mp4', help="Path to input video file")
    parser.add_argument("--output", default=f'{TEST_DIR}\\output.mp4', help="Path to output style video file")
    parser.add_argument("--label", default="car", help="Path to output style video file")
    parser.add_argument("--use-cpu", action='store_true', help="Use CPU instead of GPU")
    parser.add_argument("--gpu-device", type=int, default=None, help="GPU device index")
    parser.add_argument("--fps", type=int, default=None, help="FPS of output video")
    args = parser.parse_args()
    label_index = -1
    with open(LABEL_CSV, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if label_index > 0: break
            for column in row:
                if (args.label in column): label_index = int(row[0]) - 1
    style_model = StyleModel(DEFAULT_MODEL, use_gpu=not args.use_cpu, gpu_device=args.gpu_device)
    cfg.merge_from_file("config/ade20k-resnet50dilated-ppm_deepsup.yaml")
    segmentation_model = SegmentationModel(
            DEFAULT_MODEL,
            use_gpu=not args.use_cpu,
            gpu_device=args.gpu_device,
            cfg=cfg)
    fix_path = lambda x: x.replace('\\', r'\\')
    reader = VideoReader(fix_path(args.input), fps=args.fps)
    output_writer = VideoWriter(fix_path(args.output), reader.width, reader.height, reader.fps)
    with output_writer:
        frame_number = 0
        for frame in reader:
            print(f"Processing frame {frame_number}")
            frame_number+=1
            image = np.array(frame)
            style = style_model.run(image)
            segmentation = segmentation_model.run(image)
            mask = get_mask(segmentation, label_index)
            inverse_mask = (255 - mask)// 255
            masked_image = from_numpy(image).cuda() * mask
            masked_style = squeeze(style * inverse_mask)
            output_image = masked_image + masked_style
            output_writer.write(to_image(output_image))
