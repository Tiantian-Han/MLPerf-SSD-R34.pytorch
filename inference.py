import os
import time
import torch
import numpy as np
from math import ceil
from ssd import SSD
from utils import SSDTransformer
from argparse import ArgumentParser
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils import DefaultBoxes, Encoder, COCODetection

try:
  from apex.fp16_utils import *
except ImportError:
  raise ImportError("no apex")

from coco import COCO
from pycocotools.cocoeval import COCOeval

def parse_args():
    parser = ArgumentParser(description="Train Single Shot MultiBox Detector"
                                        " on COCO")
    parser.add_argument('--data', '-d', type=str, default='/coco',
                        help='path to test and training data files')
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                        help='number of examples for each iteration')
    parser.add_argument('--seed', '-s', type=int,
                        help='manually set random seed for torch')
    parser.add_argument('--model', type=str, default='./model.pt',
                        help='path to pretrained model')
    parser.add_argument('--dboxes-scale', type=float, default=1.0,
                        help='scale factor for 300x300 default boxes')
    parser.add_argument('--dboxes-steps', nargs='*', type=int, default=[],
                        help='steps for default boxes')
    parser.add_argument('--image-size', default=[300,600], type=int, nargs='+',
                        help='input image sizes (e.g 300 600, 750 1400')
    parser.add_argument('--small-mask', default=[1, 1, 1, 1, 1, 1], type=int, nargs='+',
                        help='binary mask for features maps of small input images')
    parser.add_argument('--large-mask', default=[1, 1, 1, 1, 1, 1], type=int, nargs='+',
                        help='binary mask for features maps of small input images')
    parser.add_argument('--use-filter', action='store_true',
                        help='use filter')
    parser.add_argument('--use-fp16', action='store_true')
    return parser.parse_args()


# Returns a list of feature map sizes for R34 SSD network, given an input size.
def get_r34_featuremap_sizes(input_size):
    featuremap_sizes = []

    # Go through the modified R34 backbone, till the first feature map for detector heads.
    output_size = input_size
    output_size = ceil(output_size / 2) # 7x7s2 output
    output_size = ceil(output_size / 2) # maxpool output
    output_size = ceil(output_size / 2) # layer2 output
    featuremap_sizes.append(output_size)

    # The next 3 feature maps are from downsampling.
    output_size = ceil(output_size / 2)
    featuremap_sizes.append(output_size)
    output_size = ceil(output_size / 2)
    featuremap_sizes.append(output_size)
    output_size = ceil(output_size / 2)
    featuremap_sizes.append(output_size)

    # The last 2 feature maps are just unpadded 3x3 filters.
    output_size = output_size - 2
    featuremap_sizes.append(output_size)
    output_size = output_size - 2
    featuremap_sizes.append(output_size)
    return featuremap_sizes


def default_boxes(figsize, scale_factor, steps, mask):
    #feat_size = [38, 19, 10, 5, 3, 1]
    feat_size = get_r34_featuremap_sizes(figsize)
    #steps = [8, 16, 32, 64, 100, 300]
    # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
    scales = [np.ceil(s*scale_factor) for s in [21, 45, 99, 153, 207, 261, 315]]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios, mask)
    return dboxes


def areas(bboxes, wtot, htot):
    b = bboxes.squeeze(0)
    w = (b[:,2] - b[:,0])*wtot
    h = (b[:,3] - b[:,1])*htot
    return  w * h


def inference(model, coco1, coco2, cocoGt, encoder1, encoder2, inv_map, use_cuda=True, use_fp16=False, use_filter=False):
    ret = []
    start = time.time()
    # Loop over images.
    for idx, image_id in enumerate(coco1.img_keys):
        print("Iteration {} from {}".format(idx+1, len(coco1)), end="\n")
        # Transform image.
        img1, (htot, wtot), _, _ = coco1[idx]
        img2, (htot, wtot), _, _ = coco2[idx]
        #with torch.no_grad():
        inp1 = img1.unsqueeze(0)
        inp2 = img2.unsqueeze(0)

        # Copy to device and convert to half precision
        if use_cuda:
            inp1 = inp1.cuda()
            inp2 = inp2.cuda()
        if use_fp16:
            inp1 = inp1.half()
            inp2 = inp2.half()

        # Run inference.
        ploc1, plabel1 = model(inp1)
        ploc2, plabel2 = model(inp2)
        ploc1, plabel1 = ploc1.float(), plabel1.float()
        ploc2, plabel2 = ploc2.float(), plabel2.float()

        # Convert.
        bboxes1, probs1 = encoder1.scale_back_batch(ploc1, plabel1)
        bboxes2, probs2 = encoder2.scale_back_batch(ploc2, plabel2)

        # Filter large or small/medium detections
        if use_filter:
            mask1 = areas(bboxes1, wtot, htot) > (96 * 96)
            mask2 = areas(bboxes2, wtot, htot) < (96 * 96)
            bboxes1 = bboxes1[:,mask1,:]
            bboxes2 = bboxes2[:,mask2,:]
            probs1 = probs1[:,mask1,:]
            probs2 = probs2[:,mask2,:]

        # Clamp bounding boxes.
        bboxes1.clamp(0, 1)
        bboxes2.clamp(0, 1)

        # Run nms on all detections.
        bboxes = torch.cat((bboxes1, bboxes2), 1)
        probs = torch.cat((probs1, probs2), 1)
        result = encoder1.decode_batch_(bboxes, probs, 0.5, 200)[0]

        # Prepare results for cocoapi.
        loc, label, prob = [r.cpu().numpy() for r in result]
        for loc_, label_, prob_ in zip(loc, label, prob):
            ret.append([image_id, loc_[0]*wtot, \
                                  loc_[1]*htot,
                                  (loc_[2] - loc_[0])*wtot,
                                  (loc_[3] - loc_[1])*htot,
                                  prob_,
                                  inv_map[label_]])

    # Evaluate predictions.
    cocoDt = cocoGt.loadRes(np.array(ret))
    E = COCOeval(cocoGt, cocoDt, iouType='bbox')
    E.evaluate()
    E.accumulate()
    E.summarize()

    # Print time and score.
    print("Inference time: {:.2f} s".format(time.time()-start))
    print("Inference score: {:.5f}".format(E.stats[0]))


def parse_state_dict(state_dict):
    import re
    parsed_state_dict = {}
    for k,v in state_dict.items():
        prefix = ""
        if "1." in k: prefix = "^1."
        if "module.1." in k: prefix = "^module.1."
        parsed_state_dict[re.sub(prefix, "", k)] = v
    return parsed_state_dict

def mlperf_inference(args):
    use_cuda = torch.cuda.is_available()
    # Build default boxes, encoder, and transformer.
    #mask1 = [0, 1, 1, 1, 1, 1]
    #mask2 = [1, 1, 1, 1, 0, 0]
    dboxes1 = default_boxes(args.image_size[0], args.dboxes_scale, args.dboxes_steps, args.small_mask)
    dboxes2 = default_boxes(args.image_size[1], args.dboxes_scale, args.dboxes_steps, args.large_mask)
    encoder1 = Encoder(dboxes1)
    encoder2 = Encoder(dboxes2)
    trans1 = SSDTransformer(dboxes1, (args.image_size[0], args.image_size[0]), val=True)
    trans2 = SSDTransformer(dboxes2, (args.image_size[1], args.image_size[1]), val=True)

    # Get coco detection data set.
    annotate = os.path.join(args.data, "annotations/instances_val2017.json")
    coco_root = os.path.join(args.data, "images/val2017")
    cocoGt = COCO(annotation_file=annotate)
    coco1 = COCODetection(coco_root, annotate, trans1)
    coco2 = COCODetection(coco_root, annotate, trans2)
    inv_map = {v:k for k,v in coco1.label_map.items()}

    # Load model.
    masks = {}
    masks[args.image_size[0]] = args.small_mask
    masks[args.image_size[1]] = args.large_mask
    model = SSD(coco1.labelnum, masks)
    checkpoint = torch.load(args.model)
    model.load_state_dict(parse_state_dict(checkpoint['model']))
    model.eval()

    # Copy to device and convert to half precision.
    if use_cuda:
        model.cuda()
    if args.use_fp16:
        model = network_to_half(model)

    # Run inference.
    with torch.no_grad():
        inference(model, coco1, coco2, cocoGt, encoder1, encoder2, inv_map, use_fp16=args.use_fp16, use_filter=args.use_filter)


def main():
    # Initialization.
    args = parse_args()
    if args.seed is not None:
        print("Using seed = {}".format(args.seed))
        torch.manual_seed(args.seed)
        np.random.seed(seed=args.seed)
    torch.backends.cudnn.benchmark = True

    # MLPerf inference.
    mlperf_inference(args)

if __name__ == "__main__":
    main()
