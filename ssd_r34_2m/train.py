import os
from argparse import ArgumentParser
from utils import DefaultBoxes, Encoder, COCODetection
from base_model import Loss
from utils import SSDTransformer
from ssd_r34 import SSD_R34
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import time
import numpy as np


def parse_args():
    parser = ArgumentParser(description="Train Single Shot MultiBox Detector"
                                        " on COCO")
    parser.add_argument('--data', '-d', type=str, default='../coco',
                        help='path to test and training data files')
    parser.add_argument('--epochs', '-e', type=int, default=800,
                        help='number of epochs for training')
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                        help='number of examples for each iteration')
    parser.add_argument('--no-cuda', action='store_true',
                        help='use available GPUs')
    parser.add_argument('--seed', '-s', type=int,
                        help='manually set random seed for torch')
    parser.add_argument('--device', '-did', type=int,
                        help='device id')  
    parser.add_argument('--device-ids', default=[0], type=int, nargs='+',
                    help='device ids assignment (e.g 0 1 2 3')                                      
    parser.add_argument('--threshold', '-t', type=float, default=0.212,
                        help='stop training early at threshold')
    parser.add_argument('--iteration', type=int, default=0,
                        help='iteration to start from')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path to model checkpoint file')
    parser.add_argument('--no-save', action='store_true',
                        help='save model checkpoints')
    parser.add_argument('--save-path', '-d', type=str, default='./models',
                        help='path to saved models files')                        
    parser.add_argument('--evaluation', nargs='*', type=int,
                        default=[1000,10000, 40000, 80000, 120000, 160000, 180000, 200000, 220000, 240000],
                        help='iterations at which to evaluate')
    parser.add_argument('--image-size', default=[1400,1400], type=int, nargs='+',
                        help='input image sizes (e.g 1400 1400,300 300,1600 1200')  
    parser.add_argument('--strides', default=[1,1,2,2,2,1], type=int, nargs='+',
                        help='stides for ssd model must include 6 numbers')                                       
    return parser.parse_args()


def show_memusage(device=0):
    import gpustat
    gpu_stats = gpustat.GPUStatCollection.new_query()
    item = gpu_stats.jsonify()["gpus"][device]
    print("{}/{}".format(item["memory.used"], item["memory.total"]))

def get_scales(min_scale=0.2, max_scale=0.9,num_layers=6):
    """ Following the ssd arxiv paper, regarding the calculation of scales & ratios
    Parameters
    ----------
    min_scale : float
    max_scales: float
    num_layers: int
        number of layers that will have a detection head
    anchor_ratios: list
    first_layer_ratios: list
    return
    ------
    sizes : list
        list of scale sizes per feature layer
    ratios : list
        list of anchor_ratios per feature layer
    """

    # this code follows the original implementation of wei liu
    # for more, look at ssd/score_ssd_pascal.py:310 in the original caffe implementation
    min_ratio = int(min_scale * 100)
    max_ratio = int(max_scale * 100)
    step = int(np.floor((max_ratio - min_ratio) / (num_layers - 2)))
    min_sizes = []
    max_sizes = []
    for ratio in range(min_ratio, max_ratio + 1, step):
        min_sizes.append(ratio / 100.)
        max_sizes.append((ratio + step) / 100.)
    min_sizes = [int(100*min_scale / 2.0) / 100.0] + min_sizes
    max_sizes = [min_scale] + max_sizes

    # convert it back to this implementation's notation:
    scales = []
    for layer_idx in range(num_layers):
        scales.append([min_sizes[layer_idx], np.single(np.sqrt(min_sizes[layer_idx] * max_sizes[layer_idx]))])
    return scales

def dboxes_coco(figsize,strides):
    ssd_r34=SSD_R34(81,strides=strides).to('cuda')
    synt_img=torch.rand([1,3]+figsize).to('cuda')
    _,_,feat_size =ssd_r34(synt_img, extract_shapes = True)
    steps=[(int(figsize[0]/fs[0]),int(figsize[1]/fs[1])) for fs in feat_size]
    scales = [(int(s*figsize[0]/300),int(s*figsize[1]/300)) for s in [21, 45, 99, 153, 207, 261, 315]] 
    aspect_ratios =  [[2], [2, 3], [2, 3], [2, 3], [2], [2]] 
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    print('Total number of anchors is: ', dboxes.dboxes.shape[0])
    return dboxes


def coco_eval(model, coco, cocoGt, encoder, inv_map, threshold,device_ids,use_cuda=True):
    from pycocotools.cocoeval import COCOeval
    print("")
    model.eval()
    device=device_ids[0]
    model.to('cuda')
    if use_cuda:
        if device_ids and len(device_ids) > 1:
            model = nn.DataParallel(model, device_ids)
        
    ret = []
    start = time.time()
    for idx, image_id in enumerate(coco.img_keys):
        img, (htot, wtot), _, _ = coco[idx]

        with torch.no_grad():
            print("Parsing image: {}/{}".format(idx+1, len(coco)), end="\r")
            inp = img.unsqueeze(0)
            if use_cuda:
                inp = inp.to('cuda')
            ploc, plabel, _ = model(inp)
            try:
                result = encoder.decode_batch(ploc, plabel, 0.50, 200,device=device)[0]
            except:
                #raise
                print("")
                print("No object detected in idx: {}".format(idx))
                continue

            loc, label, prob = [r.cpu().numpy() for r in result]
            for loc_, label_, prob_ in zip(loc, label, prob):
                ret.append([image_id, loc_[0]*wtot, \
                                      loc_[1]*htot,
                                      (loc_[2] - loc_[0])*wtot,
                                      (loc_[3] - loc_[1])*htot,
                                      prob_,
                                      inv_map[label_]])
    print("")
    print("Predicting Ended, total time: {:.2f} s".format(time.time()-start))
    cocoDt = cocoGt.loadRes(np.array(ret))

    E = COCOeval(cocoGt, cocoDt, iouType='bbox')
    E.evaluate()
    E.accumulate()
    E.summarize()
    print("Current AP: {:.5f} AP goal: {:.5f}".format(E.stats[0], threshold))

    # put your model back into training mode
    model.train()

    return (E.stats[0] >= threshold) #Average Precision  (AP) @[ IoU=050:0.95 | area=   all | maxDets=100 ]



def train_mlperf_coco(args):
    from coco import COCO
    # Check that GPUs are actually available
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    ssd_r34 = SSD_R34(81,strides=args.strides)
    #img_size=[args.image_size,args.image_size]
    dboxes = dboxes_coco(args.image_size,args.strides)
    encoder = Encoder(dboxes)
    train_trans = SSDTransformer(dboxes, tuple(args.image_size), val=False)
    val_trans = SSDTransformer(dboxes, tuple(args.image_size), val=True)

    val_annotate = os.path.join(args.data, "annotations/instances_val2017.json")
    val_coco_root = os.path.join(args.data, "val2017")
    train_annotate = os.path.join(args.data, "annotations/instances_train2017.json")
    train_coco_root = os.path.join(args.data, "train2017")

    cocoGt = COCO(annotation_file=val_annotate)
    val_coco = COCODetection(val_coco_root, val_annotate, val_trans)
    train_coco = COCODetection(train_coco_root, train_annotate, train_trans)

    #print("Number of labels: {}".format(train_coco.labelnum))
    train_dataloader = DataLoader(train_coco, batch_size=args.batch_size, shuffle=True, num_workers=4)

    ssd_r34 = SSD_R34(train_coco.labelnum,strides=args.strides)
    if args.checkpoint is not None:
        print("loading model checkpoint", args.checkpoint)
        od = torch.load(args.checkpoint)
        ssd_r34.load_state_dict(od["model"])
    ssd_r34.train()
    ssd_r34.to('cuda')
    if use_cuda:
        if args.device_ids and len(args.device_ids) > 1:
            ssd_r34 = nn.DataParallel(ssd_r34, args.device_ids)   

    loss_func = Loss(dboxes)
    if use_cuda:
        loss_func.to('cuda')
        loss_func = nn.DataParallel(loss_func, args.device_ids)   

    optim = torch.optim.SGD(ssd_r34.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    print("epoch", "nbatch", "loss")

    iter_num = args.iteration
    avg_loss = 0.0
    last_loss=[0.0]*10
    inv_map = {v:k for k,v in val_coco.label_map.items()}

    for epoch in range(args.epochs):

        for nbatch, (img, img_size, bbox, label) in enumerate(train_dataloader):

            if iter_num == 160000:
                print("")
                print("lr decay step #1")
                for param_group in optim.param_groups:
                    param_group['lr'] = 1e-4

            if iter_num == 200000:
                print("")
                print("lr decay step #2")
                for param_group in optim.param_groups:
                    param_group['lr'] = 1e-5


            img = Variable(img, requires_grad=True)
            ploc, plabel,_ = ssd_r34(img.to('cuda'))
            trans_bbox = bbox.transpose(1,2).contiguous()

            gloc, glabel = Variable(trans_bbox, requires_grad=False), \
                           Variable(label, requires_grad=False)
          
            loss = loss_func(ploc, plabel, gloc, glabel).mean()

            if not np.isinf(loss.item()): avg_loss = 0.999*avg_loss + 0.001*loss.item()
            last_loss.pop()
            last_loss=[loss.item()]+last_loss
            avg_last_loss=sum(last_loss)/len(last_loss)
            print("Iteration: {:6d}, Loss function: {:5.3f}, Average Loss: {:.3f}, Average Last 10 Loss: {:.3f}"\
                        .format(iter_num, loss.item(), avg_loss,avg_last_loss), end="\r")
            optim.zero_grad()
            loss.backward()
            optim.step()

            loss = None

            if iter_num in args.evaluation:
                if not args.no_save:
                    print("")
                    print("saving model...")
                    module = ssd_r34.module if len(args.device_ids)>1 else ssd_r34
                    torch.save({"model" : module.state_dict(), "label_map": train_coco.label_info},
                                args.save_path+"/iter_{}.pt".format(iter_num))
                if coco_eval(ssd_r34, val_coco, cocoGt, encoder, inv_map, args.threshold,args.device_ids):
                    return

            iter_num += 1

def main():
    args = parse_args()

    if not os.path.isdir('./models'):
        os.mkdir('./models')

    if args.seed is not None:
        print("Using seed = {}".format(args.seed))
        torch.manual_seed(args.seed)
        np.random.seed(seed=args.seed)

    torch.cuda.set_device(args.device_ids[0])
    torch.backends.cudnn.benchmark = True
    train_mlperf_coco(args)

if __name__ == "__main__":
    main()
