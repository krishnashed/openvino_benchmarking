from __future__ import print_function

import logging as log
import os
import pathlib
import json
import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore
import torch
import torchvision
import time
from matplotlib import pyplot as plt
import os.path
import argparse
import time


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    prediction=torch.from_numpy(prediction)
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                #print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output

def process_image(net, input_image, model_xml):
    """Do inference to analysis input_image and get output

    Attributes:
        net: model handle
        input_image (numpy.ndarray): image to be process, format: (h, w, c), BGR
        thresh: thresh value

    Returns: process result

    """
    newImage = input_image.copy()
    classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
    if not net or input_image is None:
        log.error('Invalid input args')
        return None
    # log.info(f'process_image, ({input_image.shape}')
    ih, iw, _ = input_image.shape
    # model_xml = "yolov5/yolov5s.xml
    ie = IECore()
    net = ie.read_network(model=model_xml)
    input_blob = next(iter(net.input_info))
    n, c, h, w = net.input_info[input_blob].input_data.shape
    global input_h, input_w, input_c, input_n
    input_h, input_w, input_c, input_n = h, w, c, n

    # --------------------------- Prepare input blobs -----------------------------------------------------
    if ih != input_h or iw != input_w:
        input_image = cv2.resize(input_image, (input_w, input_h))
    #input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = input_image/255
    input_image = input_image.transpose((2, 0, 1))
    images = np.ndarray(shape=(input_n, input_c, input_h, input_w))
    images[0] = input_image

    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))

    # --------------------------- Performing inference ----------------------------------------------------
    log.info("Creating infer request and starting inference")
    exec_net= ie.load_network(network=net, device_name="CPU")
    start_time = time.time()
    res = exec_net.infer(inputs={input_blob: images})
    
    # --------------------------- Read and postprocess output ---------------------------------------------

#    print(res.keys())
    data = res['561'] # add key of the layer
#    print(data.shape)
    data=non_max_suppression(data, 0.2, 0.4)
    end_time = time.time()
#    print("after NMS:",data)
    
    try:
        data=data[0].numpy() # batch size is one. modify to include further batches
        image = cv2.cvtColor(newImage, cv2.COLOR_BGR2RGB)
        detect_objs = []
        for proposal in data:
            if proposal[4] > 0 :
                confidence = proposal[4]
                xmin = np.int(iw * (proposal[0]/input_w))
                ymin = np.int(ih * (proposal[1]/input_h))
                xmax = np.int(iw * (proposal[2]/input_w))
                ymax = np.int(ih * (proposal[3]/input_h))
                # if label not in label_id_map:
                #     log.warning(f'{label} does not in {label_id_map}')
                #     continue
                detect_objs.append({
                    'name': classes[int(proposal[5])],
                    'xmin': int(xmin),
                    'ymin': int(ymin),
                    'xmax': int(xmax),
                    'ymax': int(ymax),
                    'confidence': float(confidence)
                })
                # Draw box and label for detected object
                color = color=colors(int(proposal[5]), True)
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 4)
                det_label = classes[int(proposal[5])]
                cv2.putText(image, det_label + ' ' + str(round(proposal[4] * 100, 1)) + ' %', (xmin, ymin - 7),
                            cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
        plt.axis("off")
        #plt.imshow(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #cv2.imwrite("result.jpg",image)
        #print(detect_objs)
        print("latency for image 640 x 640: ", end_time-start_time)
#        print(f'Done. ({time.time() - start_time:.3f}s)')
        #cv2.imshow("predicted image", image)
      
        #waits for user to press any key 
        #(this is necessary to avoid Python kernel form crashing)
        #cv2.waitKey(0) 
          
        #closing all open windows 
        #cv2.destroyAllWindows() 
    
    except:
#        print("###not able to process image")
#        print(input_image.shape)
        pass

if __name__ == '__main__':
    # Test API
    # img = cv2.imread('data/images/bus.jpg')
    #img = cv2.imread('C:/Users/pbarre/Downloads/cloud/public_cloud/aws/ssh_login_rel/onnx/yolov5_demo/zidane.jpg')
#    model_xml = "C:/Users/pbarre/Downloads/cloud/public_cloud/aws/ssh_login_rel/onnx/ov_yolo5/ov_yolo5/yolo5.xml"
    predictor = True
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='+', type=str, default='yolo5.xml', help='model.xml path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  
    args = parser.parse_args()
    model_xml = args.model[0]
    if os.path.isfile(args.source):
        # print ("File exist")
        img = cv2.imread(args.source)
        process_image(predictor, img, model_xml)
    else:
        # print ("Folder exist")
        path = args.source
        
        for img_path in os.listdir(args.source):
#            print(img_path)
            if img_path.endswith(".jpg") or img_path.endswith(".png") or img_path.endswith(".jpeg"):
                img_f_path = os.path.join(path, img_path)
                img = cv2.imread(img_f_path)
                
                process_image(predictor, img, model_xml)
