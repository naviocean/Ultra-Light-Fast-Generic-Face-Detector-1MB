import torch

from ..utils import box_utils
from .data_preprocessing import PredictionTransform
from ..utils.misc import Timer


class Predictor:
    def __init__(self, net, config, model_path=None, nms_method=None,
                 iou_threshold=0.3, filter_threshold=0.01, candidate_size=200, sigma=0.5, device=None, fuse=False):
        self.net = net
        self.transform = PredictionTransform(
            config.image_size, config.image_mean_test, config.image_std)
        self.iou_threshold = config.iou_threshold
        self.filter_threshold = filter_threshold
        self.candidate_size = candidate_size
        self.nms_method = nms_method
        self.config = config
        self.sigma = sigma
        if device:
            self.device = device
        else:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")

        if model_path is not None:
            self.net.load(model_path)
        self.net.to(self.device)
        self.net.eval()
        if fuse:
            self.net.fuse()
        self.timer = Timer()

    def predict(self, image, top_k=-1, prob_threshold=None, fuse=False):
        if fuse:
            self.net.fuse()
        cpu_device = torch.device("cpu")
        height, width, _ = image.shape
        image = self.transform(image)
        images = image.unsqueeze(0)
        images = images.to(self.device)
        with torch.no_grad():
            for i in range(1):
                self.timer.start()
                scores, locations = self.net.forward(images)

                print("Inference time: ", self.timer.end())
        locations = locations[0]
        scores = scores[0]
        if not prob_threshold:
            prob_threshold = self.filter_threshold
        # this version of nms is slower on GPU, so we move data to CPU.
        locations = locations.to(cpu_device)
        scores = scores.to(cpu_device)
        boxes = box_utils.convert_locations_to_boxes(
            locations, self.config.priors.to(
                self.device), self.config.center_variance, self.config.size_variance
        )
        boxes = box_utils.center_form_to_corner_form(boxes)
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, scores.size(1)):
            probs = scores[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.size(0) == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
            box_probs = box_utils.nms(box_probs, self.nms_method,
                                      score_threshold=prob_threshold,
                                      iou_threshold=self.iou_threshold,
                                      sigma=self.sigma,
                                      top_k=top_k,
                                      candidate_size=self.candidate_size)
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.size(0))
        if not picked_box_probs:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        picked_box_probs = torch.cat(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]

    def predict_wo_scale(self, image, top_k=-1, prob_threshold=None):
        cpu_device = torch.device("cpu")
        height, width, _ = image.shape
        image = self.transform(image)
        images = image.unsqueeze(0)
        images = images.to(self.device)
        with torch.no_grad():
            for i in range(1):
                self.timer.start()
                scores, locations = self.net.forward(images)

                print("Inference time: ", self.timer.end())
        locations = locations[0]
        scores = scores[0]
        if not prob_threshold:
            prob_threshold = self.filter_threshold
        # this version of nms is slower on GPU, so we move data to CPU.
        locations = locations.to(cpu_device)
        scores = scores.to(cpu_device)
        boxes = box_utils.convert_locations_to_boxes(
            locations, self.config.priors.to(
                self.device), self.config.center_variance, self.config.size_variance
        )
        boxes = box_utils.center_form_to_corner_form(boxes)
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, scores.size(1)):
            probs = scores[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.size(0) == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
            box_probs = box_utils.nms(box_probs, self.nms_method,
                                      score_threshold=prob_threshold,
                                      iou_threshold=self.iou_threshold,
                                      sigma=self.sigma,
                                      top_k=top_k,
                                      candidate_size=self.candidate_size)
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.size(0))
        if not picked_box_probs:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        picked_box_probs = torch.cat(picked_box_probs)
        return picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]
