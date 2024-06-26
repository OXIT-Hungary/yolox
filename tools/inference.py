import argparse
import time
from pathlib import Path

import cv2
import hydra
import torch
from loguru import logger
from omegaconf import DictConfig
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import CUSTOM_CLASSES
from yolox.utils import get_model, postprocess, vis


def visual(output, img_info, cls_conf=0.35):
    ratio = img_info["ratio"]
    img = img_info["raw_img"]
    if output is None:
        return img
    output = output.cpu()

    bboxes = output[:, 0:4]
    bboxes /= ratio

    cls = output[:, 6]
    scores = output[:, 4] * output[:, 5]

    vis_res = vis(img, bboxes, scores, cls, cls_conf, CUSTOM_CLASSES)
    return vis_res


@hydra.main(config_path="../configs", config_name="yolox_m", version_base=None)
def main(cfg: DictConfig):

    model, decoder = get_model(cfg.model)
    preprocess = ValTransform(legacy=False)

    video_cap = cv2.VideoCapture(str(cfg.experiment.input))
    fps = video_cap.get(cv2.CAP_PROP_FPS)

    if cfg.experiment.save_result:
        width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        save_folder = Path("") / time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        video_writer = cv2.VideoWriter(save_folder / "out.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    else:
        cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)

    while True:
        has_frame, frame = video_cap.read()
        if not has_frame:
            break

        frame_info = {
            "id": 0,
            "height": frame.shape[0],
            "width": frame.shape[1],
            "raw_img": frame,
            "ratio": min(cfg.inference.output_size[0] / frame.shape[0], cfg.inference.output_size[1] / frame.shape[1]),
        }

        frame, _ = preprocess(frame, None, cfg.inference.output_size)
        frame = torch.from_numpy(frame).unsqueeze(0).float()

        frame = frame.to(cfg.model.device)
        if cfg.model.mixed_precision:
            frame = frame.half()  # to FP16

        model.eval()
        with torch.inference_mode():
            outputs = model(frame)

            if decoder is not None:
                outputs = decoder(outputs, dtype=outputs.type())

            outputs = postprocess(
                outputs,
                cfg.model.num_classes,
                cfg.inference.confidence,
                cfg.inference.nms_threshold,
                class_agnostic=True,
            )

        frame_result = visual(outputs[0], frame_info, cfg.inference.confidence)

        if cfg.experiment.save_result:
            video_writer.write(frame_result)
        else:
            cv2.imshow("yolox", frame_result)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q") or key == ord("Q"):
            break

    if cfg.experiment.save_result:
        video_writer.release()
    else:
        cv2.destroyAllWindows()

    video_cap.release()


if __name__ == "__main__":
    main()
