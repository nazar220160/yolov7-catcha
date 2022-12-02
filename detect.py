from pathlib import Path

import cv2
import torch
from aiogram import Bot, Dispatcher, executor, types
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, TracedModel

bot = Bot(token='5321960286:AAEjHSALqXkeEXczGdslhrLt23gLOrxNo20')
dp = Dispatcher(bot)


@dp.message_handler(content_types='photo')
async def start(mes: types.Message):
    down = await mes.photo[1].download(chunk_size=200000)
    det = detect(source=down.name)
    await mes.reply(text=det['new'])


def detect(source, save_img=True, view_img=False, agnostic=True,
           save_txt=False, imgsz=640,
           project='runs/detect', name='exp',
           exist_ok=False, augment=False, conf_thres=0.25,
           iou_thres=0.45, classes=None, save_conf=False):
    result = []

    save_dir = Path(increment_path(Path(project) / name, exist_ok=exist_ok))
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
    half = device.type != 'cpu'
    if half:
        model.half()
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    names = model_two.module.names if hasattr(model_two, 'module') else model_two.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    model_two(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model_two.parameters())))

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        for i in range(3):
            model_two(img, augment=augment)
        with torch.no_grad():
            pred = model_two(img, augment=augment)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic)
        for i, det in enumerate(pred):
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                for *xyxy, conf, cls in reversed(det):
                    if save_txt:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    if save_img or view_img:  # Add bbox to image
                        old = str(names[int(cls)])
                        new = repl(old)
                        result.append({'number': new, 'pos': int(xyxy[0])})
                        label = f'{old} {new} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)

            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    newlist = sorted(result, key=lambda d: d['pos'])
    result = []
    for i in newlist:
        result.append(i['number'])

    text = ''.join(result)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        return {"new": text, 'dir': f"{save_dir}{s}/{source.split('/')[1]}"}


def repl(string: str):
    if string == '0':
        new = '1'
    elif string == '1':
        new = '0'
    elif string == '2':
        new = '5'
    elif string == '3':
        new = '6'
    elif string == '4':
        new = '8'
    elif string == '5':
        new = '9'
    elif string == '6':
        new = '3'
    elif string == '7':
        new = '4'
    elif string == '8':
        new = '7'
    elif string == '9':
        new = '2'
    else:
        new = None
    return new


if __name__ == '__main__':
    device = select_device('')
    model = attempt_load('best.pt', map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(640, s=stride)
    model_two = TracedModel(model, device, 640)
    executor.start_polling(dp)
