import time
import torch
import argparse
from torch.utils.data import DataLoader

from model.unet_resnet import Unet
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.train_and_eval import pixel_accuracy, mean_accuracy, mean_iou, frequency_weighted_iou


class LogColor:
    GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    RED = "\033[1;31m"
    RESET = "\033[0m"
    BLUE = "\033[1;34m"


def evaluate(model, val_loader, device, num_classes):
    model_eval = model.eval()
    model_eval = model_eval.to(device)

    total_pixel_acc = 0
    total_mean_acc = 0
    total_mean_iou = 0
    total_fw_iou = 0
    num_batches = len(val_loader)

    print(f"{LogColor.YELLOW}Start Evaluation on {len(val_loader.dataset)} images...{LogColor.RESET}")

    with torch.no_grad():
        for iteration, batch in enumerate(val_loader):
            imgs, pngs, labels = batch

            imgs = imgs.to(device)
            pngs = pngs.to(device)

            outputs = model_eval(imgs)

            pixel_acc = pixel_accuracy(outputs, pngs, num_classes)
            mean_acc = mean_accuracy(outputs, pngs, num_classes)
            mean_iou_value = mean_iou(outputs, pngs, num_classes)
            fw_iou = frequency_weighted_iou(outputs, pngs, num_classes)

            total_pixel_acc += pixel_acc
            total_mean_acc += mean_acc
            total_mean_iou += mean_iou_value
            total_fw_iou += fw_iou

            if iteration == 0:
                print(f"{'Data Count':^12} | {'Pixel Acc':^12} | {'Mean Acc':^12} | {'mIoU':^12} | {'FwIoU':^12}")
                print("-" * 75)

            current_pa = total_pixel_acc / (iteration + 1)
            current_ma = total_mean_acc / (iteration + 1)
            current_miou = total_mean_iou / (iteration + 1)
            current_fw = total_fw_iou / (iteration + 1)

            print(f"\r{iteration + 1}/{num_batches:<10} | {current_pa:.4f}       | {current_ma:.4f}       | "
                  f"{current_miou:.4f}       | {current_fw:.4f}", end="", flush=True)

    avg_pixel_acc = total_pixel_acc / num_batches
    avg_mean_acc = total_mean_acc / num_batches
    avg_mean_iou = total_mean_iou / num_batches
    avg_fw_iou = total_fw_iou / num_batches

    print(f"\n\n{LogColor.GREEN}" + "=" * 75)
    print(f"Final Evaluation Results:")
    print(f"Pixel Accuracy : {avg_pixel_acc:.4f}")
    print(f"Mean Accuracy  : {avg_mean_acc:.4f}")
    print(f"Mean IoU       : {avg_mean_iou:.4f}")
    print(f"Fw IoU         : {avg_fw_iou:.4f}")
    print("=" * 75 + f"{LogColor.RESET}\n")

    time.sleep(1)


def val(args):
    num_classes = args.num_classes + 1

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    input_shape = [480, 512]

    val_dataset = UnetDataset(args.data_path, input_shape, num_classes, augmentation=False, txt_name="test.txt")

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=unet_dataset_collate,
        sampler=None
    )

    model = Unet(num_classes=num_classes)

    print(f"Loading weights from: {args.weights}")
    weights_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(weights_dict)
    model.to(device)

    evaluate(model, val_loader, device, num_classes)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Semantic segmentation validation")

    parser.add_argument("--data-path", type=str, help="Dataset root directory")
    parser.add_argument("--weights", type=str, help="Path to model weights")
    parser.add_argument("--num-classes", type=int, help="Number of segmentation classes")
    parser.add_argument("--device", type=str, help="Device (cuda/cpu)")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    val(args)
