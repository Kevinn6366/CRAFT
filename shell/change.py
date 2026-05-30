import os
from pathlib import Path


def sync_segmentation_files():
    segment_class_dir = "/home/u241003661121/U-Net/FoodSeg103/VOC2012/SegmentationClass"
    jpeg_images_dir = "/home/u241003661121/U-Net/FoodSeg103/VOC2012/JPEGImages"

    if not os.path.exists(segment_class_dir):
        print(f"Error: SegmentationClass directory not found: {segment_class_dir}")
        return

    if not os.path.exists(jpeg_images_dir):
        print(f"Error: JPEGImages directory not found: {jpeg_images_dir}")
        return

    segment_files = {}
    jpeg_files = {}

    for file_path in Path(segment_class_dir).glob('*'):
        if file_path.is_file():
            segment_files[file_path.stem] = file_path.name

    for file_path in Path(jpeg_images_dir).glob('*'):
        if file_path.is_file():
            jpeg_files[file_path.stem] = file_path.name

    print(f"SegmentationClass files: {len(segment_files)}")
    print(f"JPEGImages files: {len(jpeg_files)}")

    common_files = set(segment_files.keys()) & set(jpeg_files.keys())
    print(f"Matched files: {len(common_files)}")

    only_in_segment = set(segment_files.keys()) - set(jpeg_files.keys())
    only_in_jpeg = set(jpeg_files.keys()) - set(segment_files.keys())

    print(f"SegmentationClass-only files: {len(only_in_segment)}")
    print(f"JPEGImages-only files: {len(only_in_jpeg)}")

    deleted_segment_count = 0
    if only_in_segment:
        print("\nRemoving SegmentationClass files without matching JPEG:")
        for file_stem in only_in_segment:
            file_name = segment_files[file_stem]
            file_path = Path(segment_class_dir) / file_name
            print(f"  Removing: {file_name}")
            file_path.unlink()
            deleted_segment_count += 1

    deleted_jpeg_count = 0
    if only_in_jpeg:
        print("\nRemoving JPEG files without matching SegmentationClass:")
        for file_stem in only_in_jpeg:
            file_name = jpeg_files[file_stem]
            file_path = Path(jpeg_images_dir) / file_name
            print(f"  Removing: {file_name}")
            file_path.unlink()
            deleted_jpeg_count += 1

    remaining_segment = len([f for f in Path(segment_class_dir).glob('*') if f.is_file()])
    remaining_jpeg = len([f for f in Path(jpeg_images_dir).glob('*') if f.is_file()])

    print(f"\nCleanup complete:")
    print(f"  Removed SegmentationClass: {deleted_segment_count}")
    print(f"  Removed JPEGImages: {deleted_jpeg_count}")
    print(f"  Remaining SegmentationClass: {remaining_segment}")
    print(f"  Remaining JPEGImages: {remaining_jpeg}")

    if remaining_segment == remaining_jpeg:
        print("  File counts synchronized")
    else:
        print("  Warning: file counts still differ")


if __name__ == "__main__":
    sync_segmentation_files()
