import os
from pathlib import Path

def sync_segmentation_files():
    """
    检查SegmentationClass和JPEGImages文件夹中的文件，
    删除不匹配的文件，确保标注图片和原始图片一一对应
    """
    
    # 直接在代码中指定文件夹路径
    segment_class_dir = "/home/u241003661121/U-Net/FoodSeg103/VOC2012/SegmentationClass"
    jpeg_images_dir = "/home/u241003661121/U-Net/FoodSeg103/VOC2012/JPEGImages"
    
    # 检查文件夹是否存在
    if not os.path.exists(segment_class_dir):
        print(f"错误: SegmentationClass文件夹不存在: {segment_class_dir}")
        return
    
    if not os.path.exists(jpeg_images_dir):
        print(f"错误: JPEGImages文件夹不存在: {jpeg_images_dir}")
        return
    
    # 获取两个文件夹中的所有文件名（不含扩展名）
    segment_files = {}
    jpeg_files = {}
    
    # 收集SegmentationClass文件夹中的文件
    for file_path in Path(segment_class_dir).glob('*'):
        if file_path.is_file():
            segment_files[file_path.stem] = file_path.name
    
    # 收集JPEGImages文件夹中的文件
    for file_path in Path(jpeg_images_dir).glob('*'):
        if file_path.is_file():
            jpeg_files[file_path.stem] = file_path.name
    
    print(f"SegmentationClass文件夹文件数: {len(segment_files)}")
    print(f"JPEGImages文件夹文件数: {len(jpeg_files)}")
    
    # 找出共同的文件名
    common_files = set(segment_files.keys()) & set(jpeg_files.keys())
    print(f"匹配的文件数: {len(common_files)}")
    
    # 找出各自独有的文件
    only_in_segment = set(segment_files.keys()) - set(jpeg_files.keys())
    only_in_jpeg = set(jpeg_files.keys()) - set(segment_files.keys())
    
    print(f"SegmentationClass独有文件数: {len(only_in_segment)}")
    print(f"JPEGImages独有文件数: {len(only_in_jpeg)}")
    
    # 删除SegmentationClass中没有对应JPEGImages的文件
    deleted_segment_count = 0
    if only_in_segment:
        print("\n删除SegmentationClass中没有对应JPEGImages的文件:")
        for file_stem in only_in_segment:
            file_name = segment_files[file_stem]
            file_path = Path(segment_class_dir) / file_name
            print(f"  删除: {file_name}")
            file_path.unlink()
            deleted_segment_count += 1
    
    # 删除JPEGImages中没有对应SegmentationClass的文件
    deleted_jpeg_count = 0
    if only_in_jpeg:
        print("\n删除JPEGImages中没有对应SegmentationClass的文件:")
        for file_stem in only_in_jpeg:
            file_name = jpeg_files[file_stem]
            file_path = Path(jpeg_images_dir) / file_name
            print(f"  删除: {file_name}")
            file_path.unlink()
            deleted_jpeg_count += 1
    
    # 统计剩余文件
    remaining_segment = len([f for f in Path(segment_class_dir).glob('*') if f.is_file()])
    remaining_jpeg = len([f for f in Path(jpeg_images_dir).glob('*') if f.is_file()])
    
    print(f"\n清理完成:")
    print(f"  删除SegmentationClass文件: {deleted_segment_count}个")
    print(f"  删除JPEGImages文件: {deleted_jpeg_count}个")
    print(f"  剩余SegmentationClass文件: {remaining_segment}个")
    print(f"  剩余JPEGImages文件: {remaining_jpeg}个")
    
    if remaining_segment == remaining_jpeg:
        print("  ✓ 文件数量已同步")
    else:
        print("  ! 文件数量仍不一致，请检查")

if __name__ == "__main__":
    sync_segmentation_files()