from PIL import Image
import numpy as np
import os
from natsort import natsorted
from glob import glob


def cropped_original_image_to_patches(original_image_path, patch_size, patches_save_root="/Users/yuziquan/patches"):
    original_image = Image.open(original_image_path).convert(mode="RGB")
    width, height = original_image.size

    # "/Users/yuziquan/Downloads/woman_LRBI_x4.png" => "woman_LRBI_x4"
    original_image_name = os.path.split(original_image_path)[-1].split(".")[0]

    # Case1
    if (width % patch_size) == 0 and (height % patch_size) == 0:
        rows_num = height // patch_size
        cols_num = width // patch_size

        for row_index in range(rows_num):
            for col_index in range(cols_num):
                # box = (left, upper, right, lower)
                cropped_patch = original_image.crop(box=(
                    col_index * patch_size, row_index * patch_size,
                    col_index * patch_size + patch_size, row_index * patch_size + patch_size))
                cropped_patch.save(
                    os.path.join(patches_save_root,
                                 f"{original_image_name}%{width}mul{height}%Case1%{rows_num}mul{cols_num}%{row_index}%{col_index}.png"))

    # Case2
    elif (width % patch_size) == 0 and (height % patch_size) != 0:
        rows_num = height // patch_size + 1
        cols_num = width // patch_size

        for row_index in range(rows_num):
            for col_index in range(cols_num):
                if row_index == (rows_num - 1):
                    # box = (left, upper, right, lower)
                    cropped_patch = original_image.crop(box=(
                        col_index * patch_size, height - patch_size,
                        col_index * patch_size + patch_size, height))
                    cropped_patch.save(
                        os.path.join(patches_save_root,
                                     f"{original_image_name}%{width}mul{height}%Case2%{rows_num}mul{cols_num}%{row_index}%{col_index}.png"))
                else:
                    # box = (left, upper, right, lower)
                    cropped_patch = original_image.crop(box=(
                        col_index * patch_size, row_index * patch_size,
                        col_index * patch_size + patch_size, row_index * patch_size + patch_size))
                    cropped_patch.save(
                        os.path.join(patches_save_root,
                                     f"{original_image_name}%{width}mul{height}%Case2%{rows_num}mul{cols_num}%{row_index}%{col_index}.png"))
    # Case3
    elif (width % patch_size) != 0 and (height % patch_size) == 0:
        rows_num = height // patch_size
        cols_num = width // patch_size + 1

        for row_index in range(rows_num):
            for col_index in range(cols_num):
                if col_index == (cols_num - 1):
                    # box = (left, upper, right, lower)
                    cropped_patch = original_image.crop(box=(
                        width - patch_size, row_index * patch_size,
                        width, row_index * patch_size + patch_size))
                    cropped_patch.save(
                        os.path.join(patches_save_root,
                                     f"{original_image_name}%{width}mul{height}%Case3%{rows_num}mul{cols_num}%{row_index}%{col_index}.png"))
                else:
                    # box = (left, upper, right, lower)
                    cropped_patch = original_image.crop(box=(
                        col_index * patch_size, row_index * patch_size,
                        col_index * patch_size + patch_size, row_index * patch_size + patch_size))
                    cropped_patch.save(
                        os.path.join(patches_save_root,
                                     f"{original_image_name}%{width}mul{height}%Case3%{rows_num}mul{cols_num}%{row_index}%{col_index}.png"))

    # Case4
    else:
        rows_num = height // patch_size + 1
        cols_num = width // patch_size + 1

        for row_index in range(rows_num):
            for col_index in range(cols_num):
                if row_index == (rows_num - 1) and col_index == (cols_num - 1):
                    # box = (left, upper, right, lower)
                    cropped_patch = original_image.crop(box=(
                        width - patch_size, height - patch_size,
                        width, height))
                    cropped_patch.save(
                        os.path.join(patches_save_root,
                                     f"{original_image_name}%{width}mul{height}%Case4%{rows_num}mul{cols_num}%{row_index}%{col_index}.png"))
                elif col_index == (cols_num - 1):
                    # box = (left, upper, right, lower)
                    cropped_patch = original_image.crop(box=(
                        width - patch_size, row_index * patch_size,
                        width, row_index * patch_size + patch_size))
                    cropped_patch.save(
                        os.path.join(patches_save_root,
                                     f"{original_image_name}%{width}mul{height}%Case4%{rows_num}mul{cols_num}%{row_index}%{col_index}.png"))
                elif row_index == (rows_num - 1):
                    # box = (left, upper, right, lower)
                    cropped_patch = original_image.crop(box=(
                        col_index * patch_size, height - patch_size,
                        col_index * patch_size + patch_size, height))
                    cropped_patch.save(
                        os.path.join(patches_save_root,
                                     f"{original_image_name}%{width}mul{height}%Case4%{rows_num}mul{cols_num}%{row_index}%{col_index}.png"))
                else:
                    # box = (left, upper, right, lower)
                    cropped_patch = original_image.crop(box=(
                        col_index * patch_size, row_index * patch_size,
                        col_index * patch_size + patch_size, row_index * patch_size + patch_size))
                    cropped_patch.save(
                        os.path.join(patches_save_root,
                                     f"{original_image_name}%{width}mul{height}%Case4%{rows_num}mul{cols_num}%{row_index}%{col_index}.png"))


def cropped_original_images_to_patches(original_images_root, patch_size, patches_save_root):
    original_images_paths_list = natsorted(glob(os.path.join(original_images_root, "*.png")))
    for original_image_path in original_images_paths_list:
        cropped_original_image_to_patches(original_image_path=original_image_path, patch_size=patch_size,
                                          patches_save_root=patches_save_root)
    print(f"Finished the cropping of {len(original_images_paths_list)} images!")


def recover_original_image_from_patches(recovered_image_name, which_case, width, height, rows_num, cols_num, patch_size,
                                        patches_root, recovered_image_save_root):
    recovered_image = Image.new("RGB", (width, height))
    if which_case == "Case1":
        for row_index in range(rows_num):
            for col_index in range(cols_num):
                patch = Image.open(os.path.join(patches_root,
                                                f"{recovered_image_name}%{width}mul{height}%Case1%{rows_num}mul{cols_num}%{row_index}%{col_index}.png"))
                recovered_image.paste(patch, box=(
                    col_index * patch_size, row_index * patch_size,
                    col_index * patch_size + patch_size, row_index * patch_size + patch_size))

    elif which_case == "Case2":
        for row_index in range(rows_num):
            for col_index in range(cols_num):
                if row_index == (rows_num - 1):
                    patch = Image.open(os.path.join(patches_root,
                                                    f"{recovered_image_name}%{width}mul{height}%Case2%{rows_num}mul{cols_num}%{row_index}%{col_index}.png"))
                    recovered_image.paste(patch, box=(
                        col_index * patch_size, height - patch_size,
                        col_index * patch_size + patch_size, height))

                else:
                    patch = Image.open(os.path.join(patches_root,
                                                    f"{recovered_image_name}%{width}mul{height}%Case2%{rows_num}mul{cols_num}%{row_index}%{col_index}.png"))
                    recovered_image.paste(patch, box=(
                        col_index * patch_size, row_index * patch_size,
                        col_index * patch_size + patch_size, row_index * patch_size + patch_size))


    elif which_case == "Case3":
        for row_index in range(rows_num):
            for col_index in range(cols_num):
                if col_index == (cols_num - 1):
                    patch = Image.open(os.path.join(patches_root,
                                                    f"{recovered_image_name}%{width}mul{height}%Case3%{rows_num}mul{cols_num}%{row_index}%{col_index}.png"))
                    recovered_image.paste(patch, box=(
                        width - patch_size, row_index * patch_size,
                        width, row_index * patch_size + patch_size))
                else:
                    patch = Image.open(os.path.join(patches_root,
                                                    f"{recovered_image_name}%{width}mul{height}%Case3%{rows_num}mul{cols_num}%{row_index}%{col_index}.png"))
                    recovered_image.paste(patch, box=(
                        col_index * patch_size, row_index * patch_size,
                        col_index * patch_size + patch_size, row_index * patch_size + patch_size))
    else:
        for row_index in range(rows_num):
            for col_index in range(cols_num):
                if row_index == (rows_num - 1) and col_index == (cols_num - 1):
                    patch = Image.open(os.path.join(patches_root,
                                                    f"{recovered_image_name}%{width}mul{height}%Case4%{rows_num}mul{cols_num}%{row_index}%{col_index}.png"))
                    recovered_image.paste(patch, box=(
                        width - patch_size, height - patch_size,
                        width, height))
                elif col_index == (cols_num - 1):
                    patch = Image.open(os.path.join(patches_root,
                                                    f"{recovered_image_name}%{width}mul{height}%Case4%{rows_num}mul{cols_num}%{row_index}%{col_index}.png"))
                    recovered_image.paste(patch, box=(
                        width - patch_size, row_index * patch_size,
                        width, row_index * patch_size + patch_size))
                elif row_index == (rows_num - 1):
                    patch = Image.open(os.path.join(patches_root,
                                                    f"{recovered_image_name}%{width}mul{height}%Case4%{rows_num}mul{cols_num}%{row_index}%{col_index}.png"))

                    recovered_image.paste(patch, box=(
                        col_index * patch_size, height - patch_size,
                        col_index * patch_size + patch_size, height))
                else:
                    patch = Image.open(os.path.join(patches_root,
                                                    f"{recovered_image_name}%{width}mul{height}%Case4%{rows_num}mul{cols_num}%{row_index}%{col_index}.png"))

                    recovered_image.paste(patch, box=(
                        col_index * patch_size, row_index * patch_size,
                        col_index * patch_size + patch_size, row_index * patch_size + patch_size))

    recovered_image.save(os.path.join(recovered_image_save_root, f"{recovered_image_name}.png"))


def recover_original_images_from_patches(patches_root, patch_size, recovered_images_save_root):
    patches_paths_list = natsorted(glob(os.path.join(patches_root, "*.png")))

    recovered_images_names_set = set()
    for patch_path in patches_paths_list:
        recovered_image_name = os.path.split(patch_path)[-1].split(".")[0].split("%")[0]
        recovered_images_names_set.add(recovered_image_name)

    # ['baby_HR_x4', 'bird_HR_x4', 'butterfly_HR_x4', 'head_HR_x4', 'woman_HR_x4']
    recovered_images_names_list = natsorted(list(recovered_images_names_set))

    for recovered_image_name in recovered_images_names_list:
        patches_paths_list_of_current_image = natsorted(
            glob(os.path.join(patches_root, f"{recovered_image_name}*.png")))

        meta_infor_list = os.path.split(patches_paths_list_of_current_image[0])[-1].split(".")[0].split("%")
        width = int(meta_infor_list[1].split("mul")[0])
        height = int(meta_infor_list[1].split("mul")[1])
        which_case = meta_infor_list[2]
        rows_num = int(meta_infor_list[3].split("mul")[0])
        cols_num = int(meta_infor_list[3].split("mul")[1])

        recover_original_image_from_patches(recovered_image_name=recovered_image_name, which_case=which_case,
                                            width=width, height=height, rows_num=rows_num, cols_num=cols_num,
                                            patch_size=patch_size, patches_root=patches_root,
                                            recovered_image_save_root=recovered_images_save_root)

    print(f"Finished the recovering of {len(recovered_images_names_list)} images!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of CBSD68')
    parser.add_argument('--patches_root', default='/home/amax/Jiangbo/log/QF_/results/patch/', help='patches_root')
    parser.add_argument('--save_root', default='/home/amax/Jiangbo/log/QF_/results/Whole/', help='save_root')
    parser.add_argument('--pa_s', type=int, default=200, help='patch size of training sample')
    args = parser.parse_args()
#     original_images_root = "/home/amax/DN_Dataset/BSD_DIV2K/BSDS500_DIV2K/train/GT/"
#     patches_save_root = "/home/amax/DN_Dataset/BSD_DIV2K/BSDS500_DIV2K/train/GT_patch/"
#    
#     cropped_original_images_to_patches(original_images_root=original_images_root, patch_size=256,
#                                        patches_save_root=patches_save_root)
#                                        
#     original_images_root = "/home/amax/DN_Dataset/BSD_DIV2K/BSDS500_DIV2K/train/Gussion/noise15/"
#     patches_save_root = "/home/amax/DN_Dataset/BSD_DIV2K/BSDS500_DIV2K/train/Gussion/noise15_patch/"
#    
#     cropped_original_images_to_patches(original_images_root=original_images_root, patch_size=256,
#                                        patches_save_root=patches_save_root)
                                        
#     original_images_root = "/home/amax/DN_Dataset/BSD_DIV2K/BSDS500_DIV2K/train/Gussion/noise50/"
#     patches_save_root = "/home/amax/DN_Dataset/BSD_DIV2K/BSDS500_DIV2K/train/Gussion/noise50_patch/"
#    
#     cropped_original_images_to_patches(original_images_root=original_images_root, patch_size=256,
#                                        patches_save_root=patches_save_root)
#                                        
#     original_images_root = "/home/amax/DN_Dataset/BSD_DIV2K/BSDS500_DIV2K/train/Gussion/noise25/"
#     patches_save_root = "/home/amax/DN_Dataset/BSD_DIV2K/BSDS500_DIV2K/train/Gussion/noise25_patch/"
#    
#     cropped_original_images_to_patches(original_images_root=original_images_root, patch_size=256,
#                                        patches_save_root=patches_save_root)

    patches_root = args.patches_root
    recovered_images_save_root = args.save_root

    recover_original_images_from_patches(patches_root=patches_root, patch_size=args.pa_s,
                                         recovered_images_save_root=recovered_images_save_root)



