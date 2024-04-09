import random

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from osgeo import gdal
import os

#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size):
    iw, ih  = image.size
    w, h    = size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image   = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image, nw, nh
    
#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#---------------------------------------------------#
#   设置种子
#---------------------------------------------------#
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#---------------------------------------------------#
#   设置Dataloader的种子
#---------------------------------------------------#
def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def preprocess_input(image):
    image -= np.array([123.675, 116.28, 103.53], np.float32)
    image /= np.array([58.395, 57.12, 57.375], np.float32)
    return image

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def download_weights(phi, model_dir="./model_data"):
    import os
    from torch.hub import load_state_dict_from_url
    
    download_urls = {
        'b0' : "https://github.com/bubbliiiing/segformer-pytorch/releases/download/v1.0/segformer_b0_backbone_weights.pth",
        'b1' : "https://github.com/bubbliiiing/segformer-pytorch/releases/download/v1.0/segformer_b1_backbone_weights.pth",
        'b2' : "https://github.com/bubbliiiing/segformer-pytorch/releases/download/v1.0/segformer_b2_backbone_weights.pth",
        'b3' : "https://github.com/bubbliiiing/segformer-pytorch/releases/download/v1.0/segformer_b3_backbone_weights.pth",
        'b4' : "https://github.com/bubbliiiing/segformer-pytorch/releases/download/v1.0/segformer_b4_backbone_weights.pth",
        'b5' : "https://github.com/bubbliiiing/segformer-pytorch/releases/download/v1.0/segformer_b5_backbone_weights.pth",
    }
    url = download_urls[phi]
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)


def get_legend():
    colors = [(0, 0, 0), (0.5, 0, 0), (0, 0.5, 0), (0.5, 0.5, 0), (0, 0, 0.5), (0.5, 0, 0.5), (0, 0.5, 0.5),
              (0.5, 0.5, 0.5)]
    labels = ["others", "greenhouse", "maize", "no-crop", "squash", "sunflowers", "tomato", "wheat"]
    save_path = r'E:\jupyter_save\deeplearning\segformer-pytorch-master\segformer-pytorch-master\img\legend.jpg'
    patches = []
    for color, label in zip(colors, labels):
        patches.append(plt.Rectangle((0, 0), 1, 1, fc=color))

    fig, ax = plt.subplots(figsize=(1, 8))
    ax.legend(patches, labels, loc='center', frameon=False)
    ax.axis('off')

    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def replace_inf(tensor):
    tensor[tensor == float("-inf")] = 0
    tensor[tensor == float("inf")] = 0
    return tensor


from osgeo import gdal
import numpy as np


def replace_pixel_value(input_path, output_path, original_value, new_value):
    '''
    有一张tif灰度图，其中的值为0、1、5，将其中的值替换为0，1，2，另存为tif
    '''
    # 打开TIFF文件
    dataset = gdal.Open(input_path, gdal.GA_ReadOnly)

    # 读取图像数据
    band = dataset.GetRasterBand(1)
    data = band.ReadAsArray()

    # 将特定值的像素替换为新值
    data[data == original_value] = new_value

    # 创建新的TIFF文件
    driver = gdal.GetDriverByName("GTiff")
    output_dataset = driver.Create(output_path, dataset.RasterXSize, dataset.RasterYSize, 1, band.DataType)

    # 将修改后的数据写入新的TIFF文件
    output_band = output_dataset.GetRasterBand(1)
    output_band.WriteArray(data)

    # 复制地理信息和投影信息
    output_dataset.SetGeoTransform(dataset.GetGeoTransform())
    output_dataset.SetProjection(dataset.GetProjection())

    # 关闭文件
    dataset = None
    output_dataset = None

    print(f"已将值为 {original_value} 的像素替换为 {new_value}，并保存为 {output_path}")


def stack_bands_to_oneimage():
    # 有202005-4326.tif到202010-4326.tif共6张图片，代表2020年5到10月的图像，每张图像有12个band，将这6张图片堆叠成12*6个band的一张tif图
    # 输入文件列表（6个TIF文件）
    input_files = [
        r'E:\jupyter_save\deeplearning\segformer-pytorch-master\segformer-pytorch-master\img\amerciamonthimage\202005-4326.tif',
        r'E:\jupyter_save\deeplearning\segformer-pytorch-master\segformer-pytorch-master\img\amerciamonthimage\202006-4326.tif',
        r'E:\jupyter_save\deeplearning\segformer-pytorch-master\segformer-pytorch-master\img\amerciamonthimage\202007-4326.tif',
        r'E:\jupyter_save\deeplearning\segformer-pytorch-master\segformer-pytorch-master\img\amerciamonthimage\202008-4326.tif',
        r'E:\jupyter_save\deeplearning\segformer-pytorch-master\segformer-pytorch-master\img\amerciamonthimage\202009-4326.tif',
        r'E:\jupyter_save\deeplearning\segformer-pytorch-master\segformer-pytorch-master\img\amerciamonthimage\202010-4326.tif'
    ]

    # 创建一个空列表以保存各图像的数据集
    datasets = []

    for input_file in input_files:
        dataset = gdal.Open(input_file)
        if dataset is not None:
            datasets.append(dataset)
        else:
            print(f"无法打开文件: {input_file}")

    if not datasets:
        print("无法打开任何输入文件。")
    else:
        # 获取输入图像的尺寸
        width = datasets[0].RasterXSize
        height = datasets[0].RasterYSize

        # 创建输出文件夹，如果不存在
        output_folder = r'E:\jupyter_save\deeplearning\segformer-pytorch-master\segformer-pytorch-master\img\amerciamonthimage'  # 替换为您的目标文件夹路径
        os.makedirs(output_folder, exist_ok=True)


        output_file = os.path.join(output_folder, 'stacked_image.tif')
        driver = gdal.GetDriverByName('GTiff')
        output_dataset = driver.Create(output_file, width, height, 12 * len(datasets), gdal.GDT_Float32)

        # 复制输入图像的各波段到输出数据集
        for i, dataset in enumerate(datasets):
            for band_num in range(1, 13):
                band = dataset.GetRasterBand(band_num)
                output_band = output_dataset.GetRasterBand(i * 12 + band_num)
                output_band.WriteArray(band.ReadAsArray())


def crop_and_replace_nans_infs(input_path, output_folder, crop_size=128, overlap=0.1):
    '''
    按照重叠率裁剪多波段图像
    :param input_path:
    :param output_folder:
    :param crop_size:
    :param overlap:
    :return:
    '''
    # 打开输入的TIFF图像
    dataset = gdal.Open(input_path)

    # 获取图像的宽度和高度
    width = dataset.RasterXSize
    height = dataset.RasterYSize

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    a = 0

    for i in range(0, width, int(crop_size * (1 - overlap))):
        for j in range(0, height, int(crop_size * (1 - overlap))):
            # 读取图像数据
            image_data = dataset.ReadAsArray(i, j, crop_size, crop_size)

            # 检查NaN和Inf值并替换为0
            image_data = np.nan_to_num(image_data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

            # 生成输出文件名，这里假设原始文件名是"input.tif"
            # output_filename = f"{output_folder}/{i}-{j}.tif"
            output_filename = f"{output_folder}/{a}.tif"
            # 创建一个新的TIFF文件
            driver = gdal.GetDriverByName("GTiff")
            output_dataset = driver.Create(output_filename, crop_size, crop_size, dataset.RasterCount, gdal.GDT_Float32)

            # 设置地理信息和投影信息
            output_dataset.SetGeoTransform((i, dataset.GetGeoTransform()[1], 0, j, 0, dataset.GetGeoTransform()[5]))
            output_dataset.SetProjection(dataset.GetProjection())

            # 将裁剪后的图像数据写入新文件
            for band_index in range(dataset.RasterCount):
                output_dataset.GetRasterBand(band_index + 1).WriteArray(image_data[band_index, :, :])

            # 关闭输出文件
            output_dataset = None
            a = a+1

    # 关闭输入文件
    dataset = None


def show_image():
    #从72个band中得到某个月的RGB
    from osgeo import gdal
    import numpy as np
    import matplotlib.pyplot as plt

    # 输入TIFF文件路径
    input_tiff = r"E:\jupyter_save\deeplearning\segformer-pytorch-master\segformer-pytorch-master\img\image_108.tif"

    # 打开TIFF文件
    dataset = gdal.Open(input_tiff, gdal.GA_ReadOnly)

    if dataset is None:
        print("无法打开输入TIFF文件")
    else:
        # 读取RGB波段数据
        red_band = dataset.GetRasterBand(28).ReadAsArray()*5
        green_band = dataset.GetRasterBand(27).ReadAsArray()*5
        blue_band = dataset.GetRasterBand(26).ReadAsArray()*5

        # 将RGB波段堆叠成RGB图像
        rgb_image = np.dstack((red_band, green_band, blue_band))

        # 显示RGB图像
        plt.imshow(rgb_image)
        plt.axis('off')  # 关闭坐标轴
        plt.title("RGB Image")
        plt.show()

        dataset = None  # 释放GDAL数据集

    print("RGB图像展示完成")


from osgeo import gdal
import glob

def merge_tifs(folder_path, reference_file, output_file, output_file1):
    tif_files = os.listdir(folder_path)

    # 获取第一个tif文件的信息
    first_tif_path = os.path.join(folder_path,tif_files[0])
    first_tif = gdal.Open(first_tif_path)
    geotransform = first_tif.GetGeoTransform()
    data_type = first_tif.GetRasterBand(1).DataType
    num_bands = first_tif.RasterCount

    # 获取参考文件的地理信息
    reference_dataset = gdal.Open(reference_file)
    reference_projection = reference_dataset.GetProjection()

    # 计算新tif文件的最大横纵坐标
    max_x = max_y = 0
    for tif_file in tif_files:
        x = int(tif_file.split('.')[0].split('_')[0])+128
        if max_x < x:
            max_x = x
        y = int(tif_file.split('.')[0].split('_')[1])+128
        if max_y < y:
            max_y = y

    # 创建新的tif文件
    driver = gdal.GetDriverByName('GTiff')
    out_dataset = driver.Create(output_file, max_x, max_y, num_bands, data_type)

    # 设置地理信息和投影信息
    out_dataset.SetGeoTransform(geotransform)
    out_dataset.SetProjection(reference_projection)

    # 逐个复制tif文件的数据到新文件中
    for tif_file in tqdm(tif_files):
        x, y = map(int, tif_file.split('.')[0].split('_')[:2])
        tif_path = os.path.join(folder_path, tif_file)
        tif_data = gdal.Open(tif_path).ReadAsArray()
        for i in range(num_bands):
            output_band = out_dataset.GetRasterBand(i + 1)
            output_band.WriteArray(tif_data[i, :, :], x, y)

    # 裁剪到参考tif文件的宽和高
    reference_cols = reference_dataset.RasterXSize
    reference_rows = reference_dataset.RasterYSize
    gdal.Translate(output_file1, out_dataset, format='GTiff', srcWin=[0, 0, reference_cols, reference_rows])

    # 关闭数据集
    out_dataset = None
    print("图像已拼接并保存")


def label_to_color(target_file):
    '''我有一张tif灰度图，其中的值为0，1，5，帮我写一个python代码，将其显示为彩色图，
    颜色分别为[0, 0, 0], [128, 0, 0], [0, 128, 0]
    '''
    from PIL import Image
    import numpy as np

    # 打开灰度图
    gray_image = Image.open(target_file)  # 替换为你的灰度图路径

    # 将图像转换为NumPy数组以进行处理
    gray_array = np.array(gray_image)

    # 创建一个彩色映射表，将0映射为[0, 0, 0]，1映射为[128, 0, 0]，5映射为[0, 128, 0]
    color_mapping = {
        0: [0, 0, 0],
        1: [128, 0, 0],
        5: [0, 128, 0]
    }

    # 创建一个彩色图像
    height, width = gray_array.shape
    color_image = np.zeros((height, width, 3), dtype=np.uint8)

    for value, color in color_mapping.items():
        color_image[np.where(gray_array == value)] = color

    # 将NumPy数组转换回PIL图像
    color_image = Image.fromarray(color_image)

    # 显示图像
    color_image.show()

    # 保存彩色图像
    color_image.save(r"E:\jupyter_save\deeplearning\segformer-pytorch-master\segformer-pytorch-master\1_segformer\VOCdevkit\VOC2007\label_color_image.tif")  # 保存为彩色图像


def comparasion(image1, image2, target_file):
    '''
    比较生成的CDL和标签的差异
    '''
    from PIL import Image

    # 确保两张图像的大小相同
    if image1.size != image2.size:
        print("两张图像的尺寸不一致")
    else:
        width, height = image1.size

        # 创建一个新图像，用于显示比较结果
        diff_image = Image.new("RGB", (width, height))

        # 比较两张图像的每个像素
        for x in range(width):
            for y in range(height):
                pixel1 = image1.getpixel((x, y))
                pixel2 = image2.getpixel((x, y))

                # 如果像素值一致，显示为绿色，否则显示为红色
                if pixel1 == pixel2:
                    diff_image.putpixel((x, y), (0, 255, 0))  # 绿色
                else:
                    diff_image.putpixel((x, y), (255, 0, 0))  # 红色

        # 保存比较结果为TIF图
        diff_image.save(target_file)
        print("比较结果已保存为comparison_result.tif")


from PIL import Image


def process_image(image_path):
    # 把背景变白色
    # 打开图像
    image = Image.open(image_path)

    # 获取图像的像素值并转换为列表
    pixels = list(image.getdata())

    # 根据条件处理像素值
    processed_pixels = []
    for pixel in pixels:
        # 检查RGB通道的值是否全为0
        if pixel[0] + pixel[1] + pixel[2] < 50:
            processed_pixels.append((255, 255, 255))  # 设置为[255, 255, 255]
        else:
            processed_pixels.append(pixel)  # 保持原值不变

    # 将处理后的像素值重新写入图像
    image.putdata(processed_pixels)

    # 另存为图像
    new_image_path = r"E:\jupyter_save\hetao_classification\crop_land_classification\code\images\rf1.jpg"  # 修改为你想要保存的文件名和格式
    image.save(new_image_path)

    print("图像处理完成，并保存为:", new_image_path)


def check_inf_nan_tif(folder_path):
    # 获取文件夹中所有TIF文件
    tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    n = 0

    for index, tif_file in tqdm(enumerate(tif_files), desc="Accessing List", unit=" /15800 or 31720 tif"):
        tif_path = os.path.join(folder_path, tif_file)

        # 打开TIF文件
        dataset = gdal.Open(tif_path)

        if dataset is not None:
            # 读取图像数据
            img_data = dataset.ReadAsArray()

            # 检查是否存在 inf 或 nan
            has_inf = np.isinf(img_data).any()
            has_nan = np.isnan(img_data).any()

            if has_inf or has_nan:
                print(f"{tif_file} 包含 'inf' 或 'nan' 值.")
                n = n+1

            # 关闭数据集
            dataset = None

        else:
            print(f"无法打开文件：{tif_file}")

    if n == 0:
        print('无包含inf和nan的图像')


from osgeo import gdal
import numpy as np

def create_colored_image(input_path, output_path):
    '''
    我有一张tif灰度图，其中的值为0-6，帮我写一个python代码，只用GDAL和np，将其转换为彩色图，
    颜色分别是'ff0000','109030', 'ff7f0e', 'ffff00',  '2040ff','805050','ffffff'，然后另存为tif图
    :param input_path:
    :param output_path:
    :return:
    '''
    # 打开原始TIFF文件
    dataset = gdal.Open(input_path)

    if dataset is None:
        print("无法打开TIFF文件。")
        return

    # 读取单通道的数据
    band = dataset.GetRasterBand(1)
    data = band.ReadAsArray()

    # 定义颜色映射
    color_map = np.array([
        [0, 0, 0],
        [16, 144, 48],
        [255, 0, 0],    # 'ff0000'
          # '109030'
        # [255, 127, 14], # 'ff7f0e'
        [255, 255, 0],  # 'ffff00'
        # [32, 64, 255],  # '2040ff'
        # [128, 80, 80],  # '805050'
        # [255, 255, 255]
         # 'ffffff'
    ], dtype=np.uint8)

    # 使用颜色映射将灰度图转换为彩色图
    colored_data = color_map[data]

    # 创建输出TIFF文件
    driver = gdal.GetDriverByName('GTiff')
    output_dataset = driver.Create(output_path, dataset.RasterXSize, dataset.RasterYSize, 3, gdal.GDT_Byte)

    # 将地理信息写入输出TIFF文件
    output_dataset.SetGeoTransform(dataset.GetGeoTransform())
    output_dataset.SetProjection(dataset.GetProjection())

    # 将彩色数据写入输出TIFF文件的三个波段
    for i in range(3):
        output_band = output_dataset.GetRasterBand(i + 1)
        output_band.WriteArray(colored_data[:, :, i])

    # 关闭数据集
    dataset = None
    output_dataset = None


from osgeo import gdal
import os


def crop_tiff_to_png(input_path, output_folder, output_format='PNG', tile_size=128, overlap=0.1):
    '''
    按照重叠率剪裁单波段tif图,另存为png
    '''
    input_dataset = gdal.Open(input_path)

    if input_dataset is None:
        print("无法打开输入文件")
        return

    width = input_dataset.RasterXSize
    height = input_dataset.RasterYSize

    for i in range(0, width, int(tile_size * (1 - overlap))):
        for j in range(0, height, int(tile_size * (1 - overlap))):
            x_offset = i
            y_offset = j

            if x_offset + tile_size > width:
                x_offset = width - tile_size
            if y_offset + tile_size > height:
                y_offset = height - tile_size

            output_filename = f"{i}_{j}.png"  # 根据位置创建输出文件名
            output_path = os.path.join(output_folder, output_filename)

            gdal.Translate(output_path, input_dataset, srcWin=[x_offset, y_offset, tile_size, tile_size],
                           format=output_format)
            # print(f"已创建{output_filename}")
            xml_path = os.path.join(output_folder, output_filename + '.aux.xml')
            os.remove(xml_path)

    input_dataset = None


from osgeo import gdal
import numpy as np
import os
from tqdm import tqdm


def crop_and_replace_nan_inf(input_image_path, output_folder, tile_size=128, overlap=0.1, remove_inf_nan=True):
    '''
    我有一张单波段tif图，帮我写一个python程序，用gdal读取他，数据类型不变，
    然后按照0.1的重叠率裁剪为[128,128]的小图像，检查是否有inf和nan
    ,有的话替换为0，另存为tif图像到另一个文件夹，重命名为小图像在原图中的位置
    '''
    # 打开图像文件
    dataset = gdal.Open(input_image_path)

    if dataset is None:
        print("无法打开图像文件")
        return

    band = dataset.GetRasterBand(1)  # 获取第一个波段
    data = band.ReadAsArray()  # 读取图像数据

    width = dataset.RasterXSize  # 获取图像宽度
    height = dataset.RasterYSize  # 获取图像高度

    # 计算裁剪的步长（0.1重叠率）
    a = 0
    for y in range(0, height, int(tile_size * (1 - overlap))):
        for x in range(0, width, int(tile_size * (1 - overlap))):
            # 裁剪小图像
            small_img = data[y:y+128, x:x+128]

            # 检查NaN和inf并替换为0
            small_img[np.isnan(small_img)] = 0
            small_img[np.isinf(small_img)] = 0
            if np.all(small_img == 0) and remove_inf_nan:
                continue

            # 创建新的tif文件名并保存小图像
            driver = gdal.GetDriverByName("GTiff")
            output_filename = f"{output_folder}/{x}_{y}.tif"
            new_dataset = driver.Create(output_filename, 128, 128, 1, band.DataType)

            if new_dataset is None:
                print(f"无法创建文件 {output_filename}")
                continue

            new_dataset.GetRasterBand(1).WriteArray(small_img)
            new_dataset.FlushCache()  # 刷新缓存

            del new_dataset  # 释放资源
            a = a+1

    del dataset  # 释放资源


def create_folder(folder_path):
    try:
        # 使用os.makedirs()创建文件夹，如果文件夹已存在，则会抛出异常
        os.makedirs(folder_path)
        print(f"文件夹 '{folder_path}' 创建成功！")
    except FileExistsError:
        pass


import concurrent.futures  # 用于并行处理


def read_tiff_as_array(tiff_path):
    dataset = gdal.Open(tiff_path, gdal.GA_ReadOnly)
    band = dataset.GetRasterBand(1)
    data = band.ReadAsArray()
    dataset = None
    return data


def read_files_parallel(file_paths):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 并行读取文件
        result = list(executor.map(read_tiff_as_array, file_paths))
    return result


def stack_images(input_folder, output_folder, bands):
    '''
    我有一个文件夹，里面包含72个子文件夹，所有子文件夹里包含数量相同名字相同的tif图片，
    帮我写一个python程序，用gdal挨个读取第一个子文件夹里的tif图，并在72个子文
    件夹里寻找具有相同名称的tif图，将所有名称相同的tif图堆叠在一起，并另存到另一个文
    件夹，堆叠后的tif图命名与第一个子文件夹里的tif图名称相同
    '''
    # 获取所有子文件夹

    subfolders = [f.path for f in os.scandir(input_folder) if f.is_dir()]

    # 获取第一个子文件夹的所有tif图像
    first_subfolder = subfolders[0]
    tif_files_first_subfolder = [f.path.split('.')[0].split('\\')[-1] for f in os.scandir(first_subfolder) if f.is_file() and f.name.endswith('.tif')]
    lenth  = len(tif_files_first_subfolder)

    for index, tif_file_first_subfolder in tqdm(enumerate(tif_files_first_subfolder), desc="Accessing List",
                                                unit=" /tif total " + str(lenth) + ' tifs', miniters=1,
                                                dynamic_ncols=True):
        file_list = [os.path.join(os.path.join(input_folder, band), tif_file_first_subfolder) + ".tif" for band in bands]
        jpg_arrays = read_files_parallel(file_list)
        # 对读取的数组进行堆叠
        stacked_data = np.array(jpg_arrays)

        # 创建新的tif文件名并保存堆叠后的图像
        output_filename = os.path.join(output_folder, f"{tif_file_first_subfolder}.tif")
        driver = gdal.GetDriverByName("GTiff")

        # 获取图像的元数据信息
        first = file_list[0]
        first_tif = gdal.Open(first)
        geo_transform = first_tif.GetGeoTransform()
        projection = first_tif.GetProjection()
        data_type = first_tif.GetRasterBand(1).DataType
        width = first_tif.RasterXSize
        height = first_tif.RasterYSize

        new_dataset = driver.Create(output_filename, width, height, len(stacked_data), data_type)

        new_dataset.SetGeoTransform(geo_transform)
        new_dataset.SetProjection(projection)
        # print(len(stacked_data))

        for idx in range(len(stacked_data)):
            new_dataset.GetRasterBand(idx + 1).WriteArray(stacked_data[idx])

        new_dataset.FlushCache()  # 刷新缓存
        del new_dataset  # 释放资源

        # print(f"图像 {file_name_no_extension}.tif 创建成功！")


def separate_bands(input_tif,output_path):
    '''
    分离多波段文件
    '''
    # 打开多波段 TIFF 文件
    dataset = gdal.Open(input_tif)

    if dataset is None:
        print("无法打开文件：" + input_tif)
        return

    num_bands = dataset.RasterCount
    name = input_tif.split('/')[-1][0:6]
    print(name)
    names = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12']

    for i in range(1, num_bands + 1):
        band = dataset.GetRasterBand(i)
        output_tif = os.path.join(output_path, name + f"_{names[i-1]}.tif")

        # 创建输出文件，保持原始数据类型
        driver = gdal.GetDriverByName("GTiff")
        new_dataset = driver.Create(output_tif, dataset.RasterXSize, dataset.RasterYSize, 1, band.DataType)

        # 将原始波段数据写入新文件
        new_dataset.GetRasterBand(1).WriteArray(band.ReadAsArray())

        # 将原始地理信息（坐标系统、仿射变换等）复制到新文件
        new_dataset.SetProjection(dataset.GetProjection())
        new_dataset.SetGeoTransform(dataset.GetGeoTransform())

        new_dataset = None

        print("已保存文件：" + output_tif)

    dataset = None


import os
import glob
import numpy as np
from osgeo import gdal, gdal_array
from collections import defaultdict


def process_tifs(folder_path, output_folder):
    '''
    将多个单波段tif横向拼接
    我有一个文件夹，里面是单波段tif文件，文件命名规则是‘开始时间_截至时间_卫星_波段-起始纵坐标-起始横坐标.tif'，
    帮我写一个python程序，用gdal，遍历所有tif，读取‘开始时间_截至时间_卫星_波段’相同的文件的宽和高，
    检查他们是否有nan或inf，有的话替换为0，将他们的宽加起来作为新的宽，新建tif，以第一个tif的高和新的宽作为
    新建tif的高和宽，并将‘开始时间_截至时间_卫星_波段’相同的文件写入到新tif，并以开始时间_波段命名新的tif文件
    '''
    # 获取文件夹中所有tif文件
    tif_files = [file for file in os.listdir(folder_path) if file.endswith('.tif')]

    # 存储每个 '开始时间_截至时间_卫星_波段' 对应的文件列表
    grouped_files = defaultdict(list)

    # 遍历所有tif文件
    for file in tif_files:
        # 解析文件名获取开始时间、截至时间、卫星和波段
        parts = file.split('-')[0]
        key = parts  # 开始时间_截至时间_卫星_波段作为键值
        grouped_files[key].append(os.path.join(folder_path, file))

    # 遍历每个 '开始时间_截至时间_卫星_波段' 组
    for key, file_list in grouped_files.items():

        start_time, endtime, satellite, band = key.split('_')[:4]
        start_time = start_time[0:8]
        band = band.split('-')[0]
        output_filename = f'{start_time}_{band}.tif'  # 输出文件名以开始时间和波段命名
        output_path = os.path.join(output_folder, output_filename)

        # 读取第一个tif文件获取高度和宽度
        first_tif = gdal.Open(file_list[0])
        height = first_tif.RasterYSize
        width = sum([gdal.Open(file_path).RasterXSize for file_path in file_list])

        # 创建新的tif文件
        driver = gdal.GetDriverByName('GTiff')
        output_tif = driver.Create(output_path, width, height, 1, gdal.GDT_Float32)

        # 设置地理信息和投影信息
        output_tif.SetGeoTransform(first_tif.GetGeoTransform())
        output_tif.SetProjection(first_tif.GetProjection())

        # 初始化一个数组用于存储新tif的数据
        mosaic_data = np.zeros((height, width), dtype=np.float32)

        # 逐个读取并拷贝数据到新的数组中
        current_width = 0
        for file_path in file_list:
            tif = gdal.Open(file_path)
            data = tif.GetRasterBand(1).ReadAsArray()

            # 替换nan和inf为0
            data[np.isnan(data)] = 0
            data[np.isinf(data)] = 0

            # 将数据复制到新的数组中
            mosaic_data[:, current_width:current_width+tif.RasterXSize] = data

            # 更新当前宽度
            current_width += tif.RasterXSize

        # 将新数组写入新tif文件
        output_tif.GetRasterBand(1).WriteArray(mosaic_data)

        # 关闭文件
        output_tif = None
        first_tif = None

    print("处理完成")

if __name__ == "__main__":
    pass
    # 将多个单波段tif横向拼接
    # input_folder_path = r'D:\下载\drive-download-20231219T101348Z-001'
    # output_folder_path = r'D:\下载\drive-download-20231219T101348Z-001'
    # process_tifs(input_folder_path, output_folder_path)


    # # 分离多波段图像
    # input_multi_band_tif = 'J:/research/GEE/hetao_classification/IOWAofAMERICA/amerciamonthimage/202010-4326.tif'
    # output_path = r'J:\research\GEE\hetao_classification\IOWAofAMERICA\amerciamonthimage_bands'
    # # 调用函数将每个波段另存为单独的 TIFF 文件
    # separate_bands(input_multi_band_tif, output_path)

    # 随机森林
    # folder_path = r'E:\jupyter_save\deeplearning\segformer-pytorch-master\segformer-pytorch-master\6_randomforest\logs_2020HT_EVI_RF'
    # files = os.listdir(folder_path)
    # # 遍历文件列表,裁剪单波段文件为128*128，另存到文件夹
    # for index, file_name in tqdm(enumerate(files), desc="Accessing List", unit="element"):
    #     file_path = os.path.join(folder_path, file_name)
    #     file_name1 = file_name.split('.')[0].split('_')[0] + '_' + file_name.split('.')[0].split('_')[1]
    #     if os.path.isfile(file_path):
    #         input_file = os.path.abspath(file_path)
    #         # 按照重叠率裁剪单波段tif图为128*128的图像
    #         output_folder = r'E:\jupyter_save\deeplearning\segformer-pytorch-master\segformer-pytorch-master\6_randomforest\logs_2020HT_EVI_RF\miou_out\detection-results'
    #         tif_files = [file for file in files if file.endswith(".tif")]
    #
    #         if len(tif_files) == 1:
    #             base_folder = output_folder
    #             create_folder(output_folder)
    #         else:
    #             base_folder = os.path.join(output_folder, str(file_name1))
    #             create_folder(base_folder)
    #         crop_and_replace_nan_inf(input_file, output_folder=base_folder, overlap=0.1, remove_inf_nan=False)
    # CDL
    # folder_path = r'J:\research\GEE\hetao_classification\2020HT\htCDL-rf\CDL'
    # files = os.listdir(folder_path)
    # # 遍历文件列表,裁剪单波段文件为128*128，另存到文件夹
    # for index, file_name in tqdm(enumerate(files), desc="Accessing List", unit="element"):
    #     file_path = os.path.join(folder_path, file_name)
    #     file_name1 = file_name.split('.')[0].split('_')[0] + '_' + file_name.split('.')[0].split('_')[1]
    #     if os.path.isfile(file_path):
    #         input_file = os.path.abspath(file_path)
    #         # 按照重叠率裁剪单波段tif图为128*128的图像
    #         output_folder = r'J:\research\GEE\hetao_classification\2020HT\SegmentationClass'
    #         tif_files = [file for file in files if file.endswith(".tif")]
    #
    #         if len(tif_files) == 1:
    #             base_folder = output_folder
    #             create_folder(output_folder)
    #         else:
    #             base_folder = os.path.join(output_folder, str(file_name1))
    #             create_folder(base_folder)
    #         crop_and_replace_nan_inf(input_file, output_folder=base_folder, overlap=0.1, remove_inf_nan=True)


    # 多个单波段图像堆叠到一起
    # 2019NEofCHINA
    # input_folder_path = r'J:\research\GEE\hetao_classification\NEofCHINA\JPEGImages-forPredict'  # 替换为包含子文件夹的文件夹路径
    # output_folder_path = r'J:\research\GEE\hetao_classification\NEofCHINA\12bands_forPredict'  # 替换为保存输出图像的文件夹路径
    # # band = ['EVI']
    # band = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
    # # month = ['202005', '202006', '202007', '202008', '202009', '202010']
    # month = ['201905', '201906', '201907', '201908', '201909', '201910']
    # # month = ['20200526', '20200625', '20200710', '20200809', '20200921', '20201020']
    # bands = [f"{m}_{b}" for m in month for b in band]
    # # 创建输出文件夹
    # if not os.path.exists(output_folder_path):
    #     os.makedirs(output_folder_path)
    # # 运行堆叠图像的函数
    # stack_images(input_folder_path, output_folder_path, bands)

    # 2020HT
    # input_folder_path = r'J:\research\GEE\hetao_classification\2020HT\JPEGImages'  # 替换为包含子文件夹的文件夹路径
    # output_folder_path = r'J:\research\GEE\hetao_classification\2020HT\EVI'  # 替换为保存输出图像的文件夹路径
    # # band = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
    # band = ['EVI']
    # # month = ['202005', '202006', '202007', '202008', '202009', '202010']
    # # month = ['201905', '201906', '201907', '201908', '201909', '201910']
    # month = ['20200526', '20200625', '20200710', '20200809', '20200921', '20201020']
    # # band = ['B4', 'B3', 'B2']
    # # month = ['202005']
    # bands = [f"{m}_{b}" for m in month for b in band]
    # # 创建输出文件夹
    # if not os.path.exists(output_folder_path):
    #     os.makedirs(output_folder_path)
    # # 运行堆叠图像的函数
    # stack_images(input_folder_path, output_folder_path, bands)
    #
    # # IOWAofAMERICA
    # input_folder_path = r'J:\research\GEE\hetao_classification\IOWAofAMERICA\JPEGImages'  # 替换为包含子文件夹的文件夹路径
    # output_folder_path = r'J:\research\GEE\hetao_classification\IOWAofAMERICA\EVI'  # 替换为保存输出图像的文件夹路径
    # # band = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
    # band = ['EVI']
    # month = ['202005', '202006', '202007', '202008', '202009', '202010']
    # # month = ['20200526', '20200625', '20200710', '20200809', '20200921', '20201020']
    # bands = [f"{m}_{b}" for m in month for b in band]
    # # 创建输出文件夹
    # if not os.path.exists(output_folder_path):
    #     os.makedirs(output_folder_path)
    # # 运行堆叠图像的函数
    # stack_images(input_folder_path, output_folder_path, bands)


    # # # 把背景变白色
    # # image_path = r"E:\jupyter_save\hetao_classification\crop_land_classification\code\images\rf.jpg"
    # # # 调用函数并传入图像路径
    # process_image(image_path)  # 修改为你的图像路径


    # 按照重叠率裁剪多波段图像
    # from glob import glob
    # input_path = r'J:\research\GEE\hetao_classification\2020HT\htCDL-rf\area'
    # input_files = glob(os.path.join(input_path, '*.tif'))
    # output_folder_path = r"J:\research\GEE\hetao_classification\2020HT\JPEGImages"
    # for input_file in input_files:
    #     crop_and_replace_nans_infs(input_file, output_folder_path, overlap=0.2)


    # 检查文件夹下的tif是否有inf,nan
    # input_folder = r"J:\research\GEE\hetao_classification\2020HT\JPEGImages"
    # check_inf_nan_tif(input_folder)


    # # 指定你的TIFF文件路径和输出路径
    # # input_tif_path = r"J:\research\GEE\hetao_classification\NEofCHINA_2019\NEofCHINA_crop_2019cdl-1.tif"
    # # output_tif_path = r"J:\research\GEE\hetao_classification\NEofCHINA_2019\output_image.tif"
    # # create_colored_image(input_tif_path, output_tif_path)
    # 导入os和PIL库


    # 对比两张图
    # image1 = Image.open(
    #     r"J:\research\GEE\hetao_classification\2019NEofCHINA\NEofCHINA_2019CDL\test\NEofCHINA_test_2019cdl_vis.tif")
    #  替换为第一张图像的路径
    # image2 = Image.open(
    #     r"J:\research\GEE\hetao_classification\2019NEofCHINA\NEofCHINA_2019CDL\test\prediction1.tif")
    #  替换为第二张图像的路径
    # target_file = r"J:\research\GEE\hetao_classification\2019NEofCHINA\NEofCHINA_2019CDL\test\comparison_result.tif"
    # comparasion(image1, image2, target_file)


    # # tif图5替换为2
    # input_tiff_path = r"J:\research\GEE\hetao_classification\IOWAofAMERICA\usa_cdl\AmerciaCDL_2020-10m.tif"
    # output_tiff_path = r"J:\research\GEE\hetao_classification\IOWAofAMERICA\usa_cdl\CDL\AmerciaCDL_2020-10m.tif"
    # original_value_to_replace = 5
    # new_value_to_assign = 2
    # replace_pixel_value(input_tiff_path, output_tiff_path, original_value_to_replace, new_value_to_assign)
