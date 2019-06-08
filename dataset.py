import torch
import numpy as np 
import matplotlib.pyplot as plt 
import os, math, random
from PIL import Image
import torch.utils.data as data



class Dataset_Detection(data.Dataset):
    def __init__(self, root, list_file, size, 
                    train=True, transform=None, boxarea_th=25):
        '''
        root:       放置图像的文件夹路径
        list_file:  数据集标主文件路径
        size:       训练图像的尺寸，输入一个Int类型训练图像为正方形
        train:      如果为True会进行随机裁剪以及随机翻转，否则只是中心裁剪
        transform:  使用Torch自带的转换代码对PIL图像进行转换，一般为ToTensor
        boxarea_th: 框面积如果小于这个值则去掉
        数据标主文件的格式如下：
            每一行包含一张图像的所有标主，每一行都以 '\n' 结尾
            对每一张图像的标主顺序为
            img_name ymin_1 xmin_1 ymax_1 xmax_1 class_1 ...
            举例:  2012_004276.jpg 106 109 173 226 14 233 91 295 220 14\n
            注意：类别编号从1开始，由于0被预留出作为背景类
        注意导入模块:
        import os, math, random
        from PIL import Image
        import torch.utils.data as data
        '''
        self.root = root
        self.size = size
        self.train = train
        self.transform = transform
        self.boxarea_th = boxarea_th
        # 下面这两个列表将包含图像名、每个图像的框、每个框的标签
        self.fnames = []
        self.boxes = []
        self.labels = []
        # 先一次性读取完所有标主信息
        with open(list_file) as f:
            lines = f.readlines()
            self.num_samples = len(lines)
        # 迭代每一行，即迭代每一张图像
        for line in lines:
            # split默认空格为分隔符
            # strip()为消除首尾空格
            splited = line.strip().split()
            self.fnames.append(splited[0])
            # 每一个框需要5个信息
            num_boxes = (len(splited) - 1) // 5
            box = []
            label = []
            # 按照 ymin, xmin, ymax, xmax 读取一张图的所有框
            for i in range(num_boxes):
                ymin = splited[1+5*i]
                xmin = splited[2+5*i]
                ymax = splited[3+5*i]
                xmax = splited[4+5*i]
                c = splited[5+5*i]
                box.append([float(ymin),float(xmin),float(ymax),float(xmax)])
                label.append(int(c))
            # 一张图的所有框用一个Tensor来表示
            # self.fnames: [ 'a.jpg',      'b.jpg',       ...]
            # self.boxes:  [ Tensor[Na,4],  Tensor[Nb,4], ...]
            # self.labels: [ Tensor[Na],    Tensor[Nb],   ...]
            self.boxes.append(torch.FloatTensor(box))
            self.labels.append(torch.LongTensor(label))
            
    def __getitem__(self, idx):
        '''
        return: (image  [3,H,W], t.float32
                 boxes  [N,4],   t.float32
                 Labels [N])     t.long
        如果transform不是None的话，返回PIL图像，且排布为[W,H]
        框的每个元素排布为: ymin, xmin, ymax, xmax
        '''
        # 得到图像数据
        fname = self.fnames[idx]
        img = Image.open(os.path.join(self.root, fname))
        if img.mode != 'RGB': # 黑白图要转成RGB格式
            img = img.convert('RGB')
        # 拿到一幅图像的框与标签
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()
        # 做数据增强
        size = self.size
        if self.train:
            img, boxes = random_flip(img, boxes)
            img, boxes = random_crop(img, boxes)
            img, boxes = resize(img, boxes, (size,size))
        else:
            img, boxes = resize(img, boxes, size)
            img, boxes = center_crop(img, boxes, (size,size))
        # 筛选面积大于阈值的框
        hw = boxes[:, 2:] - boxes[:, :2] # [N,2]
        area = hw[:, 0] * hw[:, 1] # [N]
        mask = area > self.boxarea_th
        boxes = boxes[mask]
        labels = labels[mask]
        # 对PIL图像做转换
        if self.transform is not None:
            img = self.transform(img)
        return img, boxes, labels

    def collate_fn(self, data):
        '''
         return: (image  [B,3,H,W],                    t.float32
                  boxes  ([N1,4], [N2,4], ..., [Nb,4]) t.float32
                  Labels ([N1], [N2], ..., [Nb])       t.long
        '''
        # 定义如何分配一个miniBatch的数据
        # 为了增加速度，最好将图像都放置到一个Tensor中
        # 标签数据量不大，可以进行迭代
        img, boxes, labels = zip(*data)
        img = torch.stack(img, dim=0)
        return img, boxes, labels

    def __len__(self):
        return self.num_samples



def resize(img, boxes, size, max_size=1000):
    '''
    将一张PIL图像映射到size大小, 同时变换标主
    size 可以为Int类型，也可以为二元组
    如果为前者，那么将图像短边整成 size 大小, 长边则按等比例缩放
    max_size 代表图像最长边调整后不能大于一个值
    '''
    w, h = img.size
    if isinstance(size, int):
        size_min = min(w,h)
        size_max = max(w,h)
        # 如果整成正方形，使用最小的那条边计算比例
        sw = sh = float(size) / size_min
        if sw * size_max > max_size:
            # 如果长边缩放后超过限定值，那就按长边的比例缩放
            sw = sh = float(max_size) / size_max
        ow = int(w * sw + 0.5)
        oh = int(h * sh + 0.5)
    else: 
        # 如果是一个二元组直接进行映射
        # 使用这种方案来保证每一张图像的尺度固定
        oh, ow = size
        sw = float(ow) / w
        sh = float(oh) / h
    return img.resize((ow,oh), Image.BILINEAR), \
           boxes*torch.Tensor([sh,sw,sh,sw])



def center_crop(img, boxes, size):
    '''
    将一张PIL图像裁剪出中间的一块, 同时变换标注
    size 表示中间那块尺寸, 为一个二元组(H,W)
    '''
    w, h = img.size
    oh, ow = size
    # 确定裁剪区域左上角坐标
    i = int(round((h - oh) / 2.))
    j = int(round((w - ow) / 2.))
    img = img.crop((j, i, j+ow, i+oh))
    # 框坐标需要减一下
    boxes -= torch.Tensor([i,j,i,j])
    # 框坐标不能超过输出图像范围
    boxes[:,1::2].clamp_(min=0, max=ow-1)
    boxes[:,0::2].clamp_(min=0, max=oh-1)
    return img, boxes



def random_crop(img, boxes, scale_min=0.85):
    '''
    将一张PIL图像按照一定尺寸与长宽比区间随机裁剪, 同时变换标注
    scale_min 表示裁剪区域最小的尺寸比例
    '''
    success = False
    # 经过最多十次尝试
    for attempt in range(10):
        area = img.size[0] * img.size[1]
        # 面积按照 scale_min~1 的比例随机选取
        target_area = random.uniform(scale_min, 1.0) * area
        # 长宽比按照区间 3/4~4/3 随机选取
        aspect_ratio = random.uniform(3. / 4, 4. / 3)
        # 根据长宽比与面积确定输出长宽
        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))
        # 长宽随机颠倒
        if random.random() < 0.5:
            w, h = h, w
        # 查看是否满足条件
        if w <= img.size[0] and h <= img.size[1]:
            # 选定区域随机移位
            x = random.randint(0, img.size[0] - w)
            y = random.randint(0, img.size[1] - h)
            success = True
            break
    # 如果尝试全部不满足要求
    # 那么就进行中心裁剪
    if not success:
        w = h = min(img.size[0], img.size[1])
        x = (img.size[0] - w) // 2
        y = (img.size[1] - h) // 2
    # 后面与中心裁剪一样操作
    img = img.crop((x, y, x+w, y+h))
    boxes -= torch.Tensor([y,x,y,x])
    boxes[:,1::2].clamp_(min=0, max=w-1)
    boxes[:,0::2].clamp_(min=0, max=h-1)
    return img, boxes



def random_flip(img, boxes):
    '''
    将一张PIL图像随机水平翻转, 同时变换标注
    '''
    if random.random() < 0.5:
        # PIL自带翻转函数
        img = img.transpose(Image.FLIP_LEFT_RIGHT) 
        w = img.width
        xmin = w - boxes[:,3]
        xmax = w - boxes[:,1]
        boxes[:,1] = xmin
        boxes[:,3] = xmax
    return img, boxes



def show_bbox(img, boxes, labels, NAME_TAB, segimg=None):
    '''
    根据一张numpy图像及其标注信息显示目标框
    img:     (C,H,W)  0~1, ndarray
    boxes:   (N,4)    ymin, xmin, ymax, xmax
    labels:  (N)      long
    NAME_TAB: 标签序号对应的文字列表
    如果存在语义图 segimg 那么分开显示出来
    导入模块:
    import numpy as np
    import matplotlib.pyplot as plt
    '''
    # 首先转化成(H,W,C)格式
    img = img.copy().transpose([1,2,0])
    # 迭代每一个框绘图
    for i in range(boxes.shape[0]):
        y_min = int(boxes[i, 0])
        x_min = int(boxes[i, 1])
        y_max = int(boxes[i, 2])
        x_max = int(boxes[i, 3])
        # 标注红色
        img[y_min, x_min:x_max] = np.array([1,0,0])
        img[y_max, x_min:x_max] = np.array([1,0,0])
        img[y_min:y_max, x_min] = np.array([1,0,0])
        img[y_min:y_max, x_max] = np.array([1,0,0])
        if segimg is not None:
            plt.subplot(1,2,1)
        xy = (x_min, y_min+5)
        bbox_t = {'fc':'red'}
        plt.annotate(NAME_TAB[int(labels[i])], color='white', 
            xy=xy, xytext=xy, bbox=bbox_t, )
    if segimg is None:
        plt.imshow(img)
    else:
        segimg = segimg.copy()
        plt.imshow(img)
        plt.subplot(1,2,2)
        plt.imshow(segimg)
    plt.show()



if __name__ == '__main__':
    import torchvision.transforms as transforms
    # TODO: 确定类名列表
    VOC_LABEL_NAMES = (
    'background',#0
    'aeroplane',#1
    'bicycle',#2
    'bird',#3
    'boat',#4
    'bottle',#5
    'bus',#6
    'car',#7
    'cat',#8
    'chair',#9
    'cow',#10
    'diningtable',#11
    'dog',#12
    'horse',#13
    'motorbike',#14
    'person',#15
    'pottedplant',#16
    'sheep',#17
    'sofa',#18
    'train',#19
    'tvmonitor'#20
    )
    # TODO: 确定路径
    root = '/home1/xyt/dataset/VOC2012/JPEGImages'
    list_file = 'data/voc_trainval.txt'
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    DS = Dataset_Detection(root, list_file, 513, transform=transform)
    dataloader = torch.utils.data.DataLoader(DS, 
        batch_size=8, shuffle=True, num_workers=0, collate_fn=DS.collate_fn)
    for imgs, boxes, labels in dataloader:
        print(len(imgs), len(boxes), len(labels))
        print(imgs[0].shape, boxes[0].shape, labels[0].shape)
        show_bbox(imgs[0].numpy(), boxes[0], labels[0], VOC_LABEL_NAMES)
        break
