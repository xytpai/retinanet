import numpy as np 



def bbox_iou_np(bbox_a, bbox_b):
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i + 1e-10)



def gen_prec_rec_np(
        pred_bboxes, pred_labels, pred_scores, 
        gt_bboxes, gt_labels, 
        iou_th=0.5):
    '''
    pred_bboxes: [arr(N1,4), arr(N2,4), ..., arr(Nb,4)] ndarray.float 
    4:ymin,xmin,ymax,xmax
    pred_labels: [arr(N1), arr(N2), ..., arr(Nb)] ndarray.long   0:background
    pred_scores: [arr(N1), arr(N2), ..., arr(Nb)] ndarray.float 
    gt_bboxes:   [arr(M1,4), arr(M2,4), ..., arr(Mb,4)]
    gt_labels:   [arr(M1), arr(M2), ..., arr(Mb)]
    return:
    prec,        list(C)
    rec          list(C)
    '''
    # 存储每一个类别在标签中出现的次数, 字典类型
    # 使用 defaultdict 一开始无需初始化就能自增
    n_pos = defaultdict(int)
    # 存储每一个类别对应预测框的分数, 注意按照帧堆积
    # 堆积帧数据之前会进行一次从高到低的排序, 是为了加速计算
    score = defaultdict(list)
    match = defaultdict(list) # 记录每一个 score[l] 中的预测分对应的正确情况, difficult为-1
    # 得到一共多时少帧
    n_frames = len(gt_labels)
    # 按照帧迭代, 计算 n_pos, score, match
    for n in range(n_frames):
        # 得到当前帧的预测信息与标注信息
        pred_bbox  = pred_bboxes[n]
        pred_label = pred_labels[n]
        pred_score = pred_scores[n]
        gt_bbox    = gt_bboxes[n]
        gt_label   = gt_labels[n]
        # 该帧中，遍历预测与标注出现过的全部类别编号
        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            # 选中这个类别的预测框
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            # 对该类预测框按照分值从高到低排序
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]
            # 选中这个类别的标注框
            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            # 某类出现的总数目增加, 注意这里是增加标注出现数目
            n_pos[l] += gt_mask_l.sum()
            # 将排序好的该类的分数堆积, 注意 extend 得到的是一个列表
            score[l].extend(c)
            # 该类在预测框中没有出现过, 无需增加 match 项
            if len(pred_score_l) == 0:
                continue
            # 该类在标注框中没有出现过, 全部失匹
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue
            # 做一下预处理防止面积为 0
            pred_bbox_l = pred_bbox_l.copy()
            gt_bbox_l = gt_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l[:, 2:] += 1
            # 计算该帧该类每个预测框与所有标注框的IOU
            iou = bbox_iou_np(pred_bbox_l, gt_bbox_l) # arr(N,M)
            # 得到每个预测框对应最大的那个标注框的索引
            gt_index = iou.argmax(axis=1) # arr(N)
            # 如果某个预测框与对应最大的那个标注框的 IOU<th 则记号 -1
            gt_index[iou.max(axis=1) < iou_th] = -1 
            del iou
            # selec 表示某个标签的框是否被选中过
            # 由于标签框只能匹配一次
            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool) # arr(M)
            # 依次计算 match
            # gt_idx 表示每个预测框最大IOU的那个标注框索引
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    # 每个标注框只能匹配依次
                    if not selec[gt_idx]:
                        match[l].append(1)
                    else:
                        match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)
    # 得到标注中出现过的类别的数目
    # 由于0表示背景不可能出现, 因此这里不用+1
    n_fg_class = max(n_pos.keys())
    # 初始化prec 与 rec
    prec = [None]*n_fg_class
    rec  = [None]*n_fg_class
    # 迭代每个类别
    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)
        # 对该类的 match 按照分数从高到低排序
        order = score_l.argsort()[::-1]
        match_l = match_l[order]
        # 按照不同分数划分Th的所有TP与FP
        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)
        # 注意:  tp+fp == 0 时对应的值为 inf
        prec[l] = tp / (fp + tp)
        # 如果某类在标注中没有出现过, 那么其prec与rec都为None
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]
    return prec, rec



def gen_ap_np(prec, rec):
    '''
        1
    prec| .
        |    .
        |      .
        |       .
        |        .
       0 ———————————1
                   rec
    '''
    n_class = len(prec)
    # 这个 ap 包括了背景, 在索引0处
    # ap[0] = np.nan
    ap = np.empty(n_class) # np.float
    for l in range(n_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue
        # 放置哨点
        mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
        mrec = np.concatenate(([0], rec[l], [1]))
        # 倒着做 cum_max
        # mpre =  np.array([8,2,1,3,4,1])
        # np.maximum.accumulate(mpre[::-1])[::-1]
        # return: np.array([8,4,4,4,4,1])
        # 该措施为了平滑曲线, 即求闭包
        mpre = np.maximum.accumulate(mpre[::-1])[::-1]
        # 捕捉recall中变化的索引
        # mrec = np.array([1,2,3,3,5,6])
        # np.where(mrec[1:] != mrec[:-1])[0]
        # return: array([0,1,3,4], dtype=int64)
        i = np.where(mrec[1:] != mrec[:-1])[0]
        # 求包围面积: (\Delta recall) * prec
        ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap



def eval_detection(
    pred_bboxes, pred_labels, pred_scores, 
    gt_bboxes, gt_labels,
    iou_th=0.5):
    '''
    pred_bboxes: [arr(N1,4), arr(N2,4), ..., arr(Nb,4)] ndarray.float 
    4:ymin,xmin,ymax,xmax
    pred_labels: [arr(N1), arr(N2), ..., arr(Nb)] ndarray.long   0:background
    pred_scores: [arr(N1), arr(N2), ..., arr(Nb)] ndarray.float 
    gt_bboxes:   [arr(M1,4), arr(M2,4), ..., arr(Mb,4)]
    gt_labels:   [arr(M1), arr(M2), ..., arr(Mb)]
    '''
    prec, rec = gen_prec_rec_np(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels,
        iou_th=iou_th)
    ap = gen_ap_np(prec, rec)
    return {'ap': ap, 'map': np.nanmean(ap)}
