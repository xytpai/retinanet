import torch 
import torch.nn.functional as F 



def focal_loss_detection(
    feature_cls, feature_reg, 
    targets_cls, targets_reg,
    alpha=0.25, gamma=2,
    factor_cls=1.0, factor_reg=1.0):
    '''
    Param:
    feature_cls: FloatTensor(b, an, classes)
    feature_reg: FloatTensor(b, an, 4)
    targets_cls: LongTensor(b, an)
    targets_reg: FloatTensor(b, an, 4)

    Note:
    auto allocate GPU
    '''
    # feature_cls = feature_cls.to(targets_cls.device)
    # feature_reg = feature_reg.to(targets_cls.device)

    b, an, classes = feature_cls.shape[0:3]
    feature_cls = feature_cls.view(b*an, classes)
    feature_reg = feature_reg.view(b*an, 4)
    targets_cls = targets_cls.view(b*an)
    targets_reg = targets_reg.view(b*an, 4)
    
    mask_cls = targets_cls > -1
    feature_reg = feature_reg[mask_cls]
    targets_reg = targets_reg[mask_cls]
    feature_cls = feature_cls[mask_cls]
    targets_cls = targets_cls[mask_cls]

    p = feature_cls.sigmoid() # [S+-, classes]

    targets_cls = targets_cls.to(feature_cls.device) #[S+-]
    one_hot = torch.zeros(feature_cls.shape[0], 
            1 + classes).to(feature_cls.device).scatter_(1, 
                targets_cls.view(-1,1), 1) # [S+-, 1+classes]
    one_hot = one_hot[:, 1:] # [S+-, classes]

    pt = p*one_hot + (1.0-p)*(1.0-one_hot)

    w = alpha*one_hot + (1.0-alpha)*(1.0-one_hot)
    w = w * torch.pow((1.0-pt), gamma)

    loss_cls = torch.sum(-w * (pt+1e-10).log())
 
    mask_reg = targets_cls > 0
    num_pos = float(torch.sum(mask_reg))

    assert num_pos>0, 'Make sure every image has assigned anchors.'
 
    feature_reg = feature_reg[mask_reg]
    targets_reg = targets_reg[mask_reg].to(feature_cls.device)

    loss_reg = F.smooth_l1_loss(feature_reg, targets_reg, reduction='sum')
    loss = (factor_cls*loss_cls + factor_reg*loss_reg) / num_pos
    return loss



def loss_detection(feature_cls, feature_reg, targets_cls, targets_reg):
    return focal_loss_detection(feature_cls, feature_reg, targets_cls, targets_reg)

