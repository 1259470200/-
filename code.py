from MedicalImageAnalysis.utils import *

if __name__ == '__main__':
    BASE_DIR = r'.\dataset'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    # config
    mode = 'train'  # mode='train'则进行训练，mode='inference'则进行测试
    lr = 0.01
    BATCH_SIZE = 1  # 由于cuda容量比较小，所以只能是batch_size为1，为2都gpu内存不够
    max_epoch = 30
    start_epoch = 0
    lr_step = 50
    val_interval = 1
    checkpoint_interval = 1  # 每隔这么多epoch就保存一次模型
    vis_num = float('inf')
    mask_thres = 0.5
    # 更换数据集只需修改此处数据集的路径就好了
    # train_dir = r'.\dataset\DRIVE\training'   # DRIVE数据集
    # valid_dir = r'.\dataset\DRIVE\test'
    train_dir = r'.\dataset\CHASEDB1\training'   # CHASEDB1数据集
    valid_dir = r'.\dataset\CHASEDB1\test'
    # 数据转换，这一步必不可少，必须得先转化为tensor才行
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    transform_compose = transforms.Compose([
        transforms.CenterCrop(560),  # 中心裁剪
        transforms.ToTensor(),
    ])
    # 验证数据集不用做数据增强
    valid_transform_compose = transforms.Compose([
        transforms.CenterCrop(560),  # 中心裁剪
        transforms.ToTensor(),
    ])

    # step1: prepare data
    train_set = MyDataset(data_dir=train_dir, transform=transform_compose)  # 需要重写Dataset
    valid_set = MyDataset(data_dir=valid_dir, transform=valid_transform_compose)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=True, drop_last=False)

    # step2: model
    net = UNet(in_channels=3, out_channels=1, init_features=32)
    net.to(device)   # 注意cuda的内存非常有限，有时候在python console中没释放cuda内存也会造成内存不够的问题

    # step3: loss
    loss_fn = nn.MSELoss()

    # step4: optimize
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=0.1)

    if mode == 'train':
        # step5: iterate
        train_curve = []
        valid_curve = []
        train_dice_curve = []
        valid_dice_curve = []
        for epoch in range(start_epoch, max_epoch):
            train_loss_total = 0.
            train_dice_total = 0.
            net.train()
            for iter, (inputs, labels) in enumerate(train_loader):
                st = time.time()
                if torch.cuda.is_available:
                    inputs, labels = inputs.to(device), labels.to(device)
                # forward
                outputs = net(inputs)
                # backward
                optimizer.zero_grad()
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                # print
                train_dice = compute_dice(outputs.ge(mask_thres).cpu().data.numpy(), labels.cpu())
                train_dice_curve.append(train_dice)
                train_curve.append(loss.item())
                train_loss_total += loss.item()
                print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] running_loss: {:.4f}, mean_loss: {:.4f}"
                      " running_dice: {:.4f} lr:{}, time: {}".format(epoch, max_epoch, iter+1, len(train_loader),
                        loss.item(), train_loss_total/(iter+1), train_dice, scheduler.get_lr(), time.time()-st))
            scheduler.step()

            # save model
            if (epoch+1) % checkpoint_interval == 0:
                print('saving model................')
                PATH = r'.\models\cmodel.pkl'
                torch.save(net.state_dict(), PATH)

            if (epoch+1) % val_interval == 0:
                net.eval()
                valid_loss_total = 0
                valid_dice_total = 0
                with torch.no_grad():
                    for j, (inputs, labels) in enumerate(valid_loader):
                        if torch.cuda.is_available():
                            inputs, labels = inputs.to(device), labels.to(device)
                        outputs = net(inputs)
                        loss = loss_fn(outputs, labels)
                        valid_loss_total += loss.item()
                        valid_dice = compute_dice(outputs.ge(mask_thres).cpu().data, labels.cpu())
                        valid_dice_total += valid_dice
                    valid_loss_mean = valid_loss_total/len(valid_loader)
                    valid_dice_mean = valid_dice_total/len(valid_loader)
                    valid_curve.append(valid_loss_mean)
                    valid_dice_curve.append(valid_dice_mean)
                    print("Valid:\t Epoch[{:0>3}/{:0>3}] mean_loss: {:.4f} dice_mean: {:.4f}".format(epoch, max_epoch, valid_loss_mean, valid_dice_mean))
                    # 训练中记得保存训练loss曲线，也是重写覆盖法避免过程中数据丢失
                    loss_dict = {'train_curve': train_curve, 'valid_curve': valid_curve, 'train_dice_curve': train_dice_curve, 'valid_dice_curve': valid_dice_curve}
                    save_obj(loss_dict, r'.\models\c3loss_dict')
    else:
        # 加载模型
        device = torch.device('cpu')
        net = UNet(in_channels=3, out_channels=1, init_features=32)
        PATH = r'.\models\cmodel.pkl'
        net.load_state_dict(torch.load(PATH))  # 加载参数
        net.to(device)
        # 计算各个指标，分别为sensitivity, specificity, F1-score, accuracy
        index = {'se': [], 'sp': [], 'f1': [], 'acc': []}
        for idx, (inputs, labels) in enumerate(valid_loader):
            if torch.cuda.is_available():
                inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            pred = outputs.ge(mask_thres)
            # 对预测结果进行逆transforms操作
            mask_pred = outputs.ge(0.5).cpu().data.numpy().astype('uint8')

            img_hwc = inputs.cpu().data.numpy()[0, :, :, :].transpose((1, 2, 0))
            label = labels.cpu().data.numpy()[0, :, :, :].transpose((1, 2, 0)).astype('uint8')
            plt.figure()
            plt.subplot(131).imshow(img_hwc), plt.title('input_image')
            label = label.squeeze() * 255
            plt.subplot(132).imshow(label, cmap='gray'), plt.title('ground truth')
            # mask_pred_gray = np.transpose(mask_pred.squeeze()*255)
            mask_pred_gray = mask_pred.squeeze() * 255
            plt.subplot(133).imshow(mask_pred_gray, cmap='gray'), plt.title('predict')
            # 保存图像到指定目录
            filename = r'.\tmp\\'+str(idx)+'c2.png'
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
            eva = Evaluation(label, mask_pred_gray)
            index['se'].append(eva.sensitivity())
            index['sp'].append(eva.specificity())
            index['f1'].append(eva.F1())
            index['acc'].append(eva.accuracy())
        save_obj(index, r'.\models\index_c2')
