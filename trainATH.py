import shutil
import os
import sys
import time
import random
import string
import re
import datetime
import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np

from utils import CTCLabelConverter, CTCLabelConverterForBaiduWarpctc, AttnLabelConverter, Averager, TokenLabelConverter
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model
from test import validation
from utils import get_argsAnhTH

from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, MultiStepLR, ReduceLROnPlateau


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# python3 train.py --train_data data_lmdb_release/training --valid_data data_lmdb_release/validation --select_data MJ-ST --batch_ratio 0.5-0.5 --Transformation None --FeatureExtraction None --SequenceModeling None --Prediction None --Transformer --imgH 224 --imgW 224


def train(opt):
    """ dataset preparation  """
    "lọc dữ liệu"
    if not opt.data_filtering_off:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')
        # see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L130

    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    opt.eval = False
    train_dataset = Batch_Balanced_Dataset(opt)

    log = open(f'./saved_models/{opt.exp_name}/log_dataset.txt', 'a')
    opt.eval = True
    if opt.sensitive:
        opt.data_filtering_off = True
    AlignCollate_valid = AlignCollate( #  căng chỉnh lại hình ảnh
        imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, opt=opt)
    valid_dataset, valid_dataset_log = hierarchical_dataset(
        root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size,
        # 'True' to check training progress with validation function.
        shuffle=True,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)
    log.write(valid_dataset_log)
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()

    """ model configuration """
    if opt.Transformer:
        converter = TokenLabelConverter(opt)
    elif 'CTC' in opt.Prediction:
        if opt.baiduCTC:
            converter = CTCLabelConverterForBaiduWarpctc(opt.character)
        else:
            converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    print("llllllllllllllllll")
    print(opt.character)
    best_valid_loss=100
    best_train_loss=100
    if opt.rgb:
        opt.input_channel = 3
    'tạo ra model'
    model = Model(opt)

    # weight initialization
    if not opt.Transformer:
        for name, param in model.named_parameters():
            if 'localization_fc2' in name:
                print(f'Skip {name} as it is already initialized')
                continue
            try:
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_normal_(param)
            except Exception as e:  # for batchnorm.
                if 'weight' in name:
                    param.data.fill_(1)
                continue

    # data parallel for multi-GPU
    model = torch.nn.DataParallel(model).to(device)
    model.train()
    if opt.saved_model != '':
        print(f'loading pretrained model from {opt.saved_model}')
        if opt.FT:
            model.load_state_dict(torch.load(opt.saved_model), strict=False)
        else:
            model.load_state_dict(torch.load(opt.saved_model))
    # print("Model:")
    # print(model)

    """ setup loss """
    # README: https://github.com/clovaai/deep-text-recognition-benchmark/pull/209
    if 'CTC' in opt.Prediction:
        if opt.baiduCTC:
            # need to install warpctc. see our guideline.
            from warpctc_pytorch import CTCLoss
            criterion = CTCLoss()
        else:
            criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(
            device)  # ignore [GO] token = ignore index 0
    # loss averager
    loss_avg = Averager()

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    # print('Trainable params num : ', sum(params_num))
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

    # setup optimizer
    scheduler = None
    # if opt.adam:
    #     optimizer = optim.Adam(filtered_parameters,
    #                            lr=opt.lr, betas=(opt.beta1, opt.beta2))
    # else:
    #     optimizer = optim.Adadelta(
    #         filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)

    optimizer = optim.Adadelta(
        filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    if opt.scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=opt.num_iter)

    """ final options """
    # print(opt)
    with open(f'./saved_models/{opt.exp_name}/opt.txt', 'a') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        # print(opt_log)
        opt_file.write(opt_log)
        total_params = int(sum(params_num))
        total_params = f'Trainable network params num : {total_params:,}'
        print(total_params)
        opt_file.write(total_params)

    """ start training """
    start_iter = 0
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            print(f'continue to train, start_iter: {start_iter}')
        except:
            pass

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    iteration = start_iter

    while(True):
        # train part
        image_tensors, labels = train_dataset.get_batch()
        image = image_tensors.to(device)
        if not opt.Transformer:
            text, length = converter.encode(
                labels, batch_max_length=opt.batch_max_length)
        batch_size = image.size(0)

        if 'CTC' in opt.Prediction:
            preds = model(image, text)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            if opt.baiduCTC:
                preds = preds.permute(1, 0, 2)  # to use CTCLoss format
                cost = criterion(preds, text, preds_size, length) / batch_size
            else:
                preds = preds.log_softmax(2).permute(1, 0, 2)
                cost = criterion(preds, text, preds_size, length)
        elif opt.Transformer:
            target = converter.encode(labels)
            preds = model(image, text=target,
                          seqlen=converter.batch_max_length)
            cost = criterion(
                preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
        else:
            preds = model(image, text[:, :-1])  # align with Attention.forward
            target = text[:, 1:]  # without [GO] Symbol
            cost = criterion(
                preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

        model.zero_grad()
        cost.backward()
        # gradient clipping with 5 (Default)
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()

        loss_avg.add(cost)

        # validation part
        # To see training progress, we also conduct validation when 'iteration == 0'
        '----------------------------dang thu nghiem-------------------------------------'
        # if (iteration + 1) % 500 == 0:
        #     threading.Thread(target=ocr_test_de_tang_Do_chinh_xac_thu.tang_do_chinhxac(opt.train_data,f'./saved_models/{opt.exp_name}/best_accuracy.pth',opt.batch_max_length,opt.imgH,opt.imgW,opt.character)).start()
            
        '-------------------------------------------------------------------'    
        if (iteration + 1) % opt.valInterval == 0 :
            elapsed_time = time.time() - start_time
            print("THOI GIAN BAT DAU VALUE :",datetime.datetime.now())
           # for log
            with open(f'./saved_models/{opt.exp_name}/log_train.txt', 'a') as log:
                model.eval()
                with torch.no_grad():
                    thoigianbatdauvalid=time.time()
                    valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels, infer_time, length_of_data = validation(
                        model, criterion, valid_loader, converter, opt)
                model.train()
                print("thoi gian valid :",time.time()-thoigianbatdauvalid)
                print("kkk :",length_of_data)

                # training loss and validation loss
                loss_log = f'[{iteration+1}/{opt.num_iter}] Train loss: {loss_avg.val():0.9f}, Valid loss: {valid_loss:0.9f}, Elapsed_time: {elapsed_time:0.5f}\n now time: {str(datetime.datetime.now())},best_valid_loss:{best_valid_loss},best_train_loss : {best_train_loss}'
                # loss_log = f'[{iteration+1}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}, now time: {datetime.datetime.now()},best_valid_loss:{best_valid_loss}'
                # print(f'a:{loss_avg.val():0.9f}')
                # print(f'{loss_avg.val():0.12f}')
                loss_train_now=float(loss_avg.val())
                # print(loss_train_now)
                if loss_train_now<best_train_loss:
                    best_train_loss=loss_train_now
                    torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_train_losss.pth')
                loss_avg.reset()

                current_model_log = f'{"Current_accuracy":17s}: {current_accuracy}, {"Current_norm_ED":17s}: {current_norm_ED}'
                # current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.2f}'
                # keep best accuracy model (on valid dataset)
                torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/now.pth')
                if valid_loss<best_valid_loss:
                    best_valid_loss=valid_loss
                    torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_valid_losss.pth')
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(
                        model.state_dict(), f'./saved_models/{opt.exp_name}/best_accuracy.pth')
                if current_norm_ED > best_norm_ED:
                    best_norm_ED = current_norm_ED
                    torch.save(model.state_dict(),
                               f'./saved_models/{opt.exp_name}/best_norm_ED.pth')
                best_model_log = f'{"Best_accuracy":17s}: {best_accuracy}, {"Best_norm_ED":17s}: {best_norm_ED}'
                loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                print(loss_model_log)
                log.write(loss_model_log + '\n')
                    
                # show some predicted results
                dashed_line = '-' * 80
                head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                for gt, pred, confidence in zip(labels[:so_print], preds[:so_print], confidence_score[:so_print]):
                    if opt.Transformer:
                        pred = pred[:pred.find('[s]')]
                    elif 'Attn' in opt.Prediction:
                        gt = gt[:gt.find('[s]')]
                        pred = pred[:pred.find('[s]')]

                    # To evaluate 'case sensitive model' with alphanumeric and case insensitve setting.
                    if opt.sensitive and opt.data_filtering_off:
                        alphanumeric_case_insensitve =  string.printable[:-6]
                        out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
                        pred = re.sub(
                            out_of_alphanumeric_case_insensitve, '', pred)
                        gt = re.sub(
                            out_of_alphanumeric_case_insensitve, '', gt)

                    predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.8f}\t{str(pred == gt)}\n'
                predicted_result_log += f'{dashed_line}'
                print(predicted_result_log)
                # print("lr : ",opt.lr,len(labels),len( preds),len( confidence_score))

                log.write(predicted_result_log + '\n')
                # if best_accuracy> best_accuracy_can_du_an:
                #     print("du yêu cầu của dự án")
                #     sys.exit()
          
          

        # save model per 1e+5 iter.
        if (iteration + 1) % 1e+5 == 0:
            if os.path.exists(f'./saved_models/{opt.exp_name}/best_accuracy.pth'):
                shutil.copyfile(f'./saved_models/{opt.exp_name}/best_accuracy.pth',f'./saved_models/{opt.exp_name}/best_accuracy_{iteration+1}.pth')
            # torch.save(
            #     model.state_dict(), f'./saved_models/{opt.exp_name}/iter_{iteration+1}.pth')

        if (iteration + 1) == opt.num_iter:
            print('end the training')
            sys.exit()
        iteration += 1
        if scheduler is not None:
            scheduler.step()

"CUDA_VISIBLE_DEVICES=0 python3 trainATH.py --manualSeed=$RANDOM"
if __name__ == '__main__':
    so_print=10
    # lấy các tham số cần thuyết
    # best_accuracy_can_du_an=99.8
    opt = get_argsAnhTH()
    "nếu k đưa vào tên model thì mật định là cái tên đó"
    if not opt.exp_name:
        opt.exp_name = f'{opt.TransformerModel}' if opt.Transformer else f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
    print('--------------------------------')
    print(opt.TransformerModel)
    print('--------------------------------')
    opt.exp_name += f'-Seed{opt.manualSeed}'
    # nếu không tồn tại thì tạo ra model 
    os.makedirs(f'./saved_models/{opt.exp_name}', exist_ok=True)
    data_thongsocan=''
    data_thongsocan+="character : "+opt.character+"\n"
    data_thongsocan+="imgH : "+str(opt.imgH)+"\n"
    data_thongsocan+="imgW : "+str(opt.imgW)+"\n"
    data_thongsocan+="batch_max_length : "+str(opt.batch_max_length)+"\n"

    data_thongsocan+="Transformation : "+opt.Transformation+"\n"
    data_thongsocan+="FeatureExtraction : "+opt.FeatureExtraction+"\n"
    data_thongsocan+="SequenceModeling : "+opt.SequenceModeling+"\n"
    data_thongsocan+="Prediction : "+opt.Prediction+"\n"
    data_thongsocan+="select_data : "+opt.select_data+"\n"
    data_thongsocan+="batch_ratio : "+opt.batch_ratio+"\n"
    data_thongsocan+="batch_size : "+str(opt.batch_size)+"\n"
    data_thongsocan+="num_iter : "+str(opt.num_iter)+"\n"
    
    """ vocab / character number configuration """
    print(opt.character)
    data_thongsocan+="adam : "+str(opt.adam)+"\n"
    data_thongsocan+="lr : "+str(opt.lr)+"\n"
    data_thongsocan+="beta1 : "+str(opt.beta1)+"\n"
    data_thongsocan+="beta2 : "+str(opt.beta2)+"\n"
    with open(f'./saved_models/{opt.exp_name}/data_thongsocan.txt',"w") as f:
        f.write(data_thongsocan)
        f.close()
    print("name model : ",opt.exp_name)
    """ Seed and GPU setting ,lưu số mật định của ran dom"""
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    if opt.workers <= 0:
        opt.workers = (os.cpu_count() // 2) // opt.num_gpu

    if opt.num_gpu > 1:
        print('------ Use multi-GPU setting ------')
        print('if you stuck too long time with multi-GPU setting, try to set --workers 0')
        # check multi-GPU issue https://github.com/clovaai/deep-text-recognition-benchmark/issues/1
        opt.workers = opt.workers * opt.num_gpu
        opt.batch_size = opt.batch_size * opt.num_gpu

        """ previous version
        print('To equlize batch stats to 1-GPU setting, the batch_size is multiplied with num_gpu and multiplied batch_size is ', opt.batch_size)
        opt.batch_size = opt.batch_size * opt.num_gpu
        print('To equalize the number of epochs to 1-GPU setting, num_iter is divided with num_gpu by default.')
        If you dont care about it, just commnet out these line.)
        opt.num_iter = int(opt.num_iter / opt.num_gpu)
        """

    train(opt)
