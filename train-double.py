import os
import random
import json
from tqdm import tqdm
import jittor as jt
from jittor import nn, transform as jt_transform

from run_utils import *
from datasets.TrainSet import TrainSet_double
# from jittor.dataset import Dataset
# from datasets import build_dataset
from clip.amu import AMU_Model ,tfm_clip,tfm_aux
from clip.moco import load_moco
# from clip.mocovit import load_vit
from clip import clip
from parse_args import parse_args
from utils import *
from datasets.utils import DatasetWrapper
import numpy
from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora
def freeze_bn(m):
    classname = m.__class__.__name__
    if 'BatchNorm' in classname:
        m.eval()

def write_top5_results_to_txt(logits, output_file, image_names):
    
    top5_indices = jt.argsort(logits, dim=1, descending=True)[0].numpy()[:,:5]
    with open(output_file, 'w') as f:
        for i in range(len(image_names)):
            image_name = image_names[i]
            top5_classes = top5_indices[i].tolist()
            f.write(f"{image_name[9:]} {' '.join(map(str, top5_classes))}\n")


def get_image_names_from_txt(txt_file):
    image_names = []
    with open(txt_file, 'r') as f:
        for line in f:
            columns = line.strip().split()
            if columns:  # 确保行不为空
                image_names.append(columns[0])
    return image_names
    
def train_one_epoch(model, data_loader, optimizer, scheduler, logger):
    # Train
    model.train()
    model.apply(freeze_bn)  # freeze BN-layer
    correct_samples, all_samples = 0, 0
    loss_list = []
    loss_aux_list = []
    loss_merge_list = []
    # for i, (images, target) in enumerate(data_loader):
    #     print(f"Batch {i + 1}/{len(data_loader)}: Loaded {len(images)} samples")
    for i, (images, target) in enumerate(tqdm(data_loader)):

        return_dict = model(images, labels=target)

        acc = cls_acc(return_dict['logits'], target)
        correct_samples += acc / 100 * len(return_dict['logits'])
        all_samples += len(return_dict['logits'])
        
        loss_list.append(return_dict['loss'].item())
        loss_aux_list.append(return_dict['loss_aux'].item())
        loss_merge_list.append(return_dict['loss_merge'].item())

        optimizer.zero_grad()
        optimizer.backward(return_dict['loss'])
        optimizer.step()
        scheduler.step()
        jt.sync_all()
        # jt.display_memory_info()

    logger.info(' Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))
    logger.info("Loss_aux: {:.4f}, Loss_merge: {:.4f}".format(
        sum(loss_aux_list)/len(loss_aux_list), sum(loss_merge_list)/len(loss_merge_list)))

def train_and_eval(args, dataset,logger, model, clip_test_features, aux_test_features, test_labels, clip_val_features, aux_val_features, val_labels, train_loader_F):
    jt.flags.use_cuda = 1
    # model.requires_grad_(False)
    model.aux_adapter.requires_grad_(True)
    model.tipadapter.requires_grad_(True)
    for name, param in model.named_parameters():
        if param.requires_grad :
            print(name, param.requires_grad)

    optimizer = jt.optim.AdamW(
        # model.parameters(),
        [
    {"params": model.aux_adapter.parameters()},
    {"params": model.tipadapter.parameters()}],
        weight_decay=0.01,
        lr=args.lr,
        eps=1e-4
    )

    scheduler = jt.lr_scheduler.CosineAnnealingLR(optimizer, args.train_epoch * len(train_loader_F))

    best_acc, best_epoch = 0.0, 0

    for train_idx in range(1, args.train_epoch + 1):
        logger.info('Train Epoch: {:} / {:}'.format(train_idx, args.train_epoch))
        train_one_epoch(model, train_loader_F, optimizer, scheduler, logger)
        # Eval
        model.eval()
        with jt.no_grad():
            return_dict = model(
                clip_features=clip_val_features,
                aux_features=aux_val_features,
                labels=val_labels
            )
            all_logits = return_dict['logits']
            aux_logits = return_dict['aux_logits']
            tip_logits = return_dict['tip_logits']

            acc = cls_acc(all_logits, val_labels)
            acc_aux = cls_acc(aux_logits, val_labels)
            acc_tip = cls_acc(tip_logits, val_labels)

        logger.info("----- aux  val Acc: {:.4f} ----".format(acc_aux))
        logger.info("----- tip  val Acc: {:.4f} ----".format(acc_tip))
        logger.info("----- all  val Acc: {:.4f} -----".format(acc))

        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            logger.info("-----开始测试----- ")
            return_dict = model(
                clip_features=clip_test_features,
                aux_features=aux_test_features,
                labels=test_labels
                
            )
            clip_logits = return_dict['logits']
            # aux_logits = return_dict['aux_logits']
            # tip_logits = clip_logits + aux_logits * 3
            image_names = get_image_names_from_txt('caches/test.txt')
            write_top5_results_to_txt(clip_logits,'result/result.txt', image_names)
            logger.info("-----写入完成-----\n ")
            
            #分别获取tip和aux的logits
            tip_top1_indices = np.argmax(tip_logits.numpy(), axis=1)  # 每行的top1索引
            tip_top1_logits = np.max(tip_logits.numpy(), axis=1)
            
            all_top1_indices = np.argmax(all_logits.numpy(), axis=1)  # 每行的top1索引
            all_top1_logits = np.max(all_logits.numpy(), axis=1)
            # # 定义类别区间
            # class_ranges = [
            #     (0, 142, 16),#100
            #     (143, 243, 16),#120
            #     (244, 373, 16)
            # ]
            
            # top_indices = []
            
            # # 按类别区间进行排序并选取每个类别前n个logits最大的样本
            # for start, end, num_samples in class_ranges:
            #     for cls in range(start, end + 1):
            #         cls_indices = np.where(all_top1_indices == cls)[0]
            #         cls_logits = all_top1_logits[cls_indices]
            #         if len(cls_logits) > 0:
            #             sorted_cls_indices = cls_indices[np.argsort(cls_logits)[-num_samples:]]
            #             top_indices.extend(sorted_cls_indices)
            
            # top_indices = np.array(top_indices)
            
            # # 提取选取的标签和预测值
            # top_val_labels = val_labels[top_indices]
            # top_predictions = all_top1_indices[top_indices]
            
            # # 计算准确度
            # accuracy = np.mean(top_val_labels.numpy() == top_predictions)
            # print('最终的准确率', accuracy)
            # print('样本总数:', len(top_val_labels.numpy()))
            
            # # 提取初始索引
            # image_paths = [itemval.impath.strip() for itemval in dataset.val]
            # same_paths_labels = [[image_paths[i], all_top1_indices[i]] for i in top_indices]
            
            ##----以下为随机----
            # # 定义类别区间
            # class_ranges = [
            #     (0, 142, 100),
            #     (143, 243, 120)
            # ]
            
            # top_indices = []
            
            # # 按类别区间进行排序并选取每个类别前n个logits最大的样本
            # for start, end, num_samples in class_ranges:
            #     for cls in range(start, end + 1):
            #         cls_indices = np.where(all_top1_indices == cls)[0]
            #         cls_logits = all_top1_logits[cls_indices]
            #         if len(cls_logits) > 0:
            #             sorted_cls_indices = cls_indices[np.argsort(cls_logits)[-num_samples:]]
            #             top_indices.extend(sorted_cls_indices)
            
            # # 处理(244, 373)区间，获取前160个
            # for cls in range(244, 373 + 1):
            #     cls_indices = np.where(all_top1_indices == cls)[0]
            #     cls_logits = all_top1_logits[cls_indices]
            #     if len(cls_logits) > 0:
            #         sorted_cls_indices = cls_indices[np.argsort(cls_logits)[-160:]]
            #         top_indices.extend(sorted_cls_indices)
            
            # # 处理(244, 373)区间，第160到230个
            # for cls in range(244, 373 + 1):
            #     cls_indices = np.where(all_top1_indices == cls)[0]
            #     cls_logits = all_top1_logits[cls_indices]
            #     if len(cls_logits) > 160:
            #         additional_cls_indices = cls_indices[np.argsort(cls_logits)[-230:-160]]
            #         random_labels = np.random.randint(244, 374, size=len(additional_cls_indices))
            #         for i, new_cls in zip(additional_cls_indices, random_labels):
            #             all_top1_indices[i] = new_cls
            #         top_indices.extend(additional_cls_indices)
            
            # top_indices = np.array(top_indices)
            
            # # 提取选取的标签和预测值
            # top_val_labels = val_labels[top_indices]
            # top_predictions = all_top1_indices[top_indices]
            
            # # 计算准确度
            # accuracy = np.mean(top_val_labels.numpy() == top_predictions)
            # print('最终的准确率', accuracy)
            # print('样本总数:', len(top_val_labels.numpy()))
            
            # # 提取初始索引
            # image_paths = [itemval.impath.strip() for itemval in dataset.val]
            # same_paths_labels = [[image_paths[i], all_top1_indices[i]] for i in top_indices]

            # #---以下为两者取1-----
            same_indices = np.where(tip_top1_indices == all_top1_indices)[0]
            same_val_labels = val_labels[same_indices]
            same_predictions = tip_top1_indices[same_indices]  # 可以选择 aux_top1_indices 或 tip_top1_indices，因为它们相同
            accuracy = np.mean(same_val_labels.numpy() == same_predictions)
            print('两个的准确率',accuracy)
            
            same_all_logits = all_top1_logits[same_indices]
            
            # 按类别进行排序并选取每个类别前4个logits最大的样本
            unique_classes = np.unique(same_predictions)
            top4_indices = []
            
            for cls in unique_classes:
                cls_indices = np.where(same_predictions == cls)[0]
                cls_logits = same_all_logits[cls_indices]
                sorted_cls_indices = cls_indices[np.argsort(cls_logits)[-16:]]
                top4_indices.extend(sorted_cls_indices)
            
            top4_indices = np.array(top4_indices)
            
            # 提取前4个的标签和预测值
            top4_val_labels = same_val_labels[top4_indices]
            top4_predictions = same_predictions[top4_indices]
            
            # 计算准确度
            
            accuracy = np.mean(top4_val_labels.numpy() == top4_predictions)
            print('最终的准确率',accuracy)
            print(len(top4_val_labels.numpy()))
            
            initial_indices = same_indices[top4_indices]
            image_paths= []
            same_paths_labels = []
            # 获取图片路径
            image_paths = [itemval.impath.strip() for itemval in dataset.val]
            same_paths_labels = [[image_paths[i], tip_top1_indices[i]] for i in initial_indices]
            
    logger.info(f"----- Best Test Acc: {best_acc:.4f}, at epoch: {best_epoch}.-----\n")
    with open('caches/temp_labels.txt', 'w') as file:
        for path, label in same_paths_labels:
            file.write(f"{path[7:]} {label}\n")
    

if __name__ == '__main__':
    jt.flags.use_cuda = 1

    # Load config file
    parser = parse_args()
    args = parser.parse_args()
    argslora = get_arguments()
    cache_dir = os.path.join('./caches', args.dataset)
    os.makedirs(cache_dir, exist_ok=True)
    args.cache_dir = cache_dir

    logger = config_logging(args)
    logger.info("\nRunning configs.")
    args_dict = vars(args)
    message = '\n'.join([f'{k:<20}: {v}' for k, v in args_dict.items()])
    logger.info(message)


    clip_model, preprocess = clip.load('ViT-B-32.pkl')#ViT-B-32.pkl
    clip_model.eval()
    
    list_lora_layers = apply_lora(argslora, clip_model)
    load_lora(argslora, list_lora_layers)
    clip_model.eval()

    aux_model, args.feat_dim = load_moco("r-50-1000ep.pkl")
    # aux_model, args.feat_dim = load_vit("vit-b-300ep.pkl")
    aux_model.eval()


    random.seed(args.rand_seed)
    numpy.random.seed(args.rand_seed)
    jt.set_global_seed(args.rand_seed)
    jt.seed(args.rand_seed)
    jt.set_seed(args.rand_seed)

    
    logger.info("Loading ImageNet dataset....")
    # dataset = build_dataset('trainset', '', 4)
    

    dataset = TrainSet_double()
    train_loader = DatasetWrapper(data_source=dataset.train_x, batch_size=32, tfm=tfm_train_base, is_train=True, shuffle=False)
    train_loader_shuffle = DatasetWrapper(data_source=dataset.train_x, batch_size=32, tfm=tfm_train_base, is_train=True, shuffle=True)

    val_loader = DatasetWrapper(data_source=dataset.val, batch_size=128, is_train=False, tfm=tfm_test_base, shuffle=False)
    test_loader = DatasetWrapper(data_source=dataset.test, batch_size=128, is_train=False, tfm=tfm_test_base, shuffle=False)


    logger.info("Getting textual features as CLIP's classifier...")
    classnames = open('laoa.txt').read().splitlines()
    clip_weights = clip_classifier(classnames, dataset.template, clip_model)


    logger.info(" load  aux_features...")
    aux_features, aux_labels = load_aux_weight(args, aux_model, train_loader, tfm_norm=tfm_aux)


    logger.info("Constructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values = build_cache_model(args, clip_model, train_loader, tfm_norm=tfm_clip)


    logger.info("Loading clip features and labels from val set.")
    val_clip_features, val_labels = load_test_features(args, "val", clip_model, val_loader, tfm_norm=tfm_clip, model_name='clip')
    logger.info("Loading aux features and labels from val set.")
    val_aux_features, val_labels = load_test_features(args, "val", aux_model, val_loader, tfm_norm=tfm_aux, model_name='aux')

    val_clip_features = jt.array(val_clip_features)
    val_labels = jt.array(val_labels)
    
    val_aux_features =  jt.array(val_aux_features)



    logger.info("Loading clip features and labels from test set.")
    test_clip_features, test_labels = load_test_features(args, "test", clip_model, test_loader, tfm_norm=tfm_clip, model_name='clip')
    logger.info("Loading aux features and labels from test set.")
    test_aux_features, test_labels = load_test_features(args, "test", aux_model, test_loader, tfm_norm=tfm_aux, model_name='aux')

    test_clip_features =  jt.array(test_clip_features)
    test_aux_features = jt.array(test_aux_features)



    val_clip_features = val_clip_features.astype(jt.float32)
    clip_weights = clip_weights.astype(jt.float32)
    cache_values = cache_values.astype(jt.float32)


    tmp =  val_clip_features / val_clip_features.norm(dim=-1, keepdim=True)
    l = 100. * tmp @ clip_weights
    l_argmax = l.argmax(dim=-1)
    argmax_result = l_argmax[0]  
    correct_predictions = (argmax_result == val_labels).sum().item()
    print(f"Correct predictions:{correct_predictions} {correct_predictions/len(val_labels) * 100:.4f}")
    
    model = AMU_Model(
        clip_model=clip_model,
        cache_keys=cache_keys,
        cache_values=cache_values,
        aux_model=aux_model,
        sample_features=[aux_features, aux_labels],
        clip_weights=clip_weights,
        feat_dim=args.feat_dim,
        class_num=374,
        lambda_merge=args.lambda_merge,
        alpha=args.alpha,
        alpha_c=args.alphatip,
        uncent_type=args.uncent_type,
        uncent_power=args.uncent_power
    )
    
    train_and_eval(args, dataset, logger, model, test_clip_features, test_aux_features, test_labels, val_clip_features, val_aux_features, val_labels, train_loader_shuffle)
    
