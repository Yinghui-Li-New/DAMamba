import torch
import torch.nn as nn
import numpy as np
import os
import json
import time
import torch.nn.functional as F
import open3d as o3d
from tools import builder
from utils import dist_utils
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.realtime_render import *
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2,UnidirectionalChamferDistance,DirectedHausdorffDistance
from utils import loss_util
from utils.AverageMeter import AverageMeter



def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()
    UCD_distance = UnidirectionalChamferDistance()
    UHD_distance = DirectedHausdorffDistance()
    completion_loss = loss_util.Completionloss(loss_func=config.consider_metric)
    # build dataset CRN train and CRN test dataset
    train_sampler, train_dataloader = builder.virtual_dataset_builder(args,  config.dataset.train)  #CRN训练集 Chair 79
    config.dataset.test._base_.SPLIT= 'test'
    _, test_dataloader = builder.virtual_dataset_builder(args,config.dataset.test) #CRN测试集
    # build real data
    config.dataset.train._base_.SPLIT = 'train'
    _,real_train_dataloader = builder.real_dataset_builder(args, config.dataset.train) #3D FUTURE训练
    config.dataset.test._base_.SPLIT = 'test'
    real_test_sampler, real_test_dataloader = builder.real_dataset_builder(args,config.dataset.test) #3D FUTURE测试集

    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)
    
    # parameter setting
    start_epoch = 0
    best_metrics = None
    metrics = None

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger=logger)
        best_metrics = Metrics(best_metric)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger=logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(base_model,
                                                         device_ids=[args.local_rank % torch.cuda.device_count()],
                                                         find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)

    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        avg_meter_loss = AverageMeter(['loss_partial', 'loss_pc', 'loss_p1', 'loss_p2', 'loss_p3'])
        num_iter = 0
        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        train_dataloader_iter = iter(train_dataloader) #CRN
        real_train_dataloader_iter = iter(real_train_dataloader)    #3D FUTURE
        len_train_dataloader = len(train_dataloader)
        len_real_train_dataloader = len(real_train_dataloader)
        max_len = max(len_train_dataloader, len_real_train_dataloader)
        for idx in range(max_len):
            #load source data
            try:
                source_data = next(train_dataloader_iter)
            except StopIteration:
                train_dataloader_iter =iter(train_dataloader)
                source_data = next(train_dataloader_iter)
            #load target data
            try:
                target_data = next(real_train_dataloader_iter)
            except StopIteration:
                real_train_dataloader_iter = iter(real_train_dataloader) 
                target_data = next(real_train_dataloader_iter)

            data_time.update(time.time() - batch_start_time)
            num_iter += 1
            n_itr = epoch * n_batches + idx

            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train._base_.N_POINTS
            source_dataset_name = config.dataset.train._base_.NAME
            target_dataset_name = config.dataset.train.real_dataset

            if source_dataset_name == 'CRNShapeNet' and target_dataset_name in ['3D_FUTURE','ModelNet']:
                source_gt, source_partial, source_index = source_data
                target_gt, target_partial, _ = target_data
            
            elif source_dataset_name == 'CRNShapeNet' and target_dataset_name in ['MatterPort', 'ScanNet','KITTI']:
                source_gt, source_partial, source_index = source_data
                target_partial, _ = target_data
            
            else:
                raise NotImplementedError(f'Train phase do not support {source_dataset_name}')


            if source_dataset_name == 'CRNShapeNet' and target_dataset_name in ['3D_FUTURE','ModelNet']:
                source_gt = source_gt.squeeze(0).cuda()
                source_partial = source_partial.squeeze(0).cuda()
                if config.dataset.train._base_.CLASS_CHOICE in ['lamp','table']: 
                       source_partial, _, _, _ = partial_render_batch(source_gt, source_partial) 
                target_partial = target_partial.cuda()
                rebuild_points, loss_sp, loss_ch= base_model(source_partial, target_partial)        
                loss_total, losses = completion_loss.get_loss(rebuild_points, source_partial, source_gt)

                loss_total = loss_total  +  0.1 * loss_sp + 0.1 * loss_ch 
            elif source_dataset_name == 'CRNShapeNet' and target_dataset_name in ['KITTI','MatterPort','ScanNet']:

                source_gt = source_gt.cuda() 
                source_partial = source_partial.squeeze().cuda() 
                target_partial = target_partial.cuda()      
                rebuild_points, loss_sp, loss_ch= base_model(source_partial, target_partial)        
                loss_total, losses = completion_loss.get_loss(rebuild_points, source_partial, source_gt)

                loss_total = loss_total +  0.1 * loss_sp + 0.1 * loss_ch 
            try:
                loss_total.backward()
            except:
                loss_total = loss_total.mean()
                loss_total.backward()

            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                losses.update([loss.item() * 10000])
            else:
                avg_meter_loss.update(losses)
                

            if args.distributed:
                torch.cuda.synchronize()

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:

                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f | loss_total = %.6f | loss_sp = %.6f|loss_ch = %.6f' %
                        (epoch, config.max_epoch, idx + 1, max_len, batch_time.val(), data_time.val(),
                        ['%.4f' % l for l in losses], optimizer.param_groups[0]['lr'], loss_total.item(),
                        loss_sp.item(), loss_ch.item()), logger=logger)
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()


        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
                  (epoch, epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses],
                   optimizer.param_groups[0]['lr']), logger=logger)

        if epoch % args.val_freq == 0:
             # Validate the current model
            metrics = validate(base_model, real_test_dataloader, epoch, 
                                    ChamferDisL1, ChamferDisL2, UCD_distance, UHD_distance, val_writer, args, config, logger=logger)
        
             # Save ckeckpoints
            if  metrics.better_than(best_metrics):
                    best_metrics = metrics
                    builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt_source_best', args, logger = logger)
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger=logger)


    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()

def validate(base_model,real_test_dataloader, epoch, ChamferDisL1, ChamferDisL2, UCD_distance, UHD_distance, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode
    test_losses = AverageMeter(['Rebuild_Loss_L1','Rebuild_Loss_L2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(real_test_dataloader) # bs is 1

    interval =  n_samples // 10
    source_dataset_name = config.dataset.train._base_.NAME 
    target_dataset_name = config.dataset.train.real_dataset
    with torch.no_grad():
        if source_dataset_name == 'CRNShapeNet' and target_dataset_name in ['3D_FUTURE', 'ModelNet']:
            for idx, data in enumerate(real_test_dataloader):

                target_gt,target_partial, _ = data 
                if args.use_gpu:
                    target_partial = target_partial.cuda()
                    target_gt = target_gt.cuda()

                rebuild_points,_,_= base_model(target_partial)

                rebuild_points = rebuild_points[-1]
                
                if idx <10:
                    visualize_point_cloud_batch(rebuild_points, f"pred_point_{idx}")
                    visualize_point_cloud_batch(target_gt, f"gt_{idx}")
                    visualize_point_cloud_batch(target_partial, f"partial_{idx}")
                
                
                rebuild_loss_l1 =  ChamferDisL1(rebuild_points, target_gt)
                rebuild_loss_l2 =  ChamferDisL2(rebuild_points, target_gt)

                test_losses.update([ rebuild_loss_l1.item() * 10000, rebuild_loss_l2.item() * 10000])

                _metrics = Metrics.get(rebuild_points, target_gt)
                if args.distributed:
                    _metrics = [dist_utils.reduce_tensor(_metric, args).item() for _metric in _metrics]
                else:
                    _metrics = [_metric.item() for _metric in _metrics]

                test_metrics.update(_metrics)


                if (idx + 1) % interval == 0:
                    print_log('Test[%d/%d] Losses = %s Metrics = %s' %
                        (idx + 1, n_samples, ['%.4f' % l for l in test_losses.val()],
                        ['%.4f' % m for m in _metrics]), logger=logger)
        elif source_dataset_name == 'CRNShapeNet' and target_dataset_name in ['MatterPort', 'ScanNet','KITTI']:
            for idx, data in enumerate(real_test_dataloader):

                target_partial, _ = data   #[1,2048,3],[1,2048,3]
                if args.use_gpu:
                    target_partial = target_partial.float().cuda()
                   

                # Calculate losses for target
                rebuild_points,_,_= base_model(target_partial)#[1, 2048,3]
                rebuild_dense_points = rebuild_points[-1]
                uhd_rebuild_loss =  UHD_distance(target_partial.permute([0,2,1]),rebuild_dense_points.permute([0,2,1]))
                ucd_rebuild_loss =  UCD_distance(target_partial, rebuild_dense_points)

                test_losses.update([ucd_rebuild_loss.item()*10000, uhd_rebuild_loss.item()*100])

                _metrics = Metrics.get(target_partial,rebuild_dense_points)
                if args.distributed:
                    _metrics = [dist_utils.reduce_tensor(_metric, args).item() for _metric in _metrics]
                else:
                    _metrics = [_metric.item() for _metric in _metrics]

                test_metrics.update(_metrics)
            
            # ... (可视化部分的代码，如果不需要可以省略)

                if (idx + 1) % interval == 0:
                    print_log('Test[%d/%d] Losses = %s Metrics = %s' %
                        (idx + 1, n_samples, ['%.6f' % l for l in test_losses.val()],
                        ['%.6f' % m for m in _metrics]), logger=logger)

        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[Validation] EPOCH: %d  Metrics = %s' % (epoch, ['%.4f' % m for m in test_metrics.avg()]), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()
    
    # Print testing results
    print_log('============================ TEST RESULTS ============================', logger=logger)
    msg = '\t\t'
    
    for metric in test_metrics.items:
        msg += metric + '\t'
    print_log(msg, logger=logger)

    msg=''
    msg += 'Overall\t'
    for value in test_metrics.avg():
        msg += '%.6f \t' % value
    print_log(msg, logger=logger)

    # Add testing results to TensorBoard
    """
    """
    if val_writer is not None:
        val_writer.add_scalar('Loss/Epoch/Sparse_L1', test_losses.avg(0), epoch)
        val_writer.add_scalar('Loss/Epoch/Sparse_L2', test_losses.avg(1), epoch)

        for i, metric in enumerate(test_metrics.items):
            val_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch)

    if source_dataset_name == 'CRNShapeNet' and target_dataset_name in ['3D_FUTURE', 'ModelNet']:

        return Metrics(config.consider_metric_2, test_metrics.avg())
    elif source_dataset_name == 'CRNShapeNet' and target_dataset_name in ['MatterPort', 'ScanNet','KITTI']:
        return Metrics(config.consider_metric_3, test_metrics.avg())

def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    config.dataset.test._base_.SPLIT = 'test'
    _, test_dataloader = builder.real_dataset_builder(args,config.dataset.test)
    category_metrics = dict()
    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger = logger)
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP    
    if args.distributed:
        raise NotImplementedError()

    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()
    UnidirectionalCD = UnidirectionalChamferDistance()

    test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, UnidirectionalCD, args, config, logger=logger)

def test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, UnidirectionlCD ,args, config, logger = None):

    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1

    with torch.no_grad():
        for idx,data in enumerate(test_dataloader):
            
            npoints = config.dataset.test._base_.N_POINTS
            dataset_name = config.dataset.test.real_dataset
            if dataset_name in ['3D_FUTURE','ModelNet']:
                partial = data[1].cuda()
                gt = data[0].cuda()

                rebuild_points,_, _= base_model(partial)

                dense_points = rebuild_points[-1]
                if idx < 500:
                    visualize_point_cloud_batch(dense_points, f"pred_point_{idx}")
                    visualize_point_cloud_batch(gt, f"gt_{idx}")
                    visualize_point_cloud_batch(partial, f"partial_{idx}")

                dense_loss_l1 =  ChamferDisL1(dense_points, gt)
                dense_loss_l2 =  ChamferDisL2(dense_points, gt)

                test_losses.update([dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

                _metrics = Metrics.get(dense_points, gt, require_emd=True)
                test_metrics.update(_metrics)
            elif dataset_name in ['KITTI','ScanNet','MatterPort']:
                partial = data[0].cuda()
                rebuild_points,_, _= base_model(partial)
                dense_points= rebuild_points[-1]
                if idx < 200:
                    visualize_point_cloud_batch(dense_points, f"pred_point_{idx}")
                    visualize_point_cloud_batch(partial, f"partial_{idx}")

                ucd_loss = UnidirectionlCD(partial, dense_points)

                _metrics = Metrics.get(partial, dense_points, require_emd=True)
                test_metrics.update(_metrics)


            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            if (idx+1) % 100 == 0:
                print_log('Test[%d/%d]  Losses = %s Metrics = %s' %
                            (idx + 1, n_samples,  ['%.4f' % l for l in test_losses.val()], 
                            ['%.6f' % m for m in _metrics]), logger=logger)
        if dataset_name == 'KITTI':
            return
        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[TEST] Metrics = %s' % (['%.6f' % m for m in test_metrics.avg()]), logger=logger)

     

    # Print testing results
    print_log('============================ TEST RESULTS ============================',logger=logger)

    msg =''
    for metric in test_metrics.items:
        msg += metric + '\t'
    print_log(msg, logger=logger)
    msg = ''
    msg += 'Overall \t\t'
    for value in test_metrics.avg():
        msg += '%.6f \t' % value
    print_log(msg, logger=logger)
    return 

def visualize_point_cloud_batch(batch_points, name):


    output_dir = "point_virtualization"
    os.makedirs(output_dir, exist_ok=True)
    if isinstance(batch_points, torch.Tensor):
        batch_points = batch_points.detach().cpu().double().numpy()
    elif isinstance(batch_points, np.ndarray):
        batch_points = batch_points.astype(np.float64).copy()

    if batch_points.ndim == 3 and batch_points.shape[2] == 3:

        for i, points in enumerate(batch_points):
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError("Must be 3D Coordinates")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            # 构建文件名并保存点云为 PLY 文件
            filename = os.path.join(output_dir, f"point_cloud_{i}_{name}.ply")
            o3d.io.write_point_cloud(filename, pcd)