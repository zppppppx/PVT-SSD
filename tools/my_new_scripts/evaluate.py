import os
import argparse
import copy

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--pvt_dir', type=str, default='/home/user/PCcompression/PVT-SSD', help="The directory storing PVT-SSD")
    parser.add_argument('--model_dir', type=str, default='/home/user/PCcompression/PVT-SSD/output', help="Where the model are saved, will be appended by the extra tag including the rates")
    parser.add_argument('--rates_based', action="store_true", help="Decide whether the training is based on compressed data -- need to train the data from multiple rates")
    parser.set_defaults(rates_based=False)
    parser.add_argument('--training_based', action="store_true", help="Add this flag, if you want to evaluate the models which were trained on different compressed data.")
    parser.set_defaults(training_based=False)
    parser.add_argument('--rates', type=str, default="1,2,3,4,5,6", help="The rates that need evaluation.")
    parser.add_argument('--data_src_dir', type=str, default=None, help="The source directory of binary data. If it is rates_based, then each rate will be automatically appended to dir name")
    parser.add_argument('--non_detected_obj_dir', type=str, default='/dev/null', help="The non-detected objects directory")
    parser.add_argument('--AP_files_dir', type=str, default=None, help="The AP files directory, which will be appended by the extra tag and rates (if it is rates_based)")
    parser.add_argument('--model_cfg_path', type=str, default=None, help="The cfg file path of the model, this should be based on PVT-SSD/tools")
    parser.add_argument('--epoch_list', type=str, default=None, help="The epochs for each model you want to evaluate with")
    parser.add_argument('--model_path', type=str, default=None, help="The model's path used for evaluation")
    parser.add_argument('--extra_tag', type=str, default=None, help="The extra tag for training, should be of the format tag1/tag2/...")
    parser.add_argument('--log_dir', type=str, default=None, help="The log directory, which will also appended by the extra tag and rates (if it is rates_based)")
    parser.add_argument('--batch_size', type=int, default=8, help="The batch size for training and final evalutation")
    

    cfgs = parser.parse_args()
    if cfgs.rates_based:
        cfgs.rates = list(map(lambda x: int(x), cfgs.rates.split(',')))
        
    if cfgs.training_based:
        cfgs.epoch_list = list(map(lambda x: int(x), cfgs.epoch_list.split(',')))
        
    cfgs.working_dir = os.path.join(cfgs.pvt_dir, 'tools')
    cfgs.flag = "pre"
    # print(cfgs)
    
    return cfgs

    
def eval_command(cfgs):
    eval_sh_path = os.path.join(cfgs.pvt_dir, 'tools/my_new_scripts/eval.sh')
    os.makedirs(cfgs.log_dir, exist_ok=True)
    log_file = '{}/evaluate_{}.log 2>&1'.format(cfgs.log_dir, cfgs.flag)
    
    command = eval_sh_path + ' '\
            + cfgs.data_src_dir + ' '\
            + cfgs.non_detected_obj_dir + ' '\
            + cfgs.AP_files_dir + ' '\
            + cfgs.model_cfg_path + ' '\
            + str(80) + ' '\
            + cfgs.extra_tag + ' '\
            + cfgs.model_path + ' '\
            + cfgs.working_dir + ' '\
            + str(cfgs.batch_size) + ' '\
            + log_file
            
    os.system(command)

# def train(cfgs):
#     relative_model_path = cfgs.model_cfg_path.split('/')[-3:]
#     relative_model_path[-1] = relative_model_path[-1][:-5]
#     if cfgs.rates_based:
#         for rate in cfgs.rates:
#             cfgs_ = copy.deepcopy(cfgs)
#             # print(cfgs_)
#             # print(os.path.join(cfgs_.AP_files_dir, cfgs_.extra_tag))
#             cfgs_.extra_tag = cfgs.extra_tag + '/r{:02d}'.format(rate)
#             cfgs_.AP_files_dir = os.path.join(cfgs_.AP_files_dir, cfgs_.extra_tag)
#             cfgs_.data_src_dir = os.path.join(cfgs_.data_src_dir, 'r{:02d}'.format(rate))
#             cfgs_.log_dir = os.path.join(cfgs_.log_dir, cfgs_.extra_tag)
#             # print(cfgs_)
#             if cfgs_.non_detected_obj_dir != '/dev/null':
#                 cfgs_.non_detected_obj_dir = os.path.join(cfgs_.non_detected_obj_dir, cfgs_.extra_tag)
            
#             train_command(cfgs_)
            
#             cfgs_.latest_model_path = os.path.join(cfgs_.pvt_dir, 'output', *relative_model_path, cfgs_.extra_tag, 'ckpt/checkpoint_epoch_{}.pth'.format(cfgs_.epochs))
#             # print(cfgs_.latest_model_path)
#             eval_train_command(cfgs_)
            
#     else:
#         cfgs_ = copy.deepcopy(cfgs)
#         cfgs_.AP_files_dir = os.path.join(cfgs_.AP_files_dir, cfgs_.extra_tag)
#         cfgs_.log_dir = os.path.join(cfgs_.log_dir, cfgs_.extra_tag)
#         if cfgs_.non_detected_obj_dir != '/dev/null':
#             cfgs_.non_detected_obj_dir = os.path.join(cfgs_.non_detected_obj_dir, cfgs_.extra_tag)

#         train_command(cfgs_)
#         cfgs_.latest_model_path = os.path.join(cfgs_.pvt_dir, 'output', *relative_model_path, cfgs_.extra_tag, 'ckpt/checkpoint_epoch_{}.pth'.format(cfgs_.epochs))
#         eval_train_command(cfgs_)
        
        
def evaluate(cfgs):
    relative_model_path = cfgs.model_cfg_path.split('/')[-3:]
    relative_model_path[-1] = relative_model_path[-1][:-5]

    if cfgs.rates_based:
        for idx, rate in enumerate(cfgs.rates):
            cfgs_ = copy.deepcopy(cfgs)
            # print(cfgs_)
            # print(os.path.join(cfgs_.AP_files_dir, cfgs_.extra_tag))
            cfgs_.extra_tag = cfgs.extra_tag + '/r{:02d}'.format(rate)
            cfgs_.AP_files_dir = os.path.join(cfgs_.AP_files_dir, cfgs_.extra_tag)
            cfgs_.data_src_dir = os.path.join(cfgs_.data_src_dir, 'r{:02d}'.format(rate))
            cfgs_.log_dir = os.path.join(cfgs_.log_dir, cfgs_.extra_tag)
            
            # print(cfgs_)
            if cfgs_.non_detected_obj_dir != '/dev/null':
                cfgs_.non_detected_obj_dir = os.path.join(cfgs_.non_detected_obj_dir, cfgs_.extra_tag)
            
            
            if cfgs_.training_based:
                cfgs_.model_path = os.path.join(cfgs_.model_dir, *relative_model_path, cfgs_.extra_tag, 'ckpt/checkpoint_epoch_{}.pth'.format(cfgs_.epoch_list[idx]))
                cfgs_.flag = "ep{}".format(cfgs_.epoch_list[idx])
            # print(cfgs_.latest_model_path)
            eval_command(cfgs_)
    else:
        cfgs_ = copy.deepcopy(cfgs)
        cfgs_.AP_files_dir = os.path.join(cfgs_.AP_files_dir, cfgs_.extra_tag)
        cfgs_.log_dir = os.path.join(cfgs_.log_dir, cfgs_.extra_tag)
        if cfgs_.non_detected_obj_dir != '/dev/null':
            cfgs_.non_detected_obj_dir = os.path.join(cfgs_.non_detected_obj_dir, cfgs_.extra_tag)
        if cfgs_.training_based:
            cfgs_.model_path = os.path.join(cfgs_.model_dir, *relative_model_path, cfgs_.extra_tag, 'ckpt/checkpoint_epoch_{}.pth'.format(cfgs_.epoch_list[idx]))
            cfgs_.flag = "ep{}".format(cfgs_.epoch_list[idx])
            
        eval_command(cfgs_)
        
if __name__ == '__main__':
    # path = '/cfg/kitti_models/pointpillar.yaml'
    # split_path = path.split('/')[-3:]
    # split_path[-1] = split_path[-1][:-5]
    # new_path = os.path.join(*split_path)
    # print(new_path)
    
    # print(os.path.join('PVT-SSD/tools', './cfg/kitti_models/pointpillar.yaml'))
    
    evaluate(parse_config())
    # parse_config()
    
    # print(os.path.join('/home/user/PCcompression/Results/GPCC_files/decoded_bin_for_kitti_training/octree-raht/octree_raht_lossy_lossy_no_dup', 'r01'))