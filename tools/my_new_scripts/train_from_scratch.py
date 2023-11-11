import os
import argparse
import copy

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--pvt_dir', type=str, default='/home/user/PCcompression/OpenPCDet', help="The directory storing OpenPCDet")
    # parser.add_argument('--rates_based', type=bool, default=True, help="Decide whether the training is based on compressed data -- need to train the data from multiple rates")
    parser.add_argument('--rates_based', action="store_true", help="Decide whether the training is based on compressed data -- need to train the data from multiple rates")
    parser.set_defaults(rates_based=False)
    parser.add_argument('--rates', type=str, default="1,2,3,4,5,6", help="The rates that need training.")
    parser.add_argument('--data_src_dir', type=str, default=None, help="The source directory of binary data. If it is rates_based, then each rate will be automatically appended to dir name")
    parser.add_argument('--non_detected_obj_dir', type=str, default='/dev/null', help="The non-detected objects directory")
    parser.add_argument('--AP_files_dir', type=str, default=None, help="The AP files directory, which will be appended by the extra tag and rates (if it is rates_based)")
    parser.add_argument('--model_cfg_path', type=str, default=None, help="The cfg file path of the model, this should be based on OpenPCDet/tools")
    parser.add_argument('--epochs', type=int, default=80, help="The number of epochs you want to train")
    parser.add_argument('--extra_tag', type=str, default=None, help="The extra tag for training, should be of the format tag1/tag2/...")
    parser.add_argument('--log_dir', type=str, default=None, help="The log directory, which will also appended by the extra tag and rates (if it is rates_based)")
    parser.add_argument('--batch_size', type=int, default=8, help="The batch size for training and final evalutation")
    

    cfgs = parser.parse_args()
    if cfgs.rates_based:
        cfgs.rates = list(map(lambda x: int(x), cfgs.rates.split(',')))
    cfgs.working_dir = os.path.join(cfgs.pvt_dir, 'tools')

    # print(cfgs)
    
    return cfgs


def train_command(cfgs):
    train_sh_path = os.path.join(cfgs.pvt_dir, 'tools/my_new_scripts/train.sh')
    os.makedirs(cfgs.log_dir, exist_ok=True)
    command = train_sh_path + ' '\
            + cfgs.data_src_dir + ' '\
            + cfgs.non_detected_obj_dir + ' '\
            + cfgs.AP_files_dir + ' '\
            + cfgs.model_cfg_path + ' '\
            + str(cfgs.epochs) + ' '\
            + cfgs.extra_tag + ' '\
            + cfgs.working_dir + ' '\
            + str(cfgs.batch_size) + ' '\
            + '{}/training_ep{}.log'.format(cfgs.log_dir, cfgs.epochs)

    # print(cfgs.data_src_dir)
    # print(command)
    os.system(command)
    
def eval_train_command(cfgs):
    eval_sh_path = os.path.join(cfgs.pvt_dir, 'tools/my_new_scripts/eval.sh')
    os.makedirs(cfgs.log_dir, exist_ok=True)
    command = eval_sh_path + ' '\
            + cfgs.data_src_dir + ' '\
            + cfgs.non_detected_obj_dir + ' '\
            + cfgs.AP_files_dir + ' '\
            + cfgs.model_cfg_path + ' '\
            + str(cfgs.epochs) + ' '\
            + cfgs.extra_tag + ' '\
            + cfgs.latest_model_path + ' '\
            + cfgs.working_dir + ' '\
            + str(cfgs.batch_size) + ' '\
            + '>> {}/training_ep{}.log 2>&1'.format(cfgs.log_dir, cfgs.epochs)
            
    os.system(command)

def train(cfgs):
    relative_model_path = cfgs.model_cfg_path.split('/')[-3:]
    relative_model_path[-1] = relative_model_path[-1][:-5]
    if cfgs.rates_based:
        for rate in cfgs.rates:
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
            
            train_command(cfgs_)
            
            cfgs_.latest_model_path = os.path.join(cfgs_.pvt_dir, 'output', *relative_model_path, cfgs_.extra_tag, 'ckpt/checkpoint_epoch_{}.pth'.format(cfgs_.epochs))
            # print(cfgs_.latest_model_path)
            eval_train_command(cfgs_)
            
    else:
        cfgs_ = copy.deepcopy(cfgs)
        cfgs_.AP_files_dir = os.path.join(cfgs_.AP_files_dir, cfgs_.extra_tag)
        cfgs_.log_dir = os.path.join(cfgs_.log_dir, cfgs_.extra_tag)
        if cfgs_.non_detected_obj_dir != '/dev/null':
            cfgs_.non_detected_obj_dir = os.path.join(cfgs_.non_detected_obj_dir, cfgs_.extra_tag)

        train_command(cfgs_)
        cfgs_.latest_model_path = os.path.join(cfgs_.pvt_dir, 'output', *relative_model_path, cfgs_.extra_tag, 'ckpt/checkpoint_epoch_{}.pth'.format(cfgs_.epochs))
        eval_train_command(cfgs_)
        


if __name__ == '__main__':
    # path = '/cfg/kitti_models/pointpillar.yaml'
    # split_path = path.split('/')[-3:]
    # split_path[-1] = split_path[-1][:-5]
    # new_path = os.path.join(*split_path)
    # print(new_path)
    
    # print(os.path.join('OpenPCDet/tools', './cfg/kitti_models/pointpillar.yaml'))
    
    train(parse_config())
    # parse_config()
    
    # print(os.path.join('/home/user/PCcompression/Results/GPCC_files/decoded_bin_for_kitti_training/octree-raht/octree_raht_lossy_lossy_no_dup', 'r01'))