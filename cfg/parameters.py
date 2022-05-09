import math
import os
import shutil
import datetime
import yaml

def create_params(settings_yaml, mode='run'):
    setting_dict = {'train': TrainSetting, 'test': TrainSetting, 'run': RunSetting}
    settings = setting_dict.get(mode, None)

    if settings is None:
        raise IOError("Unidentified Settings")

    settings = settings(settings_yaml)
    if mode == 'run' or mode == 'test':
        settings.freeze_backbone = True
    return settings


class Settings:
    def __init__(self, settings_yaml, generate_log=True):
        assert os.path.isfile(settings_yaml), settings_yaml

        with open(settings_yaml, 'r') as stream:
            settings = yaml.safe_load(stream)

            # Input mode
            self.use_rgb = settings['use_rgb']
            self.use_depth = settings['use_depth']
            self.img_width = settings['img_width']
            self.img_height = settings['img_height']
            self.future_time = settings['future_time']
            self.velocity_frame = settings['velocity_frame']
            self.input_size = (self.img_height, self.img_width, 3*self.use_rgb + 3*self.use_depth)

            # Output config
            self.state_dim = settings['state_dim']
            self.out_seq_len = settings['out_seq_len']
            self.modes = settings['modes']
            self.seq_len = settings['seq_len']

            # Net Parameters
            net_params = settings['net_params']
            self.use_bias = net_params['use_bias']
            self.g_im = net_params['g_im']
            self.g_ss = net_params['g_ss']
            self.g_pl = net_params['g_pl']
            self.epsilon = net_params['epsilon']
            self.lambda_1 = net_params['lambda_1']
            self.lambda_2 = net_params['lambda_2']
            self.devices = net_params['devices']
            self.maximum_depth_value = net_params['maximum_depth_value']
        
            # Checkpoint
            checkpoint = settings['checkpoint']
            self.resume_training = checkpoint['resume_training']
            assert isinstance(self.resume_training, bool)
            self.resume_ckpt_file = checkpoint['resume_file']

            # Save a Copy of the Parameters for Reproducibility
            log_root = settings['log_dir']
            if not log_root == '' and generate_log:
                current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                self.log_dir = os.path.join(log_root, current_time)
                os.makedirs(self.log_dir)
                net_file = "./lib/net.py"
                print(net_file)
                assert os.path.isfile(net_file)
                shutil.copy(net_file, self.log_dir)
                shutil.copy(settings_yaml, self.log_dir)

    def add_flags(self):
        self._add_flags()

    def _add_flags(self):
        raise NotImplementedError


class TrainSetting(Settings):
    def __init__(self, settings_yaml):
        super(TrainSetting, self).__init__(settings_yaml, generate_log=True)
        self.settings_yaml = settings_yaml
        self.add_flags()

    def add_flags(self):
        with open(self.settings_yaml, 'r') as stream:
            settings = yaml.safe_load(stream)

            # Train Time
            train_conf = settings['train']
            self.training_epochs = train_conf['max_training_epochs']
            self.data_save_freq = train_conf['data_save_freq']
            self.batch_size = train_conf['batch_size']
            self.summary_freq = train_conf['summary_freq']
            self.train_dir = train_conf['train_dir']

            if not os.path.isdir(self.train_dir):
                os.makedirs(self.train_dir)

            self.val_dir = train_conf['val_dir']
            self.test_dir = train_conf['test_dir']
            self.top_trajectories = train_conf['top_trajectories']
            self.freeze_backend = train_conf['freeze_backend']

            self.save_every_n_epochs = train_conf['save_every_n_epochs']
            self.ref_frame = train_conf['ref_frame']
            assert (self.ref_frame == 'bf') or (self.ref_frame == 'wf')

            self.img_rootname = train_conf['img_rootname']
            self.img_format = train_conf['img_format']
            self.data_format = train_conf['data_format']
            self.fu = train_conf['fu']
            self.fv = train_conf['fv']
            self.cu = train_conf['cu']
            self.cv = train_conf['cv']

class RunSetting(Settings):
    def __init__(self, settings_yaml):
        super(RunSetting, self).__init__(settings_yaml, generate_log=True)
        self.settings_yaml = settings_yaml
        self.add_flags()

    def add_flags(self):
        with open(self.settings_yaml, 'r') as stream:
            settings = yaml.safe_load(stream)

        # Test Time
            run_conf = settings['run']
            self.netupdate_freq = run_conf['netupdate_freq']
            self.setpoint_freq = run_conf['setpoint_freq']
            self.odometry_topic = run_conf['odometry_topic']
            self.rgb_topic = run_conf['rgb_topic']
            self.depth_topic = run_conf['depth_topic']
            self.traj_topic = run_conf['traj_topic']

        # Train Time
            train_conf = settings['train']
            self.freeze_backend = train_conf['freeze_backend']
            self.ref_frame = train_conf['ref_frame']
            assert (self.ref_frame == 'bf') or (self.ref_frame == 'wf')
