#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = 4
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.max_num_splats = 3_000_000 # Stop densifying after this number of splats is reached
        self.iterations = 10_000 # [default 30_000] Each iteration corresponds to reconstructing 1 image. The number of points being optimized increases over
        self.position_lr_init = 0.00016 # [default 0.00016] Learning rate should be smaller for more extensive scenes
        self.position_lr_final = 0.0000016 # [default 0.0000016] Learning rate should be smaller for more extensive scenes
        self.position_lr_delay_mult = 0.01 # [default 0.01]
        self.position_lr_max_steps = 30_000 # [default 30_000]
        self.feature_lr = 0.0025 # [default 0.0025]
        self.opacity_lr = 0.05 # [default 0.05]
        self.scaling_lr = 0.005 # [default 0.005]
        self.rotation_lr = 0.001 # [default 0.001]
        self.percent_dense = 0.01 # [default 0.01] percent_dense * scene_extent = threshold size to determine whether to split (current is too large) or clone (current is small) gaussian
        self.lambda_dssim = 0.2 # [default 0.2] Loss = (1-lambda) * L1_loss + lambda * D-SSIM_Loss. L1 = abs(pred_pixel - true_pixel). SSIM = similarity between 2 images (luminance, contrast, structure)
        self.lambda_silhouette = 0.01 # [default 0.01] use bce loss for silhouette
        self.densification_interval = 150 # [default 100] Increase this to avoid running out of memory (how many iterations in between densifying/splitting gaussians)
        self.opacity_reset_interval = 1000 # [default 3000] Decrease all opacities (alpha) close to zero -> algo will automatically increase opacities again for important gaussians -> cull the rest
        self.remove_outliers_interval = 1000 # [default 500]
        self.densify_from_iter = 500 # [default 500] After this many iterations, start densifying
        self.densify_until_iter = int(0.6 * self.iterations) # [default 15_000] Decrease this to avoid running out of memory (after this many iterations, stop densifying)
        self.densify_grad_threshold = 0.0002 # [default 0.0002; Section 5.2: tau_pos] Increase this to avoid running out of memory. If very high, no densification will occur
        self.start_sample_pseudo = 400000 # not use
        self.end_sample_pseudo = 1000000 # not use
        self.sample_pseudo_interval = 10 # not use
        self.random_background = False
        self.pose_iterations = 4000
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
