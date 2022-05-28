# -*- coding: utf-8 -*-
__author__ = 'S.H. Hawley'

"""
Routines for easily keeping track of & archiving run configurations.
Supports config (.ini) files, pulling previous configs from WandB, 
and overrides with command-line options.
"""

from pathlib import Path
from ast import literal_eval 
import argparse
import configparser
import wandb
import sys

DEFAULTS_FILE = 'defaults.ini'

def arg_eval(value):
    "this just packages some type checking for parsing args"
    try: 
        val = literal_eval(str(value))
    except (SyntaxError, ValueError, AssertionError):
        val = value
    return val


def read_defaults(defaults_file=DEFAULTS_FILE):
    "read the defaults file, setup defaults dict"
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--config-file', required=False, default=defaults_file,
        help='name of local configuration (.ini) file')
    config_file = p.parse_known_args()[0].config_file
    configp = configparser.ConfigParser()
    configp.read(config_file)
    defaults = dict(configp.items('DEFAULTS'))
    with open(config_file) as f:
        defaults_text = f.readlines()
    return defaults, defaults_text


def setup_args(defaults, defaults_text=''):
    """combine defaults from .ini file and add parseargs arguments, 
        with help pull from .ini"""
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)  
    p.add_argument('--wandb-config', required=False,  
                   help='wandb url to pull config from')
    p.add_argument('--training-dir', type=Path, required=False,
                   help='training data directory')
    p.add_argument('--name', type=str, required=False,
                   help='name of the run')
    p.add_argument('--config-file', required=False, default=DEFAULTS_FILE, #added so it appears on -h list
        help='name of local configuration (.ini) file')

    # add other command-line args using defaults
    for key, value in defaults.items():
        if (key in ['training_dir', 'name', 'wandb_config']): break
        help = ""
        for i in range(len(defaults_text)):  # get the help string from defaults_text
            if key in defaults_text[i]:
                help = defaults_text[i-1].replace('# ','')
        argname = '--'+key.replace('_','-')
        val = arg_eval(value)
        p.add_argument(argname, default=val, type=type(val), help=help)

    args = p.parse_args() 

    if (args.training_dir is None) and ('training_dir' in defaults):
        args.training_dir = Path(defaults['training_dir'])
    if (args.name is None) and ('name' in defaults):
        args.name = Path(defaults['name'])

    if None in [args.training_dir, args.name]:
        print("Required arguments: --training_dir <dir> --name <name>")
        sys.exit(1)
        
    return args
    

def pull_wandb_config(wandb_config, defaults):
    """overwrites parts of defaults using wandb config info 
       wandb_config is the url of one of your runs"""
    api = wandb.Api()  # might get prompted for api key login the first time
    splits = wandb_config.split('/')
    entity, project, run_id = splits[3], splits[4], splits[-1].split('?')[0]
    run = api.run(f"{entity}/{project}/{run_id}")
    for key, value in run.config.items():
        defaults[key] = arg_eval(value)
    return defaults


def get_all_args():
    " Config setup."
    #   1. Default settings are in defaults ini (or some other config) file
    defaults, defaults_text = read_defaults()
    args = setup_args(defaults, defaults_text=defaults_text)  

    #   2. if --wandb-config is given, pull config from wandb to override defaults
    if args.wandb_config is not None:
        defaults = pull_wandb_config(args.wandb_config, defaults) # 2.

    #   3. Any new command-line arguments override whatever was set earlier
    args = setup_args(defaults, defaults_text=defaults_text) # 3. this time cmd-line overrides what's there

    return args


def wandb_log_config(wandb_logger, args): 
    "save config to wandb (for possible retrieval later)"
    if hasattr(wandb_logger.experiment.config, 'update'): 
        wandb_logger.experiment.config.update(args)
