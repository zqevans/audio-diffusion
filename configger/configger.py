from pathlib import Path
from ast import literal_eval 
import argparse
import configparser
import wandb

def arg_eval(value):
    "this just packages some type checking for parsing args"
    try: 
        val = literal_eval(str(value))
    except (SyntaxError, ValueError, AssertionError):
        val = value
    return val

def read_defaults(defaults_file='defaults.ini'):
    "read the defaults file, setup defaults dict"
    configp = configparser.ConfigParser()
    configp.read(defaults_file)
    defaults = dict(configp.items('DEFAULTS'))
    with open(defaults_file) as f:
        defaults_text = f.readlines()
    return defaults, defaults_text


def setup_args(defaults, defaults_text=''):
    """combine defaults from .ini file and add parseargs arguments, 
        with help pull from .ini"""
    p = argparse.ArgumentParser()  
    p.add_argument('--wandb-config', required=False,  
                   help='wandb url to pull config from')
    p.add_argument('--training-dir', type=Path, required=False,
                   help='training data directory')
    p.add_argument('--name', type=str, required=False,
                   help='name of the run')

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

    return args
    

def pull_wandb_config(wandb_config, defaults):
    """overwrites parts of args using wandb config info 
    wandb_config is the url of one of your runs"""
    api = wandb.Api()  # might get prompted for api key login the first time
    splits = wandb_config.split('/')
    entity, project, run_id = splits[3], splits[4], splits[-1].split('?')[0]
    run = api.run(f"{entity}/{project}/{run_id}")
    for key, value in run.config.items():
        defaults[key] = arg_eval(value)
    return defaults


def get_all_args():
    # Config setup. Order of preference will be: 
    #   1. Default settings are in defaults.ini file
    #   2. if --wandb-config is given, pull config from wandb to override defaults
    #   3. Any new command-line arguments override whatever was set earlier
    defaults, defaults_text = read_defaults()
    args = setup_args(defaults, defaults_text=defaults_text)  # 1.
    if args.wandb_config is not None:
        defaults = pull_wandb_config(args.wandb_config, defaults) # 2.
    args = setup_args(defaults, defaults_text=defaults_text) # 3. this time cmd-line overrides what's there
    return args


def wandb_log_config(wandb_logger, args): 
    "save config to wandb (for possible retrieval later)"
    if hasattr(wandb_logger.experiment.config, 'update'): 
        wandb_logger.experiment.config.update(args)
