import os
import sys
import threading
from sacred import Experiment
from sacred.config import save_config_file
from pdpg.modules import utils
from os.path import join
from pdpg import program as program
from pdpg.algorithms import pdpg
import torch
import sqlite3

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

ctx = threading.local()
ex = Experiment('PDPG')
ctx.ex = ex


@ex.config
def cfg():
    # Gym environment used
    environment = 'HalfCheetah-v2'
    # Logging directory
    logdir = ''
    root = join(logdir, "pdpg")
    val_folder = join(root, 'logs', 'val')
    train_folder = join(root, 'logs', 'train')
    save_folder = join(root, 'save')
    log_freq = 100
    resume = ''
    evaluate = False
    gradient_checking = False
    n_eval_episodes = 10
    gpu = ''
    warmup = int(1E4)  # How many time steps purely random policy is run for
    ex.add_config(pdpg.config)
    seed = 300
    offline = ''
    db = ''
    wd = 0.0


@ex.capture
def init(_log, root, save_folder):
    ctx.iter = 0
    ctx.opt = utils.init_opt(ctx)
    opt = ctx.opt
    gpu = opt['gpu']
    ctx.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu") if gpu != '' else "cpu"
    if not opt['resume']:
        utils.build_filename(ctx, ('model', 'environment'))
        save_folder = join(save_folder, opt['filename'])
        ctx.opt['save_folder'] = save_folder

        if opt['db']:
            conn = sqlite3.connect(opt['db'])
            query = f"insert into executions (folder) values ('{save_folder}')"
            utils.sqlite_query(conn, query)
            conn.close()

        if not os.path.isdir(root):
            try:
                os.makedirs(root)
            except FileExistsError:
                _log.error("Cannot create output dir: A file exists with the same name specified in the path")
                sys.exit(1)

        if not os.path.isdir(save_folder):
            try:
                os.makedirs(save_folder)
            except FileExistsError:
                _log.error("Cannot create output dir: A file exists with the same name specified in the path")
                sys.exit(1)
        save_config_file(ex.current_run.config, join(save_folder, 'config.json'))


@ex.automain
def main():
    init()
    program.run(ctx)
