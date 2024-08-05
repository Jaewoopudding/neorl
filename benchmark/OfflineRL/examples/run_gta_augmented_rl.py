import fire
import numpy as np 
from loguru import logger

from offlinerl.algo import algo_select
from offlinerl.data import load_data_from_neorl
from offlinerl.evaluation import get_defalut_callback, OnlineCallBackFunction
from offlinerl.data.neorl import SampleBatch


def merge_dictionary(list_of_Dict):
    merged_data = {}

    for d in list_of_Dict:
        for k, v in d.items():
            if k not in ['observations', 'next_observations', 'actions', 'rewards', 'terminals', 'values', 'infos/goal']:
                continue
            else:
                if k not in merged_data.keys():
                    merged_data[k] = [v]
                else:
                    merged_data[k].append(v)

    for k, v in merged_data.items():
        merged_data[k] = np.concatenate(merged_data[k])

    return merged_data

def load_gta_data(path):
    dataset = np.load(path, allow_pickle=True)['data'].squeeze()
    dataset = merge_dictionary([*dataset])
    buffer = SampleBatch(
        obs = dataset['observations'],
        obs_next = dataset['next_observations'],
        act = dataset['actions'],
        rew = dataset['rewards'],
        done = dataset['terminals'],
    )
    
    logger.info('gta obs shape: {}', buffer.obs.shape)
    logger.info('gta obs_next shape: {}', buffer.obs_next.shape)
    logger.info('gta act shape: {}', buffer.act.shape)
    logger.info('gta rew shape: {}', buffer.rew.shape)
    logger.info('gta done shape: {}', buffer.done.shape)
    logger.info('gta Episode reward: {}', buffer.rew.sum() /np.sum(buffer.done) )
    logger.info('gta Number of terminals on: {}', np.sum(buffer.done))

    return buffer



def run_algo(**kwargs):
    # kwargs = {
    #     'algo_name':'cql',
    #     'exp_name':'halfcheetah',
    #     'task': 'HalfCheetah-v3', 
    #     'task_data_type': 'low',
    #     'task_train_num': 100
    # }
    print(kwargs)
    algo_init_fn, algo_trainer_obj, algo_config = algo_select(kwargs)
    train_buffer, val_buffer = load_data_from_neorl(algo_config["task"], algo_config["task_data_type"], algo_config["task_train_num"])
    
    if kwargs['gta_path'] is not None:
        logger.info("==================<GTA Augmentation>==================")
        gta_buffer = load_gta_data(kwargs['gta_path'])
    
    algo_init = algo_init_fn(algo_config)
    
    # neoRL interface
    # done : (1000000, 1)
    # obs : (1000000, 18)
    # obs_next: (1000000, 18)
    # act: (1000000, 6)
    # rew: (1000000, 1)
    
    
    algo_trainer = algo_trainer_obj(algo_init, algo_config)
    callback = OnlineCallBackFunction()
    callback.initialize(train_buffer=train_buffer, val_buffer=val_buffer, task=algo_config["task"])

    algo_trainer.train(train_buffer, val_buffer, callback_fn=callback)

if __name__ == "__main__":
    fire.Fire(run_algo)
    