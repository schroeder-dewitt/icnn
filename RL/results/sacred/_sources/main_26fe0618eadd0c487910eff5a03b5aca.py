#!/usr/bin/env python3
# Code from Repo SimonRamstedt/ddpg
# Heavily modified
from __future__ import division, print_function, unicode_literals

import os
import pprint
import json

import gym
import numpy as np
import tensorflow as tf

import agent
import normalized_env
import dill
import runtime_env

import time

import setproctitle




class Experiment(object):

    def run(self, _logger):
        self.train_timestep = 0
        self.test_timestep = 0

        # create normal
        self.env = normalized_env.make_normalized_env(gym.make(FLAGS.env))
        self.env.monitor = gym.wrappers.Monitor(self.env, 'test_monit', video_callable=lambda x: True, force=True)
        tf.set_random_seed(FLAGS.tfseed)
        np.random.seed(FLAGS.npseed)
        self.env.monitor._start(os.path.join(FLAGS.outdir, 'monitor'), force=FLAGS.force)
        #self.menv.monitor.start(os.path.join(FLAGS.outdir, 'monitor'), force=FLAGS.force)
        self.env.seed(FLAGS.gymseed)
        gym.logger.setLevel(gym.logger.WARN)

        dimO = self.env.observation_space.shape
        dimA = self.env.action_space.shape
        pprint.pprint(self.env.spec.__dict__)

        self.agent = Agent(dimO, dimA=dimA)
        test_log = open(os.path.join(FLAGS.outdir, 'test.log'), 'w')
        train_log = open(os.path.join(FLAGS.outdir, 'train.log'), 'w')

        while self.train_timestep < FLAGS.total:
            # test
            reward_list = []
            for _ in range(FLAGS.test):
                reward, timestep = self.run_episode(
                    test=True, monitor=np.random.rand() < FLAGS.monitor)
                reward_list.append(reward)
                self.test_timestep += timestep
            avg_reward = np.mean(reward_list)
            print('Average test return {} after {} timestep of training.'.format(
                avg_reward, self.train_timestep))
            test_log.write("{}\t{}\n".format(self.train_timestep, avg_reward))
            test_log.flush()

            # train
            reward_list = []
            last_checkpoint = np.floor(self.train_timestep / FLAGS.train)
            while np.floor(self.train_timestep / FLAGS.train) == last_checkpoint:
                print('=== Running episode')
                reward, timestep = self.run_episode(test=False, monitor=False)
                reward_list.append(reward)
                self.train_timestep += timestep
                train_log.write("{}\t{}\n".format(self.train_timestep, reward))
                train_log.flush()
            avg_reward = np.mean(reward_list)
            print('Average train return {} after {} timestep of training.'.format(
                avg_reward, self.train_timestep))

            os.system('{} {}'.format(plotScr, FLAGS.outdir))

        self.env.monitor.close()
        os.makedirs(os.path.join(FLAGS.outdir, "tf"))
        ckpt = os.path.join(FLAGS.outdir, "tf/model.ckpt")
        self.agent.saver.save(self.agent.sess, ckpt)

    def run_episode(self, test=True, monitor=False):
        #self.env.monitor.configure(lambda _: monitor)
        observation = self.env.reset()
        self.agent.reset(observation)
        sum_reward = 0
        timestep = 0
        term = False
        times = {'act': [], 'envStep': [], 'obs': []}
        while not term:
            start = time.clock()
            action = self.agent.act(test=test)
            times['act'].append(time.clock()-start)

            start = time.clock()
            observation, reward, term, info = self.env.step(action)
            times['envStep'].append(time.clock()-start)
            term = (not test and timestep + 1 >= FLAGS.tmax) or term

            filtered_reward = self.env.filter_reward(reward)

            start = time.clock()
            self.agent.observe(filtered_reward, term, observation, test=test)
            times['obs'].append(time.clock()-start)

            sum_reward += reward
            timestep += 1

        print('=== Episode stats:')
        for k,v in sorted(times.items()):
            print('  + Total {} time: {:.4f} seconds'.format(k, np.mean(v)))

        print('  + Reward: {}'.format(sum_reward))
        logger.log_stat(prefix + "return_mean_agent{}".format(_i), np.mean(rew[-arglist.save_rate:]), train_step)
        return sum_reward, timestep


###########################################

from sacred import Experiment
import numpy as np
import os
import collections
from os.path import dirname, abspath
import pymongo
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
from utils.logging import get_logger, Logger
from utils.dict2namedtuple import convert
import yaml

SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")

mongo_client = None



# Function to connect to a mongodb and add a Sacred MongoObserver
def setup_mongodb(db_url, db_name):
    client = None
    mongodb_fail = True

    # Try 5 times to connect to the mongodb
    for tries in range(5):
        # First try to connect to the central server. If that doesn't work then just save locally
        maxSevSelDelay = 10000  # Assume 10s maximum server selection delay
        try:
            # Check whether server is accessible
            logger.info("Trying to connect to mongoDB '{}'".format(db_url))
            client = pymongo.MongoClient(db_url, ssl=True, serverSelectionTimeoutMS=maxSevSelDelay)
            client.server_info()
            # If this hasn't raised an exception, we can add the observer
            ex.observers.append(MongoObserver.create(url=db_url, db_name=db_name, ssl=True)) # db_name=db_name,
            logger.info("Added MongoDB observer on {}.".format(db_url))
            mongodb_fail = False
            break
        except pymongo.errors.ServerSelectionTimeoutError:
            logger.warning("Couldn't connect to MongoDB on try {}".format(tries + 1))

    if mongodb_fail:
        logger.error("Couldn't connect to MongoDB after 5 tries!")
        # TODO: Maybe we want to end the script here sometimes?

    return client

FLAGS= None
@ex.main
def my_main(_run, _config, _log):
    global mongo_client


    import datetime
    unique_token = "{}__{}".format(_config["name"], datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # run the framework
    # run(_run, _config, _log, mongo_client, unique_token)
    # arglist = parse_args()

    logger = Logger(_log)
    # configure tensorboard logger
    unique_token = "{}__{}".format(FLAGS.exp_name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    use_tensorboard = False
    if use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)
    logger.setup_sacred(_run)

    def mainer():
        Experiment().run(logger)

    if __name__ == '__main__':
        runtime_env.run(mainer, FLAGS.outdir)

    os._exit(0)

def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict

def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

if __name__ == '__main__':
    import os

    #arglist = parse_args()

    from copy import deepcopy
    # params = deepcopy(sys.argv)

    # scenario_name = None
    # for _i, _v in enumerate(params):
    #     if _v.split("=")[0] == "--scenario":
    #         #scenario_name = _v.split("=")[1]
    #         scenario_name = params[_i + 1]
    #         del params[_i:_i+2]
    #         break
    #
    # name = None
    # for _i, _v in enumerate(params):
    #     if _v.split("=")[0] == "--name":
    #         #scenario_name = _v.split("=")[1]
    #         name = params[_i + 1]
    #         del params[_i:_i+2]
    #         break

    # now add all the config to sacred
    # ex.add_config({"scenario":scenario_name,
    #                "name":name})
    srcDir = os.path.dirname(os.path.realpath(__file__))
    rlDir = os.path.join(srcDir, '..')
    plotScr = os.path.join(rlDir, 'plot-single.py')

    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('exp_name', '', 'experiment name')
    flags.DEFINE_string('env', '', 'gym environment')
    flags.DEFINE_string('outdir', 'output', 'output directory')
    flags.DEFINE_boolean('force', False, 'overwrite existing results')
    flags.DEFINE_integer('train', 1000, 'training timesteps between testing episodes')
    flags.DEFINE_integer('test', 1, 'testing episodes between training timesteps')
    flags.DEFINE_integer('tmax', 1000, 'maxium timesteps each episode')
    flags.DEFINE_integer('total', 100000, 'total training timesteps')
    flags.DEFINE_float('monitor', 0.01, 'probability of monitoring a test episode')
    flags.DEFINE_string('model', 'ICNN', 'reinforcement learning model[DDPG, NAF, ICNN]')
    flags.DEFINE_integer('tfseed', 0, 'random seed for tensorflow')
    flags.DEFINE_integer('gymseed', 0, 'random seed for openai gym')
    flags.DEFINE_integer('npseed', 0, 'random seed for numpy')
    flags.DEFINE_float('ymin', 0, 'random seed for numpy')
    flags.DEFINE_float('ymax', 1000, 'random seed for numpy')

    setproctitle.setproctitle('ICNN.RL.{}.{}.{}'.format(
        FLAGS.env, FLAGS.model, FLAGS.tfseed))

    os.makedirs(FLAGS.outdir, exist_ok=True)
    with open(os.path.join(FLAGS.outdir, 'flags.json'), 'w') as f:
        # json.dump(FLAGS.__flags, f, indent=2, sort_keys=True)
        # dill.dump(FLAGS.__flags, f)
        f.write(FLAGS.flags_into_string())

    if FLAGS.model == 'DDPG':
        import ddpg
        Agent = ddpg.Agent
    elif FLAGS.model == 'NAF':
        import naf
        Agent = naf.Agent
    elif FLAGS.model == 'ICNN':
        import icnn
        Agent = icnn.Agent

    ex.add_config({"name":FLAGS.exp_name})

    # Check if we don't want to save to sacred mongodb
    no_mongodb = False

    # for _i, _v in enumerate(params):
    #     if "no-mongo" in _v:
    #     # if "--no-mongo" == _v:
    #         del params[_i]
    #         no_mongodb = True
    #         break

    config_dict={}
    config_dict["db_url"] = "mongodb://pymarlOwner:EMC7Jp98c8rE7FxxN7g82DT5spGsVr9A@gandalf.cs.ox.ac.uk:27017/pymarl"
    config_dict["db_name"] = "pymarl"

    # If there is no url set for the mongodb, we cannot use it
    if not no_mongodb and "db_url" not in config_dict:
        no_mongodb = True
        logger.error("No 'db_url' to use for Sacred MongoDB")

    if not no_mongodb:
        db_url = config_dict["db_url"]
        db_name = config_dict["db_name"]
        mongo_client = setup_mongodb(db_url, db_name)

    # Save to disk by default for sacred, even if we are using the mongodb
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, "sacred")
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline("")