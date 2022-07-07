import ray
import ray.rllib.agents.ppo as ppo
import json, os, shutil, sys
import gym
import pprint
import time
import shelve
import argparse
import csv
import math
from tensorflow import keras
from ray import tune

# Set the GPU config. gpu_opt is a string with values 'gpu0', 'gpu1', 'none' or 'both'
# indicating if you want to see as visible one of the GPUs, none of them or both.
# Returns the number of GPUs that are able to be used.
def gpu_options(gpu_opt):
    if(gpu_opt == 'gpu0'):
        # Set only GPU 0 as visible
        #tf.config.set_visible_devices(physical_devices[0], 'GPU')
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        os.system('export "CUDA_VISIBLE_DEVICES"="0"')
        num_gpus = 1

    elif(gpu_opt == 'gpu1'):
        # Set only GPU 1 as visible
        #tf.config.set_visible_devices(physical_devices[1], 'GPU')
        os.environ["CUDA_VISIBLE_DEVICES"]="1"
        os.system('export "CUDA_VISIBLE_DEVICES"="1"')
        num_gpus=1

    elif(gpu_opt == 'both'):
        os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
        os.system('export "CUDA_VISIBLE_DEVICES"="0,1"')
        num_gpus=2
    elif(gpu_opt == 'none'):
        os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
        os.system('export "CUDA_VISIBLE_DEVICES"="0,1"')
        num_gpus=0

    return num_gpus

def get_config_depth_filters(depth, filters):
    # La última capa está determinado cuál tiene que ser
    filters_per_layer = int((filters-1)/depth)
    filters_remainder = (filters-1) % depth
    config = ppo.DEFAULT_CONFIG.copy()
    config['model']['dim'] = 84
    config['model']['conv_filters'] = []
    for i in range(depth-1):
      config['model']['conv_filters'].append([filters_per_layer, [8, 8], 1])
    config['model']['conv_filters'].append([1, [8, 8], config['model']['dim']])
    for i in range(filters_remainder):
      config['model']['conv_filters'][i][0] += 1
    return config

def full_train(checkpoint_root, agent, n_iter, save_file, n_ini = 0, header = True, restore = False, restore_dir = None):
    s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f} learn_time_ms {:6.2f} total_train_time_s {:6.2f} saved {}"
    if(restore):
        if restore_dir == None:
            print("Error: you must specify a restore path")
            return
        agent.restore(restore_dir)
    #else:
        #shutil.rmtree(checkpoint_root, ignore_errors=True, onerror=None)
    results = []
    episode_data = []
    episode_json = []

    total_learn_time = 0
    for n in range(n_iter):
        # Compute one training iteration and measure time
        t0 = time.time()
        result = agent.train()
        t1 = time.time()-t0
        results.append(result)
        episode = {'n': n_ini + n + 1,
                   'episode_reward_min': result['episode_reward_min'],
                   'episode_reward_mean': result['episode_reward_mean'],
                   'episode_reward_max': result['episode_reward_max'],
                   'episode_len_mean': result['episode_len_mean'],
                   'learn_time_ms': result['timers']['learn_time_ms'],
                   'total_train_time_s': t1
                   }
        episode_data.append(episode)
        episode_json.append(json.dumps(episode))
        # Save checkpoint after training
        file_name = agent.save(checkpoint_root)
        print(s.format(
        n_ini + n + 1,
        result["episode_reward_min"],
        result["episode_reward_mean"],
        result["episode_reward_max"],
        result["episode_len_mean"],
        result["timers"]["learn_time_ms"],
        t1,
        file_name
       ))
        total_learn_time+= result["timers"]["learn_time_ms"]

    print("Total learn time: " + str(total_learn_time))
    print("Average learn time per iteration: " + str(total_learn_time/n_iter))

    # Store results in a json file
    with open(save_file + '.json', mode='a') as outfile:
        json.dump(episode_json, outfile)

    # Store results in a csv file
    with open(save_file + '.csv', mode='a') as csv_file:
        fieldnames = ['n', 'episode_reward_min', 'episode_reward_mean', 'episode_reward_max', 'episode_len_mean', 'learn_time_ms','total_train_time_s']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if header:
            writer.writeheader()
        for row in episode_data:
            writer.writerow(row)

    return results

def main():
    # Argument parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-de', '--depth', required=True, type=int, help='Integer indicating models depth.')
    parser.add_argument(
        '-n', '--filters', required=True, type=int, default = None, help='Total number of filters for the model.')
    parser.add_argument(
        '-g', '--gpu', required=False, type = str, default='none', choices =['none','gpu0','gpu1','both'], help='GPU options (if available)')
    parser.add_argument(
        '-d', '--driver-gpus', required = False, type=float, default = 0.0, help='Bumber of GPUs to asign to the driver. It can be a decimal number')
    parser.add_argument(
        '-w', '--workers', required = True, type=int, default=8, help='Number of rollout workers to create')
    parser.add_argument(
        '-s', '--save-name', required=True, type=str, help='directory to save checkpoints (in ./checkpoints/ppo/...) and timing data (in ./training_results/ppo/...)')
    parser.add_argument(
        '-i', '--iters', required = True, type=int, default=10, help= 'Number of training iters to run')
    parser.add_argument(
        '-c', '--cpus', required = False, type=int, default=None, help='Number of CPUs available to use when calling ray.init()')
    parser.add_argument(
        '-a', '--set-affinity', required = False, type=set, default = {}, help='CPUs to run executions only on them.')
    parser.add_argument(
        '-r', '--restore-dir', required = False, type =str, default=None, help='Checkpoint directory to restore model from it.')
    parser.add_argument(
        '-tp', '--total-params-file', required = False, type =str, default=None, help='File path for annotate total number of params with this configuration.')
    args = parser.parse_args()

    # execute only in a set of CPUs
    if(args.set_affinity != {}):
        os.sched_setaffinity(0,args.set_affinity)

    ray.shutdown()

    # Get GPU options
    num_gpus=gpu_options(args.gpu)

    # Init Ray with or without CPU limitation
    if args.cpus is not None:
        ray.init(num_cpus=args.cpus)
    else:
        ray.init()

    # Set agent config
    config = get_config_depth_filters(args.depth, args.filters)

    config['num_workers'] = args.workers
    config['num_gpus'] = args.driver_gpus
    config['num_gpus_per_worker'] = (num_gpus-config['num_gpus'])/args.workers

    print(config)
    # Create agent and show info
    agent = ppo.PPOTrainer(config, env='Pong-v0')
    policy=agent.get_policy()
    print(policy.model.model_config)
    print(policy.model.base_model.summary())

    print("Configuracion del agente:\n\n" + str(config))
    print("\nConfiguracion del modelo del agente:\n\n" + str(config["model"]))
    # Run training iterations storing checkpoints and results
    checkpoint_root= '../checkpoints/ppo/' + args.save_name
    save_file = '../training_results/ppo/' + args.save_name
    if args.restore_dir is not None:
        # restore dir has format checkpoints/ppo/model.../checkpoint_X/checkpoint-X 
        # and we set as n_ini the value of X
        n_ini=args.restore_dir.split('/')[len(args.restore_dir.split('/'))-1].split('-')[1]
        t0 = time.time()
        full_train(checkpoint_root, agent, args.iters, save_file, n_ini = n_ini, header=False, restore = True, restore_dir=args.restore_dir)
        t1 = time.time()-t0
    else:
        t0 = time.time()
        full_train(checkpoint_root, agent, args.iters, save_file)
        t1 = time.time()-t0
    print("Total time for the " + str(args.iters) + " training iterations: " + str(t1))

    # Ray results are by default saved in ~/ray_results/ dir. So we take
    # the corresponding files (that belongs to the last modified directory in ~/ray_results)
    # and copy them in a new directory inside our working dir.
    actual_dir = os.getcwd()
    os.chdir('/home/jorvil01/ray_results')
    ray_results_list=os.listdir()
    ray_results_list.sort(key=os.path.getctime)
    ray_results_dir=ray_results_list[len(ray_results_list)-1]

    print(args.save_name)

    if(not os.path.exists(actual_dir + '/ray_results/'+args.save_name)):
        os.mkdir(actual_dir + '/ray_results/'+args.save_name)
    os.system("cp -r {}/* {}".format(ray_results_dir, actual_dir + '/ray_results/'+args.save_name))
    os.chdir(actual_dir)
    print(checkpoint_root)
    # Copy params.pkl to checkpoint dir for rollouts
    os.system("cp ray_results/{}/params.pkl {}/".format(args.save_name, checkpoint_root))

if __name__== '__main__':
    main()
