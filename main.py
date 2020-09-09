# fedavg from https://github.com/WHDY/FedAvg/

import os
import sys
import argparse
#from tqdm import tqdm
import numpy as np
import random
import time
from datetime import datetime
import copy
import torch
import torch.nn.functional as F
from torch import optim
from Models import Mnist_2NN, Mnist_CNN
from Device import Device, DevicesInNetwork
from Block import Block
from Blockchain import Blockchain

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Block_FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-nd', '--num_devices', type=int, default=100, help='numer of the devices in the simulation network')
parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')
parser.add_argument('-mn', '--model_name', type=str, default='mnist_2nn', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                    use value from origin paper as default")
#parser.add_argument('-vf', "--val_freq", type=int, default=5, help="model validation frequency(of communications)")
#parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
parser.add_argument('-max_ncomm', '--max_num_comm', type=int, default=1000, help='maximum number of communication rounds, may terminate early if converges')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')
parser.add_argument('-ns', '--network_stability', type=float, default=0.8, help='the odds a device is online')
parser.add_argument('-gr', '--general_rewards', type=int, default=1, help='rewards for verification of one transaction, mining and so forth')
parser.add_argument('-v', '--verbose', type=int, default=0, help='print verbose debug log')
parser.add_argument('-aio', '--all_in_one', type=int, default=0, help='let all nodes be aware of each other in the network while registering')
parser.add_argument('-ko', '--kick_out_rounds', type=int, default=5, help='a device is kicked out of the network if its accuracy shows decreasing for the number of rounds recorded by a winning validator')
parser.add_argument('-ha', '--hard_assign', type=str, default='*,*,*', help='hard assign number of roles in the network, order by worker, miner and validator')
# parser.add_argument('-la', '--least_assign', type=str, default='*,*,*', help='the assigned number of roles are at least guaranteed in the network')

# def flattern_2d_to_1d(arr):
#     final_set = set()
#     for sub_arr in arr:
#         for ele in sub_arr:
#             final_set.add(sub_arr)
#     return final_set

# def find_sub_nets():
#     # sort of DFS
#     sub_nets = []
#     for device_seq, device in devices_in_network.devices_set.items():
#         sub_net = set()
#         checked_device = flattern_2d_to_1d(sub_nets)
#         while device not in checked_device and not in sub_net:
#             sub_net.add(device)
#             for peer in device.return_peers():
#                 device = peer
#         sub_nets.append(sub_net)


# TODO write logic here as the control should not be in device class, must be outside
def smart_contract_worker_upload_accuracy_to_validator(worker, validator):
    validator.accept_accuracy(worker, rewards)

# TODO should be flexible depending on loose/hard assign
# TODO since we now allow devices to go back online, may discard this function
def check_network_eligibility(check_online=False):
    num_online_workers = 0
    num_online_miners = 0
    num_online_validators = 0
    for worker in workers_this_round:
        if worker.is_online():
            num_online_workers += 1
    for miner in miners_this_round:
        if miner.is_online():
            num_online_miners += 1
    for validator in validators_this_round:
        if validator.is_online():
            num_online_validators += 1
    ineligible = False
    if num_online_workers == 0:
        print('There is no workers online in this round, ', end='')
        ineligible = True
    elif num_online_miners == 0:
        print('There is no miners online in this round, ', end='')
        ineligible = True
    elif num_online_validators == 0:
        print('There is no validators online in this round, ', end='')
        ineligible = True
    if ineligible:
        print('which is ineligible for the network to continue operating.')
        return False
    return True

def register_in_the_network(registrant, check_online=False):
    potential_registrars = set(devices_in_network.devices_set.values())
    # it cannot register with itself
    potential_registrars.discard(registrant)        
    # pick a registrar
    registrar = random.sample(potential_registrars, 1)[0]
    if check_online:
        if not registrar.is_online():
            online_registrars = set()
            for registrar in potential_registrars:
                if registrar.is_online():
                    online_registrars.add(registrar)
            if not online_registrars:
                return False
            registrar = random.sample(online_registrars, 1)[0]
    # registrant add registrar to its peer list
    registrant.add_peers(registrar)
    # this device sucks in registrar's peer list
    registrant.add_peers(registrar.return_peers())
    # registrar adds registrant(must in this order, or registrant will add itself from registrar's peer list)
    registrar.add_peers(registrant)
    return True

def register_by_aio(device):
    device.add_peers(set(devices_in_network.devices_set.values()))

if __name__=="__main__":

    # program running time for logging purpose
    date_time = datetime.now().strftime("%d%m%Y_%H%M%S")

    args = parser.parse_args()
    args = args.__dict__

    # for demonstration purposes, this reward is for everything
    rewards = args["general_rewards"]

    # get number of roles needed in the network
    roles_requirement = args['hard_assign'].split(',')
    
    try:
        workers_needed = int(roles_requirement[0])
    except:
        workers_needed = 0
    
    try:
        miners_needed = int(roles_requirement[1])
    except:
        miners_needed = 0
    
    try:
        validators_needed = int(roles_requirement[2])
    except:
        validators_needed = 0

    if args['num_devices'] < workers_needed + miners_needed + validators_needed:
        sys.exit("ERROR: Roles assigned to the devices exceed the maximum number of allowed devices in the network.")

    # check eligibility
    if args['num_devices'] < 3:
        sys.exit("ERROR: There are not enough devices in the network.\n The system needs at least one miner, one worker and one validator to start the operation.\nSystem aborted.")

    # make chechpoint save path
    if not os.path.isdir(args['save_path']):
        os.mkdir(args['save_path'])

    # create neural net based on the input model name
    net = None
    if args['model_name'] == 'mnist_2nn':
        net = Mnist_2NN()
    elif args['model_name'] == 'mnist_cnn':
        net = Mnist_CNN()

    # assign GPUs if available and prepare the net
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    print(f"{torch.cuda.device_count()} GPUs are available to use!")
    net = net.to(dev)

    # set loss_function and optimizer
    loss_func = F.cross_entropy
    opti = optim.SGD(net.parameters(), lr=args['learning_rate'])

    # TODO - # of malicious nodes, non-even dataset distribution
    # create devices in the network
    devices_in_network = DevicesInNetwork(data_set_name='mnist', is_iid=args['IID'], batch_size = args['batchsize'], loss_func = loss_func, opti = opti, num_devices=args['num_devices'], network_stability=args['network_stability'], net=net, dev=dev, kick_out_rounds=args['kick_out_rounds'])
    # test_data_loader = devices_in_network.test_data_loader

    devices_list = list(devices_in_network.devices_set.values())

    for device in devices_list:
        # set initial global weights
        device.init_global_parameters()
        # simulate peer registration, with respect to device idx order
        if not args["all_in_one"]:
            register_in_the_network(device)
        else:
            register_by_aio(device)

    # remove its own from peer list if there is
    for device in devices_list:
        device.remove_peers(device)

    # build a dict to record worker accuracies for different rounds
    workers_accuracies_records = {}
    for device_seq, device in devices_in_network.devices_set.items():
        workers_accuracies_records[device_seq] = {}

    # FL starts here
    for comm_round in range(1, args['max_num_comm']+1):
        print(f"\nCommunication round {comm_round}")
        workers_this_round = []
        miners_this_round = []
        validators_this_round = []
        # assign role first, and then simulate if online or offline
        workers_to_assign = workers_needed
        miners_to_assign = miners_needed
        validators_to_assign = validators_needed
        random.shuffle(devices_list)
        for device in devices_list:
            if workers_to_assign:
                device.assign_worker_role()
                workers_to_assign -= 1
            elif miners_to_assign:
                device.assign_miner_role()
                miners_to_assign -= 1
            elif validators_to_assign:
                device.assign_validator_role()
                validators_to_assign -= 1
            else:
                device.assign_role()
            if device.return_role() == 'worker':
                workers_this_round.append(device)
            elif device.return_role() == 'miner':
                miners_this_round.append(device)
            else:
                validators_this_round.append(device)
            device.online_switcher()
            #     # though back_online, resync chain when they are performing tasks
            #     if args['verbose']:
            #         print(f'{device.return_idx()} {device.return_role()} online - ', end='')
            # else:
            #     if args['verbose']:
            #         print(f'{device.return_idx()} {device.return_role()} offline - ', end='')
            # # debug chain length
            # if args['verbose']:
            #     print(f"chain length {device.return_blockchain_object().return_chain_length()}")
            # debug
        
        # if not check_network_eligibility():
            # print("Go to the next round.\n")
            # continue
        
        # shuffle the list(for worker, this will affect the order of dataset portions to be trained)
        random.shuffle(workers_this_round)
        random.shuffle(miners_this_round)
        random.shuffle(validators_this_round)

        if args['verbose']:
            print("\nworkers this round are")
            for worker in workers_this_round:
                print(f"d_{worker.return_idx().split('_')[-1]} online - {worker.is_online()} with chain len {worker.return_blockchain_object().return_chain_length()}")
            print("\nminers this round are")
            for miner in miners_this_round:
                print(f"d_{miner.return_idx().split('_')[-1]} online - {miner.is_online()} with chain len {miner.return_blockchain_object().return_chain_length()}")
            if validators_this_round:
                print("\nvalidators this round are")
                for validator in validators_this_round:
                    print(f"d_{validator.return_idx().split('_')[-1]} online - {validator.is_online()} with chain len {validator.return_blockchain_object().return_chain_length()}")
            else:
                print("\nThere are no validators this round.")
        
        if args['verbose']:
            print(f"\nThere are {len(workers_this_round)} workers, {len(miners_this_round)} miners and {len(validators_this_round)} validators in this round.")
            print()

        # debug peers
        if args['verbose']:
            print(f"+++++++++ Round {comm_round} Beginning Peer Lists +++++++++")
            for device_seq, device in devices_in_network.devices_set.items():
                peers = device.return_peers()
                print(f"d_{device_seq.split('_')[-1]} - {device.return_role()[0]} has peer list ", end='')
                for peer in peers:
                    print(f"d_{peer.return_idx().split('_')[-1]} - {peer.return_role()[0]}", end=', ')
                print()
            print(f"+++++++++ Round {comm_round} Beginning Peer Lists +++++++++")

        # re-init round vars
        for miner in miners_this_round:
            if miner.is_online():
                miner.miner_reset_vars_for_new_round()
        for worker in workers_this_round:
            if worker.is_online():
                worker.worker_reset_vars_for_new_round()
        for validator in validators_this_round:
            if validator.is_online():
                validator.validator_reset_vars_for_new_round()
        
        # workers, miners and validators take turns to perform jobs
        # workers
        for worker_iter in range(len(workers_this_round)):
            worker = workers_this_round[worker_iter]
            if worker.is_online():
                # update peer list
                if not worker.update_peer_list(args['verbose']):
                    # peer_list_empty, randomly register with an online node
                    if not register_in_the_network(worker, check_online=True):
                        print("No devices found in the network online in this communication round.")
                        break
                # PoW resync chain
                if worker.pow_resync_chain(args['verbose']):
                    worker.update_model_after_chain_resync()
                # worker perform local update
                print(f"{worker.return_idx()} - worker {worker_iter+1}/{len(workers_this_round)} performing local updates...")
                worker.worker_local_update(rewards)
                # worker associates with a miner
                associated_miner = worker.associate_with_miner()
                if not associated_miner:
                    print(f"Cannot find a miner in {worker.return_idx()} peer list.")
                    continue
                finally_no_associated_miner = False
                # check if the associated miner is online
                while not associated_miner.is_online():
                    worker.remove_peers(associated_miner)
                    associated_miner = worker.associate_with_miner()
                    if not associated_miner:
                        finally_no_associated_miner = True
                        break
                if finally_no_associated_miner:
                    print(f"Cannot find a online miner in {worker.return_idx()} peer list.")
                    continue
                else:
                    print(f"Worker {worker.return_idx()} associated with miner {associated_miner.return_idx()}")
                associated_miner.add_worker_to_association(worker)
                # simulate the situation that worker may go offline during model updates transmission
                worker.online_switcher()
            else:
                print(f"{worker.return_idx()} - worker {worker_iter+1}/{len(workers_this_round)} is offline")
        
        # if not check_network_eligibility():
            # print("Go to the next round.\n")
            # continue
        
        # miners accept local updates and broadcast to other miners
        print()
        for miner_iter in range(len(miners_this_round)):
            miner = miners_this_round[miner_iter]
            if miner.is_online():
                # update peer list
                if not miner.update_peer_list(args['verbose']):
                    # peer_list_empty, randomly register with a online node
                    if not register_in_the_network(worker, check_online=True):
                        print("No devices found in the network online in this communication round.")
                        break
                # PoW resync chain
                if miner.pow_resync_chain(args['verbose']):
                    miner.update_model_after_chain_resync()
                # miner accepts local updates from its workers association
                print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} accepting workers' updates...")
                potential_offline_workers = set()
                associated_workers = miner.return_associated_workers()
                if not associated_workers:
                    print(f"No workers are associated with miner {miner.return_idx()} for this communication round.")
                    continue
                for worker in associated_workers:
                    if worker.is_online():
                        miner.add_unconfirmmed_transaction(worker.return_local_updates_and_signature(comm_round), worker.return_idx())
                    else:
                        potential_offline_workers.add(worker)
                        if args["verbose"]:
                            print(f"worker {worker.return_idx()} is offline when accepting transaction. Removed from peer list.")
                miner.remove_peers(potential_offline_workers)
                if not miner.return_unconfirmmed_transactions():
                    print("Workers offline or disconnected while transmitting updates.")
                    continue
                # broadcast to other miners
                # may go offline at any point
                if miner.online_switcher() and miner.return_unconfirmmed_transactions():
                    miner.broadcast_transactions()
                if miner.is_online():
                    miner.online_switcher()
            else:
                print(f"{miner.return_idx()} - miner {worker_iter+1}/{len(workers_this_round)} is offline")

        # if not check_network_eligibility():
            # print("Go to the next round.\n")
            # continue

        # miners do self and cross-validation(only validating signature at this moment)
        # time spent included in the block_generation_time
        print()
        block_generation_time_spent = {}
        for miner_iter in range(len(miners_this_round)):
            miner = miners_this_round[miner_iter]
            if not miner.is_online():
                # give a chance for miner to go back online and run its errands
                miner.online_switcher()
                if miner.is_back_online():
                    if miner.pow_resync_chain(args['verbose']):
                        miner.update_model_after_chain_resync()
            if miner.is_online():
                start_time = time.time()
                # block index starts from 1
                candidate_block = Block(idx=miner.return_blockchain_object().return_chain_length()+1, miner_pub_key=miner.return_rsa_pub_key())
                # self verification
                unconfirmmed_transactions = miner.return_unconfirmmed_transactions()
                if unconfirmmed_transactions:
                    print(f"\n{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} doing self verification...")
                else:
                    print(f"\nNo recorded transactions by {miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} will not do self verification.")
                for unconfirmmed_transaction in unconfirmmed_transactions:
                    if miner.verify_transaction_by_signature(unconfirmmed_transaction):
                        unconfirmmed_transaction['tx_verified_by'] = miner.return_idx()
                        # TODO any idea?
                        unconfirmmed_transaction['mining_rewards'] = rewards
                        candidate_block.add_verified_transaction(unconfirmmed_transaction)
                        miner.receive_rewards(rewards)
                # cross verification
                accepted_broadcasted_transactions = miner.return_accepted_broadcasted_transactions()
                if accepted_broadcasted_transactions:
                    print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} doing cross verification...")
                else:
                    print(f"No broadcasted transactions have been recorded by {miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} will not do cross verification.")
                for unconfirmmed_transactions in accepted_broadcasted_transactions:
                    for unconfirmmed_transaction in unconfirmmed_transactions:
                        if miner.verify_transaction_by_signature(unconfirmmed_transaction):
                            unconfirmmed_transaction['tx_verified_by'] = miner.return_idx()
                            # TODO any idea?
                            unconfirmmed_transaction['mining_rewards'] = rewards
                            candidate_block.add_verified_transaction(unconfirmmed_transaction)
                            miner.receive_rewards(rewards) 
                # mine the block
                if candidate_block.return_transactions():
                    print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} mining the worker block...")
                    # return the last block and add previous hash
                    last_block = miner.return_blockchain_object().return_last_block()
                    if last_block is None:
                        # mine the genesis block
                        candidate_block.set_previous_block_hash(None)
                    else:
                        candidate_block.set_previous_block_hash(last_block.compute_hash(hash_entire_block=True))
                    # mine the candidate block by PoW, inside which the block_hash is also set
                    mined_block = miner.proof_of_work(candidate_block)
                else:
                    print("No transaction to mine for this block.")
                    continue
                # unfortunately may go offline
                if miner.online_switcher():
                    # record mining time
                    try:
                        block_generation_time_spent[miner] = (time.time() - start_time)/(miner.return_computation_power())
                        print(f"{miner.return_idx()} - miner mines a worker block in {block_generation_time_spent[miner]} seconds.")
                    except:
                        block_generation_time_spent[miner] = float('inf')
                        print(f"{miner.return_idx()} - miner mines a worker block in INFINITE time...")
                    mined_block.set_mining_rewards(rewards)
                    # sign the block
                    miner.sign_block(mined_block)
                    miner.set_mined_block(mined_block)
            else:
                print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} is offline.")

        # if not check_network_eligibility():
            # print("Go to the next round.\n")
            # continue
        # select the winning miner and broadcast its mined block
        try:
            winning_miner = min(block_generation_time_spent.keys(), key=(lambda miner: block_generation_time_spent[miner]))
        except:
            print("No worker block is generated in this round. Skip to the next round.")
            continue
        
        print(f"\n{winning_miner.return_idx()} is the winning miner for the worker block this round.")
        block_to_propagate = winning_miner.return_mined_block()
        # winning miner receives mining rewards
        winning_miner.receive_rewards(block_to_propagate.return_mining_rewards())
        # IGNORE SUBNETS, where propagated block will be tossed
        # Subnets should be found by finding connected nodes in a graph
        # IN REALITY, FORK MAY HAPPEN AT THIS MOMENT
        # actually, in this system fork can still happen - two nodes have the same length of different chain for their peers in different network group to sync. But they should eventually catch up
        # winning miner adds this block to its own chain
        winning_miner.add_block(block_to_propagate)
        print(f"Winning miner {winning_miner.return_idx()} will propagate its worker block.")

        # miner propogate the winning block (just let other miners in its peer list receive it, verify it and add to the blockchain)
        # update peer list
        if not winning_miner.update_peer_list(args['verbose']):
            # peer_list_empty, randomly register with an online node
            if not register_in_the_network(winning_miner, check_online=True):
                print("No devices found in the network online in the peer list of winning miner. Propogated block ")
                continue
        # miners_this_round will be updated to the miners in the peer list of the winnning miner and the winning miner itself
        miners_in_winning_miner_subnet = winning_miner.return_miners_eligible_to_continue()

        print()
        if miners_in_winning_miner_subnet:
            if args["verbose"]:
                print("Miners in the winning miners subnet are")
                for miner in miners_in_winning_miner_subnet:
                    print(f"d_{miner.return_idx().split('_')[-1]}", end=', ')
                miners_in_other_nets = set(miners_this_round).difference(miners_in_winning_miner_subnet)
                if miners_in_other_nets:
                    print("These miners in other subnets will not get this propagated block.")
                    for miner in miners_in_other_nets:
                        print(f"d_{miner.return_idx().split('_')[-1]}", end=', ')
        else:
            if args["verbose"]:
                print("THIS SHOULD NOT GET CALLED AS THERE IS AT LEAST THE WINNING MINER ITSELF IN THE LIST.")

        # debug_propagated_block_list = []
        # last_block_hash = {}
        print()
        miners_in_winning_miner_subnet = list(miners_in_winning_miner_subnet)
        for miner_iter in range(len(miners_in_winning_miner_subnet)):
            miner = miners_in_winning_miner_subnet[miner_iter]
            if miner == winning_miner:
                continue
            if not miner.is_online():
                miner.online_switcher()
                if miner.is_back_online():
                    if miner.pow_resync_chain(args['verbose']):
                        miner.update_model_after_chain_resync()
            if miner.is_online():
                # miner.set_block_to_add(block_to_propagate)
                print(f"\n{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} is accepting propagated worker block.")
                miner.receive_propagated_block(block_to_propagate)
                if miner.return_propagated_block():
                    verified_block = miner.verify_block(miner.return_propagated_block(), winning_miner.return_idx())
                    #last_block_hash[miner.return_idx()] = {}
                    if verified_block:
                        # if verified_block.return_block_idx() != 1:
                        #     last_block_hash[miner.return_idx()]['block_idx'] = miner.return_blockchain_object().return_last_block().return_block_idx()
                        #     last_block_hash[miner.return_idx()]['block_hash'] = miner.return_blockchain_object().return_last_block_hash()
                        #     last_block_hash[miner.return_idx()]['block_str'] = str(sorted(miner.return_blockchain_object().return_last_block().__dict__.items())).encode('utf-8')
                        miner.add_block(verified_block)
                        # debug_propagated_block_list.append(True)
                        pass
                    else:
                        # if block_to_propagate.return_block_idx() != 1:
                        #     last_block_hash[miner.return_idx()]['block_idx'] = miner.return_blockchain_object().return_last_block().return_block_idx()
                        #     last_block_hash[miner.return_idx()]['block_hash'] = miner.return_blockchain_object().return_last_block_hash()
                        #     last_block_hash[miner.return_idx()]['block_str'] = str(sorted(miner.return_blockchain_object().return_last_block().__dict__.items())).encode('utf-8')
                        #     debug_propagated_block_list.append(False)
                        #     miner.verify_block(miner.return_propagated_block())
                        miner.toss_propagated_block()
                        print("Received propagated worker block is either invalid or does not fit this chain. In real implementation, the miners may continue to mine the block. In here, we just simply pass to the next miner. We can assume at least one miner will receive a valid block in this analysis model.")
                # may go offline
                miner.online_switcher()
            else:
                print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} is offline.")
        # print(debug_propagated_block_list)
        # print()
        
        # worker_last_block_hash = {}
        # miner requests worker to download block
        for miner_iter in range(len(miners_in_winning_miner_subnet)):
            miner = miners_in_winning_miner_subnet[miner_iter]
            if not miner.is_online():
                miner.online_switcher()
                if miner.is_back_online():
                    if miner.pow_resync_chain(args['verbose']):
                        miner.update_model_after_chain_resync()
            if miner.is_online():
                if miner.return_has_added_block(): # TODO
                    print(f"\n{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} is requesting its workers to download a new worker block.")
                    block_to_send = miner.return_blockchain_object().return_last_block()
                    associated_workers = miner.return_associated_workers()
                    if not associated_workers:
                        print(f"No workers are associated with miner {miner.return_idx()} to accept the worker block.")
                        continue
                    for worker in associated_workers:
                        print(f"\n{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} is requesting worker {worker.return_idx()} to download...")
                        if not worker.is_online():
                            worker.online_switcher()
                            if worker.is_back_online():
                                if worker.pow_resync_chain(args['verbose']):
                                    worker.update_model_after_chain_resync()
                        if worker.is_online():
                            # worker_last_block_hash[worker.return_idx()] = {}
                            worker.receive_block_from_miner(block_to_send, miner.return_idx())
                            verified_block = worker.verify_block(worker.return_received_block_from_miner(), miner.return_idx())
                            if verified_block:
                                # if verified_block.return_block_idx() != 1:
                                #     worker_last_block_hash[worker.return_idx()]['block_idx'] = worker.return_blockchain_object().return_last_block().return_block_idx()
                                #     worker_last_block_hash[worker.return_idx()]['block_hash'] = worker.return_blockchain_object().return_last_block_hash()
                                #     worker_last_block_hash[worker.return_idx()]['block_str'] = str(sorted(worker.return_blockchain_object().return_last_block().__dict__.items())).encode('utf-8')
                                worker.add_block(verified_block)
                                pass
                            else:
                                # if block_to_send.return_block_idx() != 1:
                                #     worker.verify_block(worker.return_received_block_from_miner())
                                #     worker_last_block_hash[worker.return_idx()]['block_idx'] = worker.return_blockchain_object().return_last_block().return_block_idx()
                                #     worker_last_block_hash[worker.return_idx()]['block_hash'] = worker.return_blockchain_object().return_last_block_hash()
                                #     worker_last_block_hash[worker.return_idx()]['block_str'] = str(sorted(worker.return_blockchain_object().return_last_block().__dict__.items())).encode('utf-8')
                                worker.toss_received_block()
                                print("Received block from the associated miner is not valid or does not fit its chain. Pass to the next worker.")
                            worker.online_switcher()
            miner.online_switcher()
                            
        # if not check_network_eligibility():
            # print("Go to the next round.\n")
            # continue
        
        print()
        # workers do global updates
        for worker in workers_this_round:
            if not worker.is_online():
                worker.online_switcher()
                if worker.is_back_online():
                    if worker.pow_resync_chain(args['verbose']):
                        worker.update_model_after_chain_resync()
            if worker.is_online():
                print(f'Worker {worker.return_idx()} is doing global update...')
                if worker.return_received_block_from_miner():
                    worker.global_update()
                    accuracy = worker.evaluate_updated_weights()
                    accuracy = float(accuracy)
                    worker.set_accuracy_this_round(accuracy)
                    report_msg = f'Worker {worker.return_idx()} at the communication round {comm_round+1} with chain length {worker.return_blockchain_object().return_chain_length()} has accuracy: {accuracy}\n'
                    print(report_msg)
                    workers_accuracies_records[worker.return_idx()][f'round_{comm_round}'] = accuracy
                    worker.online_switcher()
                else:
                    print(f'No block has been sent to worker {worker.return_idx()}. Skipping global update.\n')
        
        # record accuries in log file
        log_file_path = f"logs/accuracy_report_{date_time}.txt"
        open(log_file_path, 'w').close()
        for device_idx, accuracy_records in workers_accuracies_records.items():
            accuracy_list = []
            for accuracy in accuracy_records.values():
                accuracy_list.append(accuracy)
            with open(log_file_path, "a") as file:
                file.write(f"{device_idx} : {accuracy_list}\n")
        
        # if not check_network_eligibility():
            # print("Go to the next round.\n")
            # continue
        
        # TODO validator may also be evil. how to validate validators?
        # or maybe in this specific settings, since devices take turns to become validators and not specifically set to some certain memvbers, we believe most of the members in the system want to benefit the whole community and trust validators by default
        # after all, they are still taking chances to send their validations to the miners
        # workers send their accuracies to validators and validators record the accuracies in a block
        # iterating validator is easier than iterating worker because of the creation of the validator block

        print("Begin validator rounds.")
        # validators request accuracies from the workers in their peer lists
        for validator_iter in range(len(validators_this_round)):
            validator = validators_this_round[validator_iter]
            if validator.is_online():
                # update peer list
                if not validator.update_peer_list(args['verbose']):
                    # peer_list_empty, randomly register with an online node
                    if not register_in_the_network(validator, check_online=True):
                        print("No devices found in the network online in this communication round.")
                        break
                # PoW resync chain
                if validator.pow_resync_chain(args['verbose']):
                    validator.update_model_after_chain_resync()
                last_block_on_validator_chain = validator.return_blockchain_object().return_last_block()
                print(f"{validator.return_idx()} - validator {validator_iter+1}/{len(validators_this_round)} accepting workers' accuracies...")
                # check last block
                if last_block_on_validator_chain == None or last_block_on_validator_chain.is_validator_block():
                    print("last block ineligible to be operated")
                    continue
                online_workers_in_peer_list = validator.return_online_workers()
                if not online_workers_in_peer_list:
                    print(f"Cannot find online workers in {validator.return_idx()} peer list.")
                    continue

                # validator_candidate_block = Block(idx=validator.blockchain.return_chain_length()+1, is_validator_block=True)
                for worker in online_workers_in_peer_list:
                    smart_contract_worker_upload_accuracy_to_validator(worker, validator)
                # validator.record_worker_performance_in_block(validator_candidate_block, comm_round, args["general_rewards"])
                # associate with a miner
                associated_miner = validator.associate_with_miner()
                if not associated_miner:
                    print(f"Cannot find a miner in {validator.return_idx()} peer list.")
                    continue
                finally_no_associated_miner = False
                # check if the associated miner is online
                while not associated_miner.is_online():
                    validator.remove_peers(associated_miner)
                    associated_miner = validator.associate_with_miner()
                    if not associated_miner:
                        finally_no_associated_miner = True
                        break
                if finally_no_associated_miner:
                    print(f"Cannot find a online miner in {validator.return_idx()} peer list.")
                    continue
                else:
                    print(f"Validator {validator.return_idx()} associated with miner {associated_miner.return_idx()}")
                associated_miner.add_validator_to_association(validator)
                # may go offline
                validator.online_switcher()
            else:
                print(f"{validator.return_idx()} - validator {validator_iter+1}/{len(validators_this_round)} is offline") 
                
        # miners accept validators' transactions and broadcast to other miners
        print()
        for miner_iter in range(len(miners_this_round)):
            miner = miners_this_round[miner_iter]
            if not miner.is_online():
                miner.online_switcher()
                if miner.is_back_online():
                    if miner.pow_resync_chain(args['verbose']):
                        miner.update_model_after_chain_resync()
            if miner.is_online():
                # update peer list
                if not miner.update_peer_list(args['verbose']):
                    # peer_list_empty, randomly register with a online node
                    if not register_in_the_network(worker, check_online=True):
                        print("No devices found in the network online in this communication round.")
                        break
                # PoW resync chain
                if miner.pow_resync_chain(args['verbose']):
                    miner.update_model_after_chain_resync()
                # miner accepts validator transactions
                miner.miner_reset_vars_for_new_validation_round()
                print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} accepting validators' transactions...")
                potential_offline_validators = set()
                associated_validators = miner.return_associated_validators()
                if not associated_validators:
                    print(f"No validators are associated with miner {miner.return_idx()} for this communication round.")
                    continue
                for validator in associated_validators:
                    if validator.is_online():
                        miner.add_unconfirmmed_transaction(validator.return_validations_and_signature(comm_round), validator.return_idx())
                    else:
                        potential_offline_validators.add(validator)
                        if args["verbose"]:
                            print(f"validator {validator.return_idx()} is offline when accepting transaction. Removed from peer list.")
                miner.remove_peers(potential_offline_validators)
                if not miner.return_unconfirmmed_transactions():
                    print("Validators offline or disconnected while transmitting validations.")
                    continue
                # broadcast to other miners
                # may go offline at any point
                if miner.online_switcher() and miner.return_unconfirmmed_transactions():
                    miner.broadcast_transactions()
                if miner.is_online():
                    miner.online_switcher()
            else:
                print(f"{miner.return_idx()} - miner {worker_iter+1}/{len(workers_this_round)} is offline")

        # miners do self and cross-validation(only validating signature at this moment)
        print()
        block_generation_time_spent = {}
        for miner_iter in range(len(miners_this_round)):
            miner = miners_this_round[miner_iter]
            if not miner.is_online():
                miner.online_switcher()
                if miner.is_back_online():
                    if miner.pow_resync_chain(args['verbose']):
                        miner.update_model_after_chain_resync()
            if miner.is_online():
                last_block_on_miner_chain = miner.return_blockchain_object().return_last_block()
                # check last block
                # though miner could still mine this block, but if it finds itself cannot add this mined block, it's unwilling to mine
                if last_block_on_miner_chain==None or last_block_on_miner_chain.is_validator_block():
                    print("last block ineligible to be operated")
                    continue
                start_time = time.time()
                # block index starts from 1
                candidate_block = Block(idx=miner.return_blockchain_object().return_chain_length()+1, miner_pub_key=miner.return_rsa_pub_key(), is_validator_block=True)
                # self verification
                unconfirmmed_transactions = miner.return_unconfirmmed_transactions()
                if unconfirmmed_transactions:
                    print(f"\n{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} doing self verification...")
                else:
                    print(f"\nNo recorded transactions by {miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} will not do self verification.")
                for unconfirmmed_transaction in miner.return_unconfirmmed_transactions():
                    if miner.verify_transaction_by_signature(unconfirmmed_transaction):
                        unconfirmmed_transaction['tx_verified_by'] = miner.return_idx()
                        # TODO any idea?
                        unconfirmmed_transaction['mining_rewards'] = rewards
                        candidate_block.add_verified_transaction(unconfirmmed_transaction)
                        miner.receive_rewards(rewards)
                # cross verification
                accepted_broadcasted_transactions = miner.return_accepted_broadcasted_transactions()
                if accepted_broadcasted_transactions:
                    print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} doing cross verification...")
                else:
                    print(f"No broadcasted transactions have been recorded by {miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} will not do cross verification.")
                for unconfirmmed_transactions in miner.return_accepted_broadcasted_transactions():
                    for unconfirmmed_transaction in unconfirmmed_transactions:
                        if miner.verify_transaction_by_signature(unconfirmmed_transaction):
                            unconfirmmed_transaction['tx_verified_by'] = miner.return_idx()
                            # TODO any idea?
                            unconfirmmed_transaction['mining_rewards'] = rewards
                            candidate_block.add_verified_transaction(unconfirmmed_transaction)
                            miner.receive_rewards(rewards) 
                # mine the block
                if candidate_block.return_transactions():
                    print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} mining the validator block...")
                    # add previous hash(last block had been checked above)
                    candidate_block.set_previous_block_hash(miner.return_blockchain_object().return_last_block().compute_hash(hash_entire_block=True))
                    # mine the candidate block by PoW, inside which the block_hash is also set
                    mined_block = miner.proof_of_work(candidate_block)
                else:
                    print("No transaction to mine for this block.")
                    continue
                # unfortunately may go offline
                if miner.online_switcher():
                    # record mining time
                    try:
                        block_generation_time_spent[miner] = (time.time() - start_time)/(miner.return_computation_power())
                        print(f"{miner.return_idx()} - miner mines a validator block in {block_generation_time_spent[miner]} seconds.")
                    except:
                        block_generation_time_spent[miner] = float('inf')
                        print(f"{miner.return_idx()} - miner mines a validator block in INFINITE time...")
                    mined_block.set_mining_rewards(rewards)
                    # sign the block
                    miner.sign_block(mined_block)
                    miner.set_mined_block(mined_block)
            else:
                print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} is offline.")

        # if not check_network_eligibility():
            # print("Go to the next round.\n")
            # continue
        # select the winning miner and broadcast its mined block
        try:
            winning_miner = min(block_generation_time_spent.keys(), key=(lambda miner: block_generation_time_spent[miner]))
        except:
            print("No validator block is generated in this round. Skip to the next round.")
            continue
        
        print(f"\n{winning_miner.return_idx()} is the winning miner for the validator block this round.")
        validator_block_to_propagate = winning_miner.return_mined_block()
        winning_miner.receive_rewards(block_to_propagate.return_mining_rewards())
        winning_miner.add_block(validator_block_to_propagate)
        print(f"Winning miner {winning_miner.return_idx()} will propagate its validator block.")

        if not winning_miner.update_peer_list(args['verbose']):
            # peer_list_empty, randomly register with an online node
            if not register_in_the_network(winning_miner, check_online=True):
                print("No devices found in the network online in the peer list of winning miner. Propogated block ")
                continue

        miners_in_winning_miner_subnet = winning_miner.return_miners_eligible_to_continue()
            
        print()
        if miners_in_winning_miner_subnet:
            if args["verbose"]:
                print("Miners in the winning miners subnet are")
                for miner in miners_in_winning_miner_subnet:
                    print(f"d_{miner.return_idx().split('_')[-1]}", end=', ')
                miners_in_other_nets = set(miners_this_round).difference(miners_in_winning_miner_subnet)
                if miners_in_other_nets:
                    print("These miners in other subnets will not get this propagated block.")
                    for miner in miners_in_other_nets:
                        print(f"d_{miner.return_idx().split('_')[-1]}", end=', ')
        else:
            if args["verbose"]:
                print("THIS SHOULD NOT GET CALLED AS THERE IS AT LEAST THE WINNING MINER ITSELF IN THE LIST.")
        
        print()
        miners_in_winning_miner_subnet = list(miners_in_winning_miner_subnet)
        for miner_iter in range(len(miners_in_winning_miner_subnet)):
            miner = miners_in_winning_miner_subnet[miner_iter]
            if miner == winning_miner:
                continue
            if not miner.is_online():
                miner.online_switcher()
                if miner.is_back_online():
                    if miner.pow_resync_chain(args['verbose']):
                        miner.update_model_after_chain_resync()
            if miner.is_online():
                # miner.set_block_to_add(block_to_propagate)
                print(f"\n{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} is accepting propagated validator block.")
                miner.receive_propagated_validator_block(validator_block_to_propagate)
                if miner.return_propagated_validator_block():
                    verified_block = miner.verify_block(miner.return_propagated_validator_block(), winning_miner.return_idx())
                    #last_block_hash[miner.return_idx()] = {}
                    if verified_block:
                        miner.add_block(verified_block)
                        pass
                    else:
                        miner.toss_ropagated_validator_block()
                        print("Received propagated validator block is either invalid or does not fit this chain. In real implementation, the miners may continue to mine the block. In here, we just simply pass to the next miner. We can assume at least one miner will receive a valid block in this analysis model.")
                # may go offline
                miner.online_switcher()
            else:
                print(f"{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} is offline.")

        # miner requests worker and validator to download the validator block
        # does not matter if the miner did not append block above and send its last block, as it won't be verified
        for miner_iter in range(len(miners_in_winning_miner_subnet)):
            miner = miners_in_winning_miner_subnet[miner_iter]
            if not miner.is_online():
                miner.online_switcher()
                if miner.is_back_online():
                    if miner.pow_resync_chain(args['verbose']):
                        miner.update_model_after_chain_resync()
            if miner.is_online():
                if miner.return_has_added_block(): # TODO
                    print(f"\n{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} is requesting its workers and validators to download a new validator block.")
                    block_to_send = miner.return_blockchain_object().return_last_block()
                    associated_workers = miner.return_associated_workers()
                    associated_validators = miner.return_associated_validators()
                    associated_devices = associated_workers.union(associated_validators)
                    if not associated_devices:
                        print(f"No devices are associated with miner {miner.return_idx()} to accept the validator block.")
                        continue
                    associated_devices = list(associated_devices)
                    for device_iter in range(len(associated_devices)):
                        device = associated_devices[device_iter]
                        print(f"\n{miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} is requesting device {device_iter+1}/{len(associated_devices)} {device.return_idx()} - {device.return_role()} to download...")
                        if not device.is_online():
                            device.online_switcher()
                            if device.is_back_online():
                                if device.pow_resync_chain(args['verbose']):
                                    device.update_model_after_chain_resync()
                        if device.is_online():
                            # worker_last_block_hash[worker.return_idx()] = {}
                            device.reset_received_block_from_miner_vars()
                            device.receive_block_from_miner(block_to_send, miner.return_idx())
                            verified_block = device.verify_block(device.return_received_block_from_miner(), miner.return_idx())
                            if verified_block:
                                device.add_block(verified_block)
                                pass
                            else:
                                device.toss_received_block()
                                print("Received block from the associated miner is not valid or does not fit its chain. Pass to the next worker.")
                            device.online_switcher()
                            if device.return_has_added_block():
                                block_to_propagate = device.return_blockchain_object().return_last_block()
                                if block_to_propagate.is_validator_block():
                                    device.operate_on_validator_block()
                    print(f"\nminer {miner.return_idx()} is processing this validator block.")
                    miner.operate_on_validator_block()

                
