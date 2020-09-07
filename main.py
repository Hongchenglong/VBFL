# fedavg from https://github.com/WHDY/FedAvg/

import os
import sys
import argparse
#from tqdm import tqdm
import numpy as np
import random
import time
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
parser.add_argument('-ko', '--kick_out_rounds', type=int, default=0, help='a device is kicked out of the network if its accuracy shows decreasing for the number of rounds recorded by a winning validator')

# for demonstration purposes, this reward is for everything
rewards = args["general_rewards"]

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
    potential_registrars = set(devices_in_network.devices_set.keys())
    # it cannot register with itself
    potential_registrars.discard(registrant.return_idx())        
    # pick a registrar
    registrar_idx = random.sample(potential_registrars, 1)[0]
    registrar = devices_in_network.devices_set[registrar_idx]
    if check_online:
        if not registrar.is_online():
            online_registrars_idxes = set()
            for registrar_idx in potential_registrars:
                if devices_in_network.devices_set[registrar_idx].is_online():
                    online_registrars_idxes.add(registrar_idx)
            if not online_registrars_idxes:
                return False
            registrar_idx = random.sample(online_registrars_idxes, 1)[0]
            registrar = devices_in_network.devices_set[registrar_idx]
    # registrant add registrar to its peer list
    registrant.add_peers(registrar)
    # this device sucks in registrar's peer list
    registrant.add_peers(registrar.return_peers())
    # registrar adds registrant(must in this order, or registrant will add itself from registrar's peer list)
    registrar.add_peers(registrant)
    return True

if __name__=="__main__":
    args = parser.parse_args()
    args = args.__dict__

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

    for device_seq, device in devices_in_network.devices_set.items():
        # set initial global weights
        device.init_global_parameters()
        # simulate peer registration, with respect to device idx order 
        register_in_the_network(device)

    # remove its own from peer list if there is
    for device_seq, device in devices_in_network.devices_set.items():
        device.remove_peers(device)

    # debug peers
    if args['verbose']:
        for device_seq, device in devices_in_network.devices_set.items():
            peers = device.return_peers()
            print(f'{device_seq} has peer list ', end='')
            for peer in peers:
                print(peer.return_idx(), end=', ')
            print()

    # FL starts here
    for comm_round in range(args['max_num_comm']):
        print(f"Communicate round {comm_round+1}")
        workers_this_round = []
        miners_this_round = []
        validators_this_round = []
        # assign role first, and then simulate if online or offline
        for device_seq, device in devices_in_network.devices_set.items():
            device.assign_role()
            if device.return_role() == 'worker':
                workers_this_round.append(device)
            elif device.return_role() == 'miner':
                miners_this_round.append(device)
            else:
                validators_this_round.append(device)
            if device.online_switcher():
                # though back_online, resync chain when they are performing tasks
                if args['verbose']:
                    print(f'{device.return_idx()} {device.return_role()} is online at the beginning')
            # debug chain length
            if args['verbose']:
                chain = device.return_blockchain_object().return_chain_structure()
                print(f"{device.return_idx()} has chain length {len(chain)}")
            # debug
        
        # if not check_network_eligibility():
            # print("Go to the next round.\n")
            # continue
        
        # shuffle the list(for worker, this will affect the order of dataset portions to be trained)
        random.shuffle(workers_this_round)
        random.shuffle(miners_this_round)
        random.shuffle(validators_this_round)
        
        if args['verbose']:
            print(f"There are {len(workers_this_round)} workers, {len(miners_this_round)} miners and {len(validators_this_round)} validators in this round.")

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
                if not worker.update_peer_list():
                    # peer_list_empty, randomly register with an online node
                    if not register_in_the_network(worker, check_online=True):
                        print("No devices found in the network online in this communication round.")
                        break
                # PoW resync chain
                if worker.pow_resync_chain():
                    worker.update_model_after_chain_resync()
                # worker perform local update
                print(f"This is {worker.return_idx()} - worker {worker_iter+1}/{len(workers_this_round)} performing local updates...")
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
                if finally_no_associated_miner:
                    print(f"Cannot find a online miner in {worker.return_idx()} peer list.")
                    continue
                associated_miner.add_worker_to_association(worker)
                # simulate the situation that worker may go offline during model updates transmission
                worker.online_switcher() 
        
        # if not check_network_eligibility():
            # print("Go to the next round.\n")
            # continue
        
        # miners accept local updates and broadcast to other miners
        for miner_iter in range(len(miners_this_round)):
            miner = miners_this_round[miner_iter]
            if miner.is_online():
                # update peer list
                if not miner.update_peer_list():
                    # peer_list_empty, randomly register with a online node
                    if not register_in_the_network(worker, check_online=True):
                        print("No devices found in the network online in this communication round.")
                        break
                # PoW resync chain
                if miner.pow_resync_chain():
                    miner.update_model_after_chain_resync()
                # miner accepts local updates from its workers association
                print(f"This is {miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} accepting workers' updates...")
                potential_offline_workers = set()
                associated_workers = miner.return_associated_workers()
                if not associated_workers:
                    print(f"No workers are associated with miner {miner.return_idx()} for this communication round.")
                    continue
                for worker in associated_workers:
                    if worker.is_online():
                        miner.add_unconfirmmed_transaction(worker.return_local_updates_and_signature(comm_round))
                    else:
                        potential_offline_workers.add(worker)
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

        # if not check_network_eligibility():
            # print("Go to the next round.\n")
            # continue

        # miners do self and cross-validation(only validating signature at this moment)
        # time spent included in the block_generation_time
        block_generation_time_spent = {}
        for miner in miners_this_round:
            if not miner.is_online():
                # give a chance for miner to go back online and run its errands
                miner.online_switcher()
                if miner.is_back_online():
                    if miner.pow_resync_chain():
                        miner.update_model_after_chain_resync()
            if miner.is_online():
                if miner.return_associated_workers():
                    start_time = time.time()
                    # block index starts from 1
                    candidate_block = Block(idx=miner.return_blockchain_object().return_chain_length()+1, miner_pub_key=miner.return_rsa_pub_key())
                    # self verification
                    for unconfirmmed_transaction in miner.return_unconfirmmed_transactions():
                        if miner.verify_transaction_by_signature(unconfirmmed_transaction):
                            unconfirmmed_transaction['tx_verified_by'] = miner.return_idx()
                            # TODO any idea?
                            unconfirmmed_transaction['mining_rewards'] = rewards
                            candidate_block.add_verified_transaction(unconfirmmed_transaction)
                            miner.receive_rewards(rewards)
                    # cross verification
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
                        # return the last block and add previous hash
                        last_block = miner.get_blockchain_object().return_last_block()
                        if last_block is None:
                            # mine the genesis block
                            candidate_block.set_previous_block_hash(None)
                        else:
                            candidate_block.set_previous_block_hash(last_block.compute_hash(hash_whole_block=True))
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
                        except:
                            block_generation_time_spent[miner] = float('inf')
                        mined_block.set_mining_rewards(rewards)
                        # sign the block
                        miner.sign_block(mined_block)
                        miner.set_mined_block(mined_block)

        # if not check_network_eligibility():
            # print("Go to the next round.\n")
            # continue
        # select the winning miner and broadcast its mined block
        try:
            winning_miner = min(block_generation_time_spent.keys(), key=(lambda miner: block_generation_time_spent[miner]))
        except:
            print("No block is generated in this round. Skip to the next round.")
            continue
        
        block_to_propagate = winning_miner.return_mined_block()
        # winning miner receives mining rewards
        winning_miner.receive_rewards(block_to_propagate.return_mining_rewards())
        # IGNORE SUBNETS, where propagated block will be tossed
        # Subnets should be found by finding connected nodes in a graph
        # IN REALITY, FORK MAY HAPPEN AT THIS MOMENT
        # actually, in this system fork can still happen - two nodes have the same length of different chain for their peers in different network group to sync. But they should eventually catch up
        # winning miner adds this block to its own chain
        winning_miner.add_block(block_to_propagate)
        print(f"Winning miner {winning_miner.return_idx()} will propagate its block.")

        # miner propogate the winning block (just let other miners in its peer list receive it, verify it and add to the blockchain)
        # update peer list
        if not winning_miner.update_peer_list():
            # peer_list_empty, randomly register with an online node
            if not register_in_the_network(winning_miner, check_online=True):
                print("No devices found in the network online in the peer list of winning miner. Propogated block ")
                continue
        # miners_this_round will be updated to the miners in the peer list of the winnning miner and the winning miner itself
        miners_this_round = winning_miner.return_miners_eligible_to_continue()

        # debug_propagated_block_list = []
        # last_block_hash = {}
        for miner in miners_this_round:
            if miner == winning_miner:
                continue
            if not miner.is_online():
                miner.online_switcher()
                if miner.is_back_online():
                    if miner.pow_resync_chain():
                        miner.update_model_after_chain_resync()
            if miner.is_online():
                # miner.set_block_to_add(block_to_propagate)
                miner.receive_propagated_block(block_to_propagate)
                verified_block = miner.verify_block(miner.return_propagated_block())
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
                    print("Received propagated block is either invalid or does not fit this chain. In real implementation, the miners may continue to mine the block. In here, we just simply pass to the next miner. We can assume at least one miner will receive a valid block in this analysis model.")
                # may go offline
                miner.online_switcher()
        # print(debug_propagated_block_list)
        # print()
        
        # worker_last_block_hash = {}
        # miner requests worker to download block
        for miner in miners_this_round:
            if not miner.is_online():
                miner.online_switcher()
                if miner.is_back_online():
                    if miner.pow_resync_chain():
                        miner.update_model_after_chain_resync()
            if miner.is_online():
                if miner.return_has_added_block(): # TODO
                    block_to_send = miner.return_blockchain_object().return_last_block()
                    associated_workers = miner.return_associated_workers()
                    if not associated_workers:
                        print(f"No workers are associated with miner {miner.return_idx()} for this communication round.")
                        continue
                    for worker in associated_workers:
                        if not worker.is_online():
                            worker.online_switcher()
                            if worker.is_back_online():
                                if worker.pow_resync_chain():
                                    worker.update_model_after_chain_resync()
                        if worker.is_online():
                            # worker_last_block_hash[worker.return_idx()] = {}
                            worker.receive_block_from_miner(block_to_send)
                            verified_block = worker.verify_block(worker.return_received_block_from_miner())
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
        
        # workers do global updates
        for worker in workers_this_round:
            if not worker.is_online():
                worker.online_switcher()
                if worker.is_back_online():
                    if worker.pow_resync_chain():
                        worker.update_model_after_chain_resync()
            if worker.is_online():
                if worker.return_received_block_from_miner():
                    worker.global_update()
                    accuracy = worker.evaluate_updated_weights()
                    worker.set_accuracy_this_round(accuracy)
                    report_msg = f'Worker {worker.return_idx()} at the communication round {comm_round+1} with chain length {worker.return_blockchain_object().return_chain_length()} has accuracy: {accuracy}\n'
                    print(report_msg)
                    with open("accuracy_report.txt", "a") as file:
                        file.write(report_msg)
                    worker.online_switcher()
        
        # if not check_network_eligibility():
            # print("Go to the next round.\n")
            # continue
        
        # TODO validator may also be evil. how to validate validators?
        # or maybe in this specific settings, since devices take turns to become validators and not specifically set to some certain memvbers, we believe most of the members in the system want to benefit the whole community and trust validators by default
        # after all, they are still taking chances to send their validations to the miners
        # workers send their accuracies to validators and validators record the accuracies in a block
        # iterating validator is easier than iterating worker because of the creation of the validator block
        for validator in validators_this_round:
            if validator.is_online():
                # update peer list
                if not validator.update_peer_list():
                    # peer_list_empty, randomly register with an online node
                    if not register_in_the_network(validator, check_online=True):
                        print("No devices found in the network online in this communication round.")
                        break
                # PoW resync chain
                if validator.pow_resync_chain():
                    validator.update_model_after_chain_resync()
                last_block_on_validator_chain = validator.return_blockchain_object().return_last_block()
                # check last block
                if last_block_on_validator_chain == None or last_block_on_validator_chain.is_validator_block():
                    print("last block ineligible to be operated")
                    continue
                online_workers_in_peer_list = validator.get_online_workers()
                if not online_workers_in_peer_list:
                    print(f"Cannot find online workers in {worker.return_idx()} peer list.")
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
                if finally_no_associated_miner:
                    print(f"Cannot find a online miner in {validator.return_idx()} peer list.")
                    continue
                associated_miner.add_validator_to_association(validator)
                # may go offline
                validator.online_switcher() 
                
        # miners accept validators' transactions and broadcast to other miners
        for miner_iter in range(len(miners_this_round)):
            miner = miners_this_round[miner_iter]
            if not miner.is_online():
                miner.online_switcher()
                if miner.is_back_online():
                    if miner.pow_resync_chain():
                        miner.update_model_after_chain_resync()
            if miner.is_online():
                # update peer list
                if not miner.update_peer_list():
                    # peer_list_empty, randomly register with a online node
                    if not register_in_the_network(worker, check_online=True):
                        print("No devices found in the network online in this communication round.")
                        break
                # PoW resync chain
                if miner.pow_resync_chain():
                    miner.update_model_after_chain_resync()
                # miner accepts validator transactions
                miner.miner_reset_vars_for_new_validation_round()
                print(f"This is {miner.return_idx()} - miner {miner_iter+1}/{len(miners_this_round)} accepting validators' transactions...")
                potential_offline_validators = set()
                associated_validators = miner.return_associated_validators()
                if not associated_validators:
                    print(f"No validators are associated with miner {miner.return_idx()} for this communication round.")
                    continue
                for validator in associated_validators:
                    if validator.is_online():
                        miner.add_unconfirmmed_transaction(validator.return_validations_and_signature(comm_round))
                    else:
                        potential_offline_validators.add(validator)
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

        # miners do self and cross-validation(only validating signature at this moment)
        block_generation_time_spent = {}
        for miner in miners_this_round:
            if not miner.is_online():
                miner.online_switcher()
                if miner.is_back_online():
                    if miner.pow_resync_chain():
                        miner.update_model_after_chain_resync()
            if miner.is_online():
                last_block_on_miner_chain = miner.return_blockchain_object().return_last_block()
                # check last block
                if last_block_on_miner_chain==None or last_block_on_miner_chain.is_validator_block():
                    print("last block ineligible to be operated")
                    continue
                if miner.return_associated_validators():
                    start_time = time.time()
                    # block index starts from 1
                    candidate_block = Block(idx=miner.return_blockchain_object().return_chain_length()+1, miner_pub_key=miner.return_rsa_pub_key(), is_validator_block=True)
                    # self verification
                    for unconfirmmed_transaction in miner.return_unconfirmmed_transactions():
                        if miner.verify_transaction_by_signature(unconfirmmed_transaction):
                            unconfirmmed_transaction['tx_verified_by'] = miner.return_idx()
                            # TODO any idea?
                            unconfirmmed_transaction['mining_rewards'] = rewards
                            candidate_block.add_verified_transaction(unconfirmmed_transaction)
                            miner.receive_rewards(rewards)
                    # cross verification
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
                        # add previous hash(last block had been checked above)
                        candidate_block.set_previous_block_hash(miner.get_blockchain_object().return_last_block().compute_hash(hash_whole_block=True))
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
                        except:
                            block_generation_time_spent[miner] = float('inf')
                        mined_block.set_mining_rewards(rewards)
                        # sign the block
                        miner.sign_block(mined_block)
                        miner.set_mined_block(mined_block)

        # if not check_network_eligibility():
            # print("Go to the next round.\n")
            # continue
        # select the winning miner and broadcast its mined block
        try:
            winning_miner = min(block_generation_time_spent.keys(), key=(lambda miner: block_generation_time_spent[miner]))
        except:
            print("No block is generated in this round. Skip to the next round.")
            continue
        
        validation_block_to_propagate = winning_miner.return_mined_block()
        winning_miner.receive_rewards(block_to_propagate.return_mining_rewards())
        winning_miner.add_block(validation_block_to_propagate)
        print(f"Winning miner {winning_miner.return_idx()} will propagate its validation block.")

        if not winning_miner.update_peer_list():
            # peer_list_empty, randomly register with an online node
            if not register_in_the_network(winning_miner, check_online=True):
                print("No devices found in the network online in the peer list of winning miner. Propogated block ")
                continue

        miners_this_round = winning_miner.return_miners_eligible_to_continue()
                
        for miner in miners_this_round:
            if miner == winning_miner:
                continue
            if not miner.is_online():
                miner.online_switcher()
                if miner.is_back_online():
                    if miner.pow_resync_chain():
                        miner.update_model_after_chain_resync()
            if miner.is_online():
                # miner.set_block_to_add(block_to_propagate)
                miner.receive_propagated_validation_block(validation_block_to_propagate)
                verified_block = miner.verify_block(miner.return_propagated_validation_block())
                #last_block_hash[miner.return_idx()] = {}
                if verified_block:
                    miner.add_block(verified_block)
                    pass
                else:
                    print("Received propagated validation block is either invalid or does not fit this chain. In real implementation, the miners may continue to mine the block. In here, we just simply pass to the next miner. We can assume at least one miner will receive a valid block in this analysis model.")
                # may go offline
                miner.online_switcher()

        # miner requests worker and validator to download the validation block
        # does not matter if the miner did not append block above and send its last block, as it won't be verified
        for miner in miners_this_round:
            if not miner.is_online():
                miner.online_switcher()
                if miner.is_back_online():
                    if miner.pow_resync_chain():
                        miner.update_model_after_chain_resync()
            if miner.is_online():
                if miner.return_has_added_block(): # TODO
                    block_to_send = miner.return_blockchain_object().return_last_block()
                    associated_workers = miner.return_associated_workers()
                    associated_validators = miner.return_associated_validators()
                    associated_devices = associated_workers.union(associated_validators)
                    if not associated_devices:
                        print(f"No devices are associated with miner {miner.return_idx()} to accept the validation block.")
                        continue
                    for device in associated_devices:
                        if not device.is_online():
                            device.online_switcher()
                            if device.is_back_online():
                                if device.pow_resync_chain():
                                    device.update_model_after_chain_resync()
                        if device.is_online():
                            # worker_last_block_hash[worker.return_idx()] = {}
                            device.reset_has_added_block()
                            device.receive_block_from_miner(block_to_send)
                            verified_block = device.verify_block(device.return_received_block_from_miner())
                            if verified_block:
                                device.add_block(verified_block)
                                pass
                            else:
                                print("Received block from the associated miner is not valid or does not fit its chain. Pass to the next worker.")
                            if device.return_has_added_block():
                                block_to_propagate = device.return_blockchain_object().return_last_block()
                                if block_to_propagate.is_validator_block():
                                    device.operate_on_validator_block()
                    miner.operate_on_validator_block()

                
