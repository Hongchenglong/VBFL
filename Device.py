import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from DatasetLoad import DatasetLoad
import random
import copy
# https://cryptobook.nakov.com/digital-signatures/rsa-sign-verify-examples
from Crypto.PublicKey import RSA
from hashlib import sha512
from Blockchain import Blockchain

class Device:
    def __init__(self, idx, assigned_ds, network_stability, dev):
        self.idx = idx
        self.train_ds = assigned_ds
        self.dev = dev
        self.train_dl = None
        self.local_parameters = None
        self.global_weights = None
        # used to assign role to the device
        self.role = None
        ''' simulating hardware equipment strength, such as good processors and RAM capacity
        # for workers, meaning the number of epochs it can perform for a communication round
        # for miners, its PoW time will be shrink by this value of times
        # for validators, haha! # TODO
        '''
        self.computation_power = random.randint(0, 4)
        self.peer_list = set()
        # used in cross_verification and in the PoS
        self.on_line = False
        self.rewards = 0
        self.network_stability = 0
        self.blockchain = Blockchain()
        # init key pair
        self.modulus = None
        self.private_key = None
        self.public_key = None
        self.generate_rsa_key()
        ''' For miners '''
        self.associated_worker_set = set()
        self.unconfirmmed_transactions = set()
        self.broadcasted_transactions = set()
        self.mined_block = None

    ''' Common Methods '''
    def get_idx(self):
        return self.idx

    def generate_rsa_key(self):
        keyPair = RSA.generate(bits=256)
        self.modulus = keyPair.n
        self.private_key = keyPair.d
        self.public_key = keyPair.e
    
    def sign_msg(self, msg):
        hash = int.from_bytes(sha512(msg).digest(), byteorder='big')
        # pow() is python built-in modular exponentiation function
        signature = pow(hash, self.private_key, self.modulus)
        return hex(signature)

    def init_global_weights(self, global_weights):
        self.global_weights = global_weights

    def add_peers(self, new_peers):
        if type(new_peers) == str:
            self.peer_list.add(new_peers)
        else:
            self.peer_list.update(new_peers)

    def get_peers(self):
        return self.peer_list
    
    def remove_peers(self, peers_to_remove):
        if type(peers_to_remove) == str:
            self.peer_list.remove(peers_to_remove)
        else:
            self.peer_list.difference_update(peers_to_remove)

    def assign_role(self):
        # equal probability
        role_choice = random.randint(0, 2)
        if role_choice == 0:
            self.role = "w"
        elif role_choice == 1:
            self.role = "m"
        else:
            self.role = "v"
        
    def get_role(self):
        return self.role

    def online_switcher(self):
        online_indicator = random.random()
        if online_indicator < self.network_stability:
            self.on_line = True
        else:
            self.on_line = False
        return self.on_line

    def is_online(self):
        return self.on_line
    
    def update_peer_list(self):
        original_peer_list = copy.deepcopy(self.peer_list)
        for peer in self.peer_list:
            if peer.is_online(original_peer_list):
                self.add_peers(peer.get_peers())
            else:
                self.remove_peers(peer)
        # remove itself from the peer_list if there is
        self.remove_peers(f'device_{self.idx}')
        # if peer_list ends up empty, randomly register with another device
        return False if not self.peer_list else True

    def get_chain(self):
        return self.blockchain

    def check_pow_proof(block_to_check, pow_proof):
        return pow_proof.startswith('0' * Blockchain.difficulty) and pow_proof == block_to_check.compute_hash()

    def check_chain_validity(self, chain_to_check):
        chain_len = chain_to_check.get_chain_length()
        if chain_len == 0 or chain_len == 1:
            pass
        else:
            for block in chain_to_check[1:]:
                if self.check_pow_proof(block, block.get_block_hash()) and block.get_previous_hash == chain_to_check[chain_to_check.index(block) - 1].compute_hash(hash_previous_block=True):
                    pass
                else:
                    return False
        return True

    def pow_resync_chain(self):
        longest_chain = None
        for peer in self.peer_list:
            if peer.is_online():
                if peer.get_chain().get_chain_length() > self.blockchain.get_chain_length():
                    if self.check_chain_validity(peer_chain):
                        # Longer valid chain found!
                        curr_chain_len = peer_chain_length
                        longest_chain = peer_chain
        if longest_chain:
            self.blockchain.replace_chain(longest_chain)
            print("chain resynced")

    def receive_rewards(self, rewards):
        self.rewards += rewards
    
    def get_computation_power(self):
        return self.computation_power

    ''' Worker '''
    # TODO change to computation power
    def worker_local_update(self, localBatchSize, Net, lossFun, opti, global_parameters):
        print(f"computation power {self.computation_power}, performing {self.computation_power} epochs")
        Net.load_state_dict(global_parameters, strict=True)
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)
        for epoch in range(self.computation_power):
            print(f"epoch {epoch+1}")
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = Net(data)
                loss = lossFun(preds, label)
                loss.backward()
                opti.step()
                opti.zero_grad()
        print("Done")
        self.local_parameters = Net.state_dict()

    def get_local_updates_and_signature(self):
        return {"local_updates_params": self.local_parameters, "signature": self.sign_updates()}

    def associate_with_miner(self):
        online_miners_in_peer_list = set()
        for peer in peer_list:
            if peer.is_online():
                if peer.get_role == 'm':
                    online_miners_in_peer_list.add(peer)
        if not online_miners_in_peer_list:
            return False
        self.worker_associated_miner = random.sample(online_miners_in_peer_list, 1)[0]
        return self.worker_associated_miner

    def sign_updates(self):
        return {"pub_key": self.public_key, "modulus": self.modulus, "signature": self.sign_msg(self.local_parameters.__dict__)}

    # def local_val(self):
    #     pass

    ''' miner '''
    def add_worker_to_association(self, worker_device):
        self.associated_worker_set.add(worker_device)

    def clear_worker_association(self):
        self.associated_worker_set.clear()

    def get_associated_workers(self):
        return self.associated_worker_set

    def clear_unconfirmmed_transactions(self):
        self.unconfirmmed_transactions.clear()

    def clear_broadcasted_transactions(self):
        self.broadcasted_transactions.clear()
    
    def add_unconfirmmed_transaction(self, add_unconfirmmed_transaction):
        self.broadcasted_transactions.add(add_unconfirmmed_transaction)

    def get_unconfirmmed_transactions(self):
        return self.unconfirmmed_transactions

    def accept_broadcasted_transactions(self, broadcasted_transactions):
        self.broadcasted_transactions.add(broadcasted_transactions)

    def broadcast_updates(self):
        for peer in self.peer_list:
            if peer.is_online():
                if peer.get_role == 'm':
                    peer.accept_broadcasted_transactions(self.unconfirmmed_transactions)

    def get_broadcasted_transactions(self):
        return self.broadcasted_transactions

    def sign_block(self, mined_block):
        mined_block['signature'] = self.sign_msg(mined_block.__dict__)

    def verify_transaction_by_signature(self, transaction_to_verify)
        local_updates_params = transaction_to_verify['local_updates']["local_updates_params"]
        modulus = transaction_to_verify['local_updates']["signature"]["modulus"]
        pub_key = transaction_to_verify['local_updates']["signature"]["pub_key"]
        signature = transaction_to_verify['local_updates']["signature"]["signature"]
        # verify
        hash = int.from_bytes(sha512(local_updates_params.__dict__).digest(), byteorder='big')
        hashFromSignature = pow(signature, pub_key, modulus)
        return hash == hashFromSignature
    
    def proof_of_work(self, candidate_block, starting_nonce=0):
        ''' Brute Force the nonce '''
        candidate_block.set_nonce(starting_nonce)
        current_hash = candidate_block.compute_hash()
        while not current_hash.startswith('0' * Blockchain.pow_difficulty):
            candidate_block.nonce_increment()
            current_hash = candidate_block.compute_hash()
        # return the qualified hash as a PoW proof, to be verified by other devices before adding the block
        # also set its hash as well. block_hash is the same as pow proof
        candidate_block.set_hash()
        return candidate_block

    def set_mined_block(self, mined_block):
        self.mined_block = mined_block

    def reset_mined_block(self):
        self.mined_block = None

    def get_mined_block(self):
        return self.mined_block

class DevicesInNetwork(object):
    def __init__(self, data_set_name, is_iid, num_of_devices, network_stability, dev):
        self.data_set_name = data_set_name
        self.is_iid = is_iid
        self.num_of_devices = num_of_devices
        self.dev = dev
        self.devices_set = {}
        self.test_data_loader = None
        self.default_network_stability = network_stability
        # distribute dataset
        self.data_set_balanced_allocation()

    # distribute the dataset evenly to the devices
    def data_set_balanced_allocation(self):
        # read dataset
        mnist_dataset = DatasetLoad(self.data_set_name, self.is_iid)
        # perpare test data
        test_data = torch.tensor(mnist_dataset.test_data)
        test_label = torch.argmax(torch.tensor(mnist_dataset.test_label), dim=1)
        self.test_data_loader = DataLoader(TensorDataset( test_data, test_label), batch_size=100, shuffle=False)
        # perpare training data
        train_data = mnist_dataset.train_data
        train_label = mnist_dataset.train_label
        # shard dataset and distribute among devices
        shard_size = mnist_dataset.train_data_size // self.num_of_devices // 2
        shards_id = np.random.permutation(mnist_dataset.train_data_size // shard_size)
        for i in range(self.num_of_devices):
            # make it more random by introducing two shards
            shards_id1 = shards_id[i * 2]
            shards_id2 = shards_id[i * 2 + 1]
            data_shards1 = train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            data_shards2 = train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            label_shards1 = train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            label_shards2 = train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            local_data, local_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
            local_label = np.argmax(local_label, axis=1)
            # assign data to a device and put in the devices set
            a_device = Device(i+1, TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.default_network_stability, self.dev)
            # device index starts from 1
            self.devices_set[f'device_{i+1}'] = a_device