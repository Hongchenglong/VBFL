import matplotlib.pyplot as plt
from os import listdir

base_logs_folder_path = "WHDY_vanilla_malicious_involved_fedavg/logs"
latest_log_folder_name = sorted([f for f in listdir(base_logs_folder_path) if not f.startswith('.')], reverse=True)[0]

log_files_folder_path = f"{base_logs_folder_path}/{latest_log_folder_name}"

all_rounds_log_files = sorted([f for f in listdir(log_files_folder_path) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1]))

# get num of devices with their maliciousness
benign_devices_idx_list = []
malicious_devices_idx_list = []
comm_1_file_path = f"{log_files_folder_path}/comm_1.txt"
file = open(comm_1_file_path,"r") 
log_whole_text = file.read() 
lines_list = log_whole_text.split("\n")
for line in lines_list:
	if line.startswith('client'):
		device_idx = line.split(":")[0].split(" ")[0]
		device_maliciousness = line.split(":")[0].split(" ")[-1]
		if device_maliciousness == 'M':
			malicious_devices_idx_list.append(f"{device_idx} {device_maliciousness}")
		else:
			benign_devices_idx_list.append(f"{device_idx} {device_maliciousness}")
	
devices_idx_list = sorted(malicious_devices_idx_list, key=lambda k: int(k.split(" ")[0].split('_')[-1])) + sorted(benign_devices_idx_list, key=lambda k: int(k.split(" ")[0].split('_')[-1]))

devices_accuracies_across_rounds = dict.fromkeys(devices_idx_list)
for client_idx, _ in devices_accuracies_across_rounds.items():
	devices_accuracies_across_rounds[client_idx] = []
round_time_record = []

for log_file in all_rounds_log_files:
	file = open(f"{log_files_folder_path}/{log_file}","r")
	log_whole_text = file.read() 
	lines_list = log_whole_text.split("\n")
	for line in lines_list:
		if line.startswith('client'):
			device_idx = line.split(":")[0].split(" ")[0]
			device_maliciousness = line.split(":")[0].split(" ")[-1]
			device_id_mali = f"{device_idx} {device_maliciousness}"
			accuracy = round(float(line.split(":")[-1]), 3)
			devices_accuracies_across_rounds[device_id_mali].append(accuracy)
		if line.startswith('comm_round_block_gen_time'):
			spent_time = round(float(line.split(":")[-1]), 2)
			round_time_record.append(spent_time)

plt.xticks(range(len(round_time_record)), [i for i in range(1, len(round_time_record) + 1)], rotation=90)
# draw graphs over all available comm rounds
for device_idx, accuracy_list in devices_accuracies_across_rounds.items():
	plt.plot(range(len(round_time_record)), accuracy_list, label=device_idx)

plt.legend(loc='best')
plt.xlabel('Comm Round')
plt.ylabel('Accuracies Across Comm Rounds')
plt.title('Learning Curve through vanilla FedAvg Comm Rounds')
plt.savefig(f"{log_files_folder_path}/learning_curve.png")
plt.show()