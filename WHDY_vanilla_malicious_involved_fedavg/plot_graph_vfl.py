import matplotlib.pyplot as plt
from os import listdir

base_logs_folder_path = "WHDY_vanilla_malicious_involved_fedavg/logs"
latest_log_folder_name = sorted([f for f in listdir(base_logs_folder_path) if not f.startswith('.')], reverse=True)[0]

log_files_folder_path = f"{base_logs_folder_path}/{latest_log_folder_name}"

all_rounds_log_files = sorted([f for f in listdir(log_files_folder_path) if f.startswith('comm')])

# get num of devices
comm_1_file_path = f"{log_files_folder_path}/comm_1.txt"
file = open(comm_1_file_path,"r") 
Counter = 0
# Reading from file 
Content = file.read() 
CoList = Content.split("\n") 
for i in CoList: 
    if i: 
        Counter += 1
device_count = Counter - 1
# done getting num of devices

devices_accuracies_across_rounds = dict.fromkeys([f"client_{i}" for i in range(1, device_count + 1)])
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
            accuracy = round(float(line.split(":")[-1]), 3)
            devices_accuracies_across_rounds[device_idx].append(accuracy)
        if line.startswith('comm_spent_time'):
            spent_time = round(float(line.split(":")[-1]), 2)
            round_time_record.append(spent_time)

# draw graphs over all available comm rounds
for device_idx, accuracy_list in devices_accuracies_across_rounds.items():
    plt.xticks(range(len(round_time_record)), round_time_record)
    plt.plot(range(len(round_time_record)), accuracy_list, label=device_idx)

plt.legend(loc='best')
plt.xlabel('Comm Round with Spent Time')
plt.ylabel('Accuracies Across Comm Rounds')
plt.title('Learning Curve through Vanilla FedAvg Comm Rounds')
plt.show()
print()