import os
import regex as re
import csv

LOG_PATH_COMBINED = "./log_AL_augmented_0101_1_combined_gain.txt"
LOG_PATH_LIGHT = "./log_AL_augmented_0102_1_light_gain.txt"
LOG_PATH_MEDIUM = "./log_AL_augmented_0103_1_medium_gain.txt"
LOG_PATH_STRONG = "./log_AL_augmented_0103_2_strong_gain.txt"
OUTPUT_CSV_PAHT = "./output_loss_bbox.csv"



pattern = r"Averaged stats: lr: [\d\.]+  loss: [\d\.]+ \(([\d\.]+)\)  loss_bbox: [\d\.]+ \(([\d\.]+)\)"

def get_stat_dict(path):
    best_stat = {}
    i = 0
    with open(path, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                loss = float(match.group(2))
                best_stat[i] = loss
                i += 1
    return best_stat

def output_csv(combined_dict, light_dict, medium_dict, strong_dict):
    fieldnames = ['epoch', 'combined', 'light', 'medium', 'strong']
    data = []
    for epoch in combined_dict.keys():
        print(f"{epoch}, {combined_dict[epoch]}, {light_dict[epoch]}, {medium_dict[epoch]}, {strong_dict[epoch]}")
        data.append({
            'epoch': epoch,
            'combined': combined_dict[epoch],
            'light': light_dict[epoch],
            'medium': medium_dict[epoch],
            'strong': strong_dict[epoch]
        })
    with open(OUTPUT_CSV_PAHT, mode='w') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)
        
combined_dict = get_stat_dict(LOG_PATH_COMBINED)
light_dict = get_stat_dict(LOG_PATH_LIGHT)
medium_dict = get_stat_dict(LOG_PATH_MEDIUM)
strong_dict = get_stat_dict(LOG_PATH_STRONG)
output_csv(combined_dict, light_dict, medium_dict, strong_dict)