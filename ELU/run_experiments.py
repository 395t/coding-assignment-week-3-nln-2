import csv 
import subprocess

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--file", type=str, default="experiments.csv")
    args = parser.parse_args()
    file = args.file

    batch_norm_mapping = {'TRUE': "--batch_norm", 'FALSE': ""}
    CMD = "python3 -m Model.train --log_dir ./logs -e 30 -a {} {}"
    with open(file, newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        with open("Experiments/results.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Activation Function","Batch Norm","Accuracy","Time"])
            for row in reader:
                command = CMD.format(str(row['activation_functions']), batch_norm_mapping[row['batch_norm']])
                print(command)
                out = subprocess.check_output(command, shell=True)
                out = out.decode()
                writer.writerow(out.split(","))
                

