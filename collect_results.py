import argparse
import os, glob
import json

parser = argparse.ArgumentParser()

parser.add_argument("--dir", default="results/baseline/cora", type=str)
parser.add_argument("--outfile_name", default="cora_baseline.json", type=str)
args = parser.parse_args()

log_dir = 'experiment_results/'
if os.path.exists(log_dir) is False:
    os.makedirs(log_dir)

all_data = {"layers": [],
            "test_acc": [],
            "val_acc": [],
            "row_diff": [],
            "col_diff": [],
            "iig": [],
            "gdr": []}
all_test = []
nlayers = []
all_val = []
all_row_diff = []
all_col_diff = []
all_iig, all_gdr = [], []
for filename in glob.glob(args.dir + '/*.json'):
    filename_only = filename.split("/")[-1]
    nlayer = int(filename_only.split("_")[2])
    f = open(filename, "r")
    data = json.loads(f.read())
    test_acc = data["Test acc"]
    val_acc = data["Val. accuracy"]
    row_diff = data["row-diff"]
    col_diff = data["col-diff"]
    iig, gdr = data["iig"], data["gdr"]
    nlayers.append(nlayer)
    all_test.append(test_acc)
    all_val.append(val_acc)
    all_row_diff.append(row_diff)
    all_col_diff.append(col_diff)
    all_iig.append(iig)
    all_gdr.append(gdr)

all_data["test_acc"] = [x for _, x in sorted(zip(nlayers, all_test))]
all_data["val_acc"] = [x for _, x in sorted(zip(nlayers, all_val))]
all_data["row_diff"] = [x for _, x in sorted(zip(nlayers, all_row_diff))]
all_data["col_diff"] = [x for _, x in sorted(zip(nlayers, all_col_diff))]
all_data["iig"] = [x for _, x in sorted(zip(nlayers, all_iig))]
all_data["gdr"] = [x for _, x in sorted(zip(nlayers, all_gdr))]
all_data["layers"] = sorted(nlayers)

save_filename = log_dir + args.outfile_name
with open(save_filename, 'w') as outfile:
    json.dump(all_data, outfile, indent=4)
    print("Val/Test acc. saved in json file:", args.outfile_name)
