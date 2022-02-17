import numpy as np

filename = "nyudepthv2_train_files_with_gt.txt"
with open(filename) as f:
    lines = f.readlines()

assign = np.random.rand(len(lines))
assign = np.floor(assign * 5.)

for i in range(5):
    train = open("nyu-train-fold-" + str(i+1) + ".txt", "w")
    test = open("nyu-test-fold-" + str(i+1) + ".txt", "w")
    for j in range(len(lines)):
        if int(assign[j]) == i: test.write(lines[j])
        else: train.write(lines[j])
    train.close(); test.close()