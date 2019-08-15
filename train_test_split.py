import pickle
import glob
folder_name ="CLEVR_64_36_FINAL_DATASET"
tfrs_files = glob.glob("/projects/katefgroup/datasets/clevr/{}/tfrs/*".format(folder_name))
print(len(tfrs_files))
trainindex = int(len(tfrs_files)*0.9)
train_data  = tfrs_files[:trainindex]
test_data  = tfrs_files[trainindex:]

train_file = "/projects/katefgroup/datasets/clevr/{}/train.p".format(folder_name)
test_file = "/projects/katefgroup/datasets/clevr/{}/test.p".format(folder_name)
pickle.dump(train_data,open(train_file,"wb"))
pickle.dump(test_data,open(test_file,"wb"))
