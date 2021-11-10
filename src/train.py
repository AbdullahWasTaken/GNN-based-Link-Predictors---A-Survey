import json
import torch
from dataset import DATASET
from model import MODEL

##########
# CONFIG #
##########
CONFIG = json.load(open("config.json", "r") )
# CONFIG = yaml.safe_load(open("config.yaml", "r"))

###########
# DATASET #
###########
dataset = DATASET(CONFIG["DATASET"])
train_data, val_data, test_data = dataset[0]

#########
# MODEL #
#########
CONFIG["MODEL"]["ENCODER"]["IN_LAYERS"] = dataset.num_features
model = MODEL(CONFIG["MODEL"])

###########
# Trainer #
###########
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    loss = model.recon_loss(z, train_data.pos_edge_label_index)
    # if args.variational:
    #     loss = loss + (1 / train_data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)

########
# Test #
########
@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)

#################
# Training Loop #
#################
result = {
    "VAL AUC"  : [],
    "VAL AP"   : [],
    "TEST AUC" : [],
    "TEST AP"  : [],
    "LOSS"     : []
}
stats = None
for epoch in range(1, CONFIG["EPOCH"] + 1):
    loss = train()
    result["LOSS"].append(loss)
    val_auc, val_ap = test(val_data)
    result["VAL AUC"].append(val_auc)
    result["VAL AP"].append(val_ap)
    test_auc, test_ap = test(test_data)
    result["TEST AUC"].append(test_auc)
    result["TEST AP"].append(test_ap)
    display = 'Epoch: {:03d}, LOSS: {:.4f}, VAL AUC: {:.4f}, VAL AP: {:.4f}, TEST AUC: {:.4f}, TEST AP: {:.4f}'\
        .format(epoch, loss, val_auc, val_ap, test_auc, test_ap)
    print(display)

##############
# Checkpoint #
##############
import os
import pandas as pd

PATH= "/home/revelo/GNN Based Link-Prediction/checkpoints"
df = pd.read_csv(PATH+"/configTable.csv", index_col=0)
cfg = json.dumps(CONFIG)

id = None
num = 1

if not cfg in df["config"].values.tolist():
    id = "config_"+str(len(df))
    df.loc[len(df)] = [id, cfg, num]
    os.mkdir(PATH+"/"+id)
else:
    id = df.loc[df["config"] == cfg, "ID"].values[0]
    num = df.loc[df["config"] == cfg, "num"].values[0] + 1
    df.loc[df["config"] == cfg, "num"] = num

df.to_csv(PATH+"/configTable.csv")
# save result
pd.DataFrame(result).to_csv(PATH+"/"+id+"/"+str(num)+".csv")

#################
# Visualization #
#################
import matplotlib.pyplot as plt

x = list(range(CONFIG["EPOCH"]))
plt.plot(x, result["VAL AUC"], label="Val AUC ROC")
plt.plot(x, result["VAL AP"], label="Val AP")
plt.plot(x, result["TEST AUC"], label="Test AUC ROC")
plt.plot(x, result["TEST AP"], label="Test AP")
plt.plot(x, result["LOSS"], label="Loss")
plt.legend()
plt.savefig("plot.png")
plt.close()
