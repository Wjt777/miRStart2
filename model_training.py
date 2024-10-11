import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(filename='/share/home/grp-huangxd/wanjingting/miRStart2/CAGE_20_output.log', level=logging.INFO, format='%(asctime)s - %(message)s')


#torch.backends.cudnn.enabled = False
torch.cuda.empty_cache()

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.set_device(0) 
    #print("Current GPU:", torch.cuda.current_device())
    logging.info("Current GPU: %d", torch.cuda.current_device())
else:
    device = torch.device("cpu")
    #print("Using CPU.")
    logging.info("Using CPU.")
    

def rearrange_data_high_res(output):

    array_format = np.append(output.columns, np.array(output)).reshape(-1, 111)
    X = []
    y = []
    x = []
    for arr in array_format:
        y.append(arr[0])
        X.append(arr[1:])
    
    X = np.array(X)
    y = np.array(y)
    feature = np.char.split(X.astype(str), ':')
    for f in feature:
        x.append(np.array([list(row) for row in f]).T[1])
    features = np.array(x).astype(float)
    labels = np.array(y).astype(float)
        
    return features, labels


def data_split(X, y):
    train_data, test_X, train_labels, test_y = train_test_split(X, y, test_size=0.25, random_state=42)
    train_X, val_X, train_y, val_y = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)
    
    return train_X, train_y, val_X, val_y, test_X, test_y


indices_to_reverse = np.load('/share/home/grp-huangxd/wanjingting/miRStart2/indices_to_reverse_human.npy')

CAGE = pd.read_csv('CAGE/output_svm_format.txt', sep = ' ')
DBTSS = pd.read_csv('DBTSS/output_svm_format.txt', sep = ' ')
DNase = pd.read_csv('DNase/output_svm_format.txt', sep = ' ')
H3K4me3 = pd.read_csv('H3K4me3/output_svm_format.txt', sep = ' ')
Pol2 = pd.read_csv('Pol2/output_svm_format.txt', sep = ' ')

    
CAGE_X, CAGE_y = rearrange_data_high_res(CAGE)
DBTSS_X, DBTSS_y = rearrange_data_high_res(DBTSS)
DNase_X, DNase_y = rearrange_data_high_res(DNase)
H3K4me3_X, H3K4me3_y = rearrange_data_high_res(H3K4me3)
Pol2_X, Pol2_y = rearrange_data_high_res(Pol2)


for index in indices_to_reverse:
    CAGE_X[index] = CAGE_X[index][::-1]
    DBTSS_X[index] = DBTSS_X[index][::-1]
    DNase_X[index] = DNase_X[index][::-1]
    H3K4me3_X[index] = H3K4me3_X[index][::-1]
    Pol2_X[index] = Pol2_X[index][::-1] 
       
indices_to_reverse_neg = indices_to_reverse + 59777
for index in indices_to_reverse_neg:
    CAGE_X[index] = CAGE_X[index][::-1]
    DBTSS_X[index] = DBTSS_X[index][::-1]
    DNase_X[index] = DNase_X[index][::-1]
    H3K4me3_X[index] = H3K4me3_X[index][::-1]
    Pol2_X[index] = Pol2_X[index][::-1] 


data = np.concatenate([np.expand_dims(CAGE_X, axis=2), 
                                np.expand_dims(H3K4me3_X, axis=2), 
                                np.expand_dims(DNase_X, axis=2), 
                                np.expand_dims(DBTSS_X, axis=2),
                                np.expand_dims(Pol2_X, axis=2)], 
                                #np.expand_dims(TFBS_X, axis=2)], 
                                axis=2)

data = np.transpose(data, (0, 2, 1))

zero_index = np.where(data.sum(axis = (1,2)) == 0)[0]
positive_indices = np.where(CAGE_y == 1)[0]
negative_indices = np.where(CAGE_y == -1)[0]

pos_zero = zero_index[np.where(np.isin(zero_index, positive_indices))]
neg_zero = zero_index[np.where(np.isin(zero_index, negative_indices))]


mask = np.ones(data.shape[:1:2], dtype = bool)
mask[pos_zero] = False
mask[neg_zero[:len(pos_zero)]] = False
data1 = data[mask]
CAGE_y1 = CAGE_y[mask]


    
train_X, train_y, val_X, val_y, test_X, test_y = data_split(data1, np.where(CAGE_y1 == -1, 0, 1))

train_data_tensor = torch.Tensor(train_X)
train_labels_tensor = torch.LongTensor(train_y)

val_data_tensor = torch.Tensor(val_X)
val_labels_tensor = torch.LongTensor(val_y)

test_data_tensor = torch.Tensor(test_X)
test_labels_tensor = torch.LongTensor(test_y)


batch_size = 32
train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(val_data_tensor, val_labels_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = TensorDataset(test_data_tensor, test_labels_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.conv1d_1 = nn.Conv1d(in_channels = 5, out_channels = 32, kernel_size = 60, stride = 5)
        self.conv1d_2 = nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=1)
        self.fc1 = nn.Linear(64*2, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, 2)

    def forward(self, x):
        x = self.conv1d_1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv1d_2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        #print(x)
        return x



# Define training function
def train(model, train_loader, val_loader, num_epochs = 100, lr = 0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            # Evaluate the model on independent validation set
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            correct = 0
            total = 0
            for batch_data, batch_labels in val_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

            val_accuracy = correct / total
            avg_val_loss = val_loss / len(val_loader)

            #print(f'Epoch [{epoch + 1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy * 100:.2f}%')
            logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy * 100:.2f}%')
                

# Define testing function 
def test(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        correct = 0
        total = 0
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        
        for batch_data, batch_labels in test_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            
            TP += ((predicted == 1) & (batch_labels == 1)).sum().item()
            TN += ((predicted == 0) & (batch_labels == 0)).sum().item()
            FP += ((predicted == 1) & (batch_labels == 0)).sum().item()
            FN += ((predicted == 0) & (batch_labels == 1)).sum().item()           
            
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

        test_accuracy = correct / total
        avg_test_loss = test_loss / len(test_loader)
        
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        f1_score = 2 * precision * recall / (precision + recall)

        #print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%')
        #print(precision, recall, accuracy, f1_score)
        logging.info(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%')
        logging.info(f'Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1_score:.4f}')



# Train the model
model = CNN1D()
train(model, train_loader, val_loader, num_epochs = 200, lr = 0.001)


# Test the model on independent test set
test(model, test_loader)
