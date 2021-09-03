import torch
import transformers 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import os, time
import numpy as np


class torch_trainer():
    
    def __init__(self, name="result"):
        self.path = name+"/"
        
        if not os.path.exists(name):
            os.makedirs(name)
            print("Created folder " + name)
        else:
            print("Warning!! folder " + name + " is existed")
            
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("will use " + str(self.device))
        
        self.model_cls = None
        
    
    def set_model_cls(self, model_cls):
        '''
        Input:
            model_cls: torch class
        '''
        self.model_cls = model_cls
        print("model is setted")

    def set_model_parameter(self, args):
        '''
        Input:
            arg: argumentation for creating torch model (type: dict)
        '''
        self.parameter = args
        print("parameter is setted")
        
        
    def do_cross_validation(self, dataset, k, batch, batch_fn, epochs):
        '''
        doing cross_validation 
        Input:
            dataset: torch dataset
            k: split dataset into k flods
            batch: batch size
            batch_fn: collate_fn of dataloader
            epochs: training epochs
        '''
        # KFold from sklearn
        kfold = KFold(n_splits=k, shuffle=True)
        
        # for storing score of each flod's test
        Score = []
        
        # just for print the other flod's idx
        flod_id = [i for i in range(1, k+1)]
        
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
            
            # create/recreate a new model to prevent using the weight of previous training step
            print("initaial a model ...", end = "")
            model = self.model_cls(**self.parameter).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            print("done")
            
            # create dataloader with ids from KFold
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
            
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch,\
                                                            collate_fn=batch_fn, shuffle=True)
            val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch,\
                                                           collate_fn=batch_fn, shuffle=False)
            
            # start training
            show_idxs = flod_id[::]
            show_idxs.remove(fold+1)
            print('Fold '+ str(show_idxs)  +' as traing set')
            print('start training...')
            
            for epoch in range(epochs):
                train_res = self.train(train_loader, model, optimizer)
                print("Epoch: " + str(epoch+1) + " Train Loss: " + str(train_res))
            
            print("... done")

            # test after training
            print("start testing...", end = "")
            val_res, acc = self.test(val_loader, model)
            print("done")
            
            print('Fold '+ str(fold+1) +' Val Loss: '+str(val_res))
            print('Fold '+ str(fold+1) +' Val Acc: '+str(acc))
            
            # record the score
            Score.append(acc)
            print("-"*80)
        
        # return the mean of Score
        print("Score:", np.mean(Score))
        return np.mean(Score)
            
    def train_find_best_epoch(self, train_ds, test_ds, batch, batch_fn, epochs):
        ''' 
        Input:
            train_ds: torch dataset
            test_ds: torch dataset
            k: split dataset into k flods
            batch: batch size
            batch_fn: collate_fn of dataloader
            epochs: training epochs
        '''
        
        # creating a model
        print("Creating model ...", end = "")
        model = self.model_cls(**self.parameter).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        print("done")
        
        # creat dataloader
        print("Creating dataloader ...", end="")
        
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch,\
                                                   collate_fn=batch_fn, shuffle=True)
        
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch*2,\
                                                  collate_fn=batch_fn, shuffle=False)
        print("done")
     
        # store the train loss and test loss for each epoch
        train_losses = []
        test_losses = []
        # record the best epoch
        best = 0
        best_acc = 0.0
        
        # start training 
        print("Starting training ...")
        print("-"*80)
        for epoch in range(epochs):
            
            # train step
            train_res = self.train(train_loader, model, optimizer)
            print("Epoch: " + str(epoch+1) + " Train Loss: " + str(train_res))
            
            # test step
            test_res, acc = self.test(test_loader, model)
            print("Epoch: " + str(epoch+1) + " Val Loss: " + str(test_res))
            
            # record loss
            train_losses.append(train_res)
            test_losses.append(test_res)
            
            # if test_res is the lowest in current losses, store the model
            if test_res <= min(test_losses):
                print("Epoch "+str(epoch+1)+ " is current best!!!  test acc: " + str(acc) )
                torch.save(model.state_dict(), self.path + "best.pt") 
                print("save model to " + self.path + "best.pt")
                best = epoch+1
                best_acc = acc
            print("-"*80)
            self.draw_loss(train_losses, test_losses)
        
        
        print("...Epoch "+str(best)+ " is best!!! acc: " + str(best_acc))

    
    def train(self, loader, model, optimizer):
        '''
        normal training step for pytorch model
        '''
        
        # set the model to train mode
        model.train()
        
        # record loss for each batch
        losses = []

        for data in loader:
            # retrive input of model from dataloader
            input_tensor, mask_tensor, target_tensor = data
            
            # put input into same device as model is (cpu or gpu)
            input_tensor = input_tensor.to(self.device)
            target_tensor = target_tensor.to(self.device)
            mask_tensor = mask_tensor.to(self.device)
            
            
            optimizer.zero_grad()

            loss, pred = model(input_tensor, mask_tensor, target_tensor)
            losses.append(loss.item())
            loss.backward()

            optimizer.step()
            m_loss= np.mean(losses)

        return np.mean(losses) 
                
                
    def test(self, loader, model):
        model.eval()
        
        preds = []
        targets = []
        losses = []
        
        with torch.no_grad():
            for data in loader:
                input_tensor, mask_tensor, target_tensor = data

                input_tensor = input_tensor.to(self.device)
                target_tensor = target_tensor.to(self.device)
                mask_tensor = mask_tensor.to(self.device)

                loss, pred = model(input_tensor, mask_tensor, target_tensor)

                losses.append(loss.item())
                preds += pred.tolist()
                targets += target_tensor.tolist()
                break
                
#         print(classification_report(targets, preds))

        acc = sum(1 for x,y in zip(preds,targets) if x == y) / float(len(preds))
        return np.mean(losses), acc 
    
    
    def draw_loss(self, y1, y2):
        x = [i+1 for i in range(len(y1))]
        plt.plot(x, y1, label='train loss')
        plt.plot(x, y2, label='eval loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(self.path+"loss.png")
        plt.close()