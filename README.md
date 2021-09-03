# torch-trainer
A general model for training and doing k fold cross validation

## usage
just download and import
```

```
## example
see notebook:

```
trainer = torch_trainer()
# torch calss
trainer.set_model_cls(spam_classifer1)

# parameter for your model
trainer.set_model_parameter({"pretrain_model":"bert-base-uncased", "drop":0.3})
```
## training
```
trainer.train_find_best_epoch(train_ds, test_ds, batch=20, batch_fn=create_mini_batch, epochs=3)
```
output 
```
Creating model ...done
Creating dataloader ...done
Starting training ...
--------------------------------------------------------------------------------
Epoch: 1 Train Loss: 0.45061845442639337
Epoch: 1 Val Loss: 0.4632672667503357
Epoch 1 is current best!!!  test acc: 0.84
--------------------------------------------------------------------------------
Epoch: 2 Train Loss: 0.44783694006402397
Epoch: 2 Val Loss: 0.4632638394832611
Epoch 2 is current best!!!  test acc: 0.85
--------------------------------------------------------------------------------
Epoch: 3 Train Loss: 0.4477944088089092
Epoch: 3 Val Loss: 0.4632629454135895
Epoch 3 is current best!!!  test acc: 0.85
--------------------------------------------------------------------------------
...Epoch 3 is best!!! acc: 0.85
```
it will save the model to path "/result"

## k fold cross validation

do 5 fold cross validation on spam_classifer1

```
trainer.set_model_cls(spam_classifer1) 
trainer.set_model_parameter({"pretrain_model":"bert-base-uncased", "drop":0.3})
trainer.do_cross_validation(train_ds, k=5, batch=20, batch_fn=create_mini_batch, epochs=3)
```

output
```
model is setted
parameter is setted
initaial a model ...done
Fold [2, 3, 4, 5] as traing set
start training...
Epoch: 1 Train Loss: 0.4487916386180929
Epoch: 2 Train Loss: 0.44787454097260276
Epoch: 3 Train Loss: 0.4479121997484712
... done
start testing...done
Fold 1 Val Loss: 0.46326225996017456
Fold 1 Val Acc: 0.85
--------------------------------------------------------------------------------
initaial a model ...done
Fold [1, 3, 4, 5] as traing set
start training...
Epoch: 1 Train Loss: 0.4488696075073807
Epoch: 2 Train Loss: 0.44779364517451403
Epoch: 3 Train Loss: 0.447871749978429
... done
start testing...done
Fold 2 Val Loss: 0.46326225996017456
Fold 2 Val Acc: 0.85
--------------------------------------------------------------------------------
initaial a model ...done
Fold [1, 2, 4, 5] as traing set
start training...
Epoch: 1 Train Loss: 0.4495103778860494
Epoch: 2 Train Loss: 0.44795267303962877
Epoch: 3 Train Loss: 0.44795126230727395
... done
start testing...done
Fold 3 Val Loss: 0.4632623791694641
Fold 3 Val Acc: 0.85
--------------------------------------------------------------------------------
initaial a model ...done
Fold [1, 2, 3, 5] as traing set
start training...
Epoch: 1 Train Loss: 0.449252583253544
Epoch: 2 Train Loss: 0.4477949693063984
Epoch: 3 Train Loss: 0.44787237823276776
... done
start testing...done
Fold 4 Val Loss: 0.4632626473903656
Fold 4 Val Acc: 0.85
--------------------------------------------------------------------------------
initaial a model ...done
Fold [1, 2, 3, 4] as traing set
start training...
Epoch: 1 Train Loss: 0.44927673422702225
Epoch: 2 Train Loss: 0.4479125230301656
Epoch: 3 Train Loss: 0.4477927361368598
... done
start testing...done
Fold 5 Val Loss: 0.46326231956481934
Fold 5 Val Acc: 0.85
--------------------------------------------------------------------------------
Score: 0.85
```
