# AL-ITS

```
Usage:
    main.py --model=<model_name> --dataset=<dataset_name>  [options]

        <model_name>  The model to be used in [DKT, DKTEnhanced]
        <dataset_name>  Dataset to be used in [STATICS, assist2009, assist2015]

Options:
    -h --help                               show this screen.
    --gpu=<int>                             use GPU [default: 0]
    --batch-size=<int>                      batch size [default: 4]
    --train-ratio=<float>                   ratio of training set [default: 0.8]
    --hidden-size=<int>                     hidden size [default: 100]
    --dropout=<float>                       dropout [default: 0.5]
    --lambda-r=<float>                      lambda_r [default: 0.01]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --max-epoch=<int>                       max epoch [default: 200]
    --max-patience=<int>                    wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --lr=<float>                            learning rate [default: 0.01]
    --model-save-to=<file>                  model save path [default: outputs/model/]
```

