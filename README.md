# AUC-ROC Maximization for Medical data

## Model selection

1. **Grid search to tune hyperparameters using images with size 28.**

    For each combination of hyperparameters, we evaluate the validation AUC every 5 epochs and select the epoch with the highest validation score.
    For each dataset, we select the combination of hyperparameters that achieves the highest validation AUC.

    (The flag `--use_best_model` saves as `last.ckpt` the epoch with the highest validation AUC)

    For **ChestMNIST**, we use early stopping (patience 50 epochs) and mixed-16 precision for faster training. The following grid search is used:
    - learning rate: 1e-2
    - batch size: 128
    - weight decay: 1e-4, 1e-5
    - epoch decay: 3e-2
    - margin: 1.0, 0.6

    ```
    ./scripts/experiments/auc_chest.sh
    ```

    For the other **2D datasets**, the following is used:
    - learning rate: 1e-1
    - batch size: 128
    - weight decay: 1e-3, 1e-4, 1e-5
    - epoch decay: 3e-2
    - margin: 1.0, 0.8, 0.6

    ```
    ./scripts/experiments/auc_2d.sh
    ```

    For the **3D datasets**, the following is used:
    - learning rate: 1e-1, 1e-2
    - batch size: 32
    - weight decay: 1e-4, 1e-5
    - epoch decay: 3e-2, 3e-3
    - margin: 1.0, 0.6

    ```
    ./scripts/experiments/auc_3d.sh
    ```

2. **Some datasets will use image size 28 while others resize the image**

    Those datasets in which the models obtained in the MedMNIST v2 paper with image size 28 outperformed size 224, will not use resize (thus, image size 28). This affects **BreastMNIST**.
    Otherwise, we use resize to 128x128 for 2D datasets (**PneumoniaMNIST**) and to 28x80x80 for 3D datasets (**NoduleMNIST3D**, **AdrenalMNIST3D**, **VesselMNIST3DA**, **SynapseMNIST3D**).
    The exception is **ChestMNIST**, which uses 28x28 due to resource constraints and to speed up training.

3. **Grid search to find select augmentation used.**

    For each augmentation and dataset, we train a model with the hyperparameters obtained in previous steps for that dataset. We then select...

    TODOOOOOOOOOOOOOOOOOO











To test all the models

```
python test.py
```

To test a particular model append the `--test [path]` flag to the `python train.py` command used for training. 
It is important to use the same options as for training to ensure the correct augmentations and other options are used.
