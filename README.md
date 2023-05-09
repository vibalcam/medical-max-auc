# AUC-ROC Maximization for Medical data

The best models are accesible at 

https://tamucs-my.sharepoint.com/:f:/g/personal/vibalcam_tamu_edu/Ekjod78yCmBAoN_ZQ0xSGxIBstQgvc8xgWBb7suabOhdaw?e=VL2yax

## Test models

To test all the models:
```
python test.py
```

To test a particular model append the `--test [path]` flag to the `python train.py` command used for training. 
It is important to use the same options as for training to ensure the correct augmentations and other options are used.

## Model selection

1. **Grid search to tune hyperparameters using images with size 28.**

    For each combination of hyperparameters, we evaluate the validation AUC every 5 epochs and select the epoch with the highest validation score.
    For each dataset, we select the combination of hyperparameters that achieves the highest validation AUC.

    (The flag `--use_best_model` saves as `last.ckpt` the epoch with the highest validation AUC)

    For **ChestMNIST**, we used early stopping (patience 50 epochs) and mixed-16 precision for faster training. The following grid search is used:
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

    For the **3D datasets**, we used early stopping (patience 50 epochs) and the following grid search:
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

    For each augmentation and dataset, we train a model with the best hyperparameters obtained in previous steps for that dataset. We train for 200 epochs and choose the augmentation that provides the highest validation AUC.

    The script to run the search is the following:
    ```
    ./scripts/experiments/run_aug.sh
    ```

    For the **2D datasets**, the following augmentations have been tested:
    - RandomAffine ([-5,5] degrees)
    - Gaussian Blur ([0.1,2]) 
    - Random Resized Crop (scale [0.7, 1])

    For the **3D datasets**, we used early stopping (patience 50 epochs) and mixed-16 precision. Additionally, for SynapseMNIST3D, we select the epoch with the highest validation AUC (validation AUC is not stable). The following augmentations have been tested:
    - RandomAffine ([-5,5] degrees)
    - Elastic Transform (alpha 0.1, sigma 0.1) 
    - Random Resized Crop (scale [0.7, 1]) 
    - Regularization from MedMNIST v2 paper, i.e., multiply the training set by a random value in [0,1] during training and multiply the images by a fixed coefficient of 0.5 during evaluation.

    For **ChestMNIST**, we used early stopping (patience 50 epochs) and mixed-16 precision for faster training. Additionally, for each augmentation, we select the epoch with the highest validation AUC. 
    The following augmentations have been tested:
    - RandomAffine ([-5,5] degrees)
    - Color Jitter (brightness 0.1, contrast 0.1) 
    - Gaussian Blur ([0.1,2]) 
    - Elastic Transform (alpha 0.1, sigma 0.1) 
    - Random Resized Crop (scale [0.7, 1])

    This step results in the model presented for **ChestMNIST** and **all 3D datasets** except **AdrenalMNIST3D**, for which we did not run the grid search due to time constraints. Instead, see section 5.

4. For **BreastMNIST** and **PneumoniaMNIST**, we use the best hyperparameters and augmentation from previous steps and train the model on the combination of the train and validation datasets. 
    We save the last epoch and that results on the model presented.

5. For **AdrenalMNIST3D** we first run pretraining and then optimize AUC on the pretrained model. We select the epoch with highest validation AUC. More details on the report.
   This results on the model presented for **AdrenalMNIST3D**.

   ```
   ./scritpts/specific/run_adrenal.sh
   ```

6. To train the best models for all datasets:
    ```
    ./scripts/run_all.sh
    ```

7. To evaluate the best models obtained:
    ```
    python test.py
    ```


- [X] ~~*readme*~~ [2023-05-08]
- [X] ~~*selection scheme*~~ [2023-05-08]
- [X] ~~*check selections for adrenal*~~ [2023-05-08]
- [ ] run all
- [ ] add additional
- [ ] test run aug
- [ ] test all
- [X] ~~*add to readme*~~ [2023-05-08]
- [X] ~~*add to paper*~~ [2023-05-08]

- [X] ~~*bold best results*~~ [2023-05-08]

- [X] ~~*pretraining*~~ [2023-05-08]
- [X] ~~*incentive*~~ [2023-05-08]
- [ ] incenvite results
- [ ] results
- [ ] additional
- [ ] conclusions

- [ ] check nodule aug

