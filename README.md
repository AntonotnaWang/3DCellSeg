# 3DCellSeg

A light and robust tool to do 3D cell instance segmentation.

Official implementation of [A novel deep learning-based 3D cell segmentation framework for future image-based disease detection](https://www.nature.com/articles/s41598-021-04048-3). (However, I think the title should be ***3DCellSeg - a robust deep learning-based 3D cell instance segmentation pipeline***.)

<div align="center">
    <img src="figs/central_illustration.png" width="700"/>
</div>

## Datasets and data pre-processing

3DCellSeg was trained and tested on four datasets: HMS (private), ATAS, [LRP](https://osf.io/2rszy/), [Ovules](https://osf.io/w38uf/).

*For HMS dataset, please send email to the authors to get data access.*

- Put the data filefolder under a given path (e.g., ```\data\CellSeg_dataset``` in my case)

- The data structure should be like:

For HMS

```
\data\CellSeg_dataset\HMS
    \raw
        \100.mha
        ...
    \segmentation
        \100.mha
        ...
```

For ATAS

```
\data\CellSeg_dataset\ATAS
    \plant1
        \processed_tiffs
            \0hrs_plant1_trim-acylYFP.tif
            ...
        \segmentation_tiffs
            \0hrs_plant1_trim-acylYFP_hmin_2_asf_1_s_2.00_clean_3.tif
            ...
        ...
    ...
```

For LRP

```
\data\CellSeg_dataset\LateralRootPrimordia
    \train
        \Movie1_t00003_crop_gt.h5
        ...
    \test
        \Movie1_t00006_crop_gt.h5
        ...
    ...
```

For Ovules

```
\data\CellSeg_dataset\Ovules
    \train
        \N_404_ds2x.h5
        ...
    \test
        \N_294_final_crop_ds2.h5
        ...
    ...
```

- Run ```prepare_datasets.py``` (which calls py files in folder ```prepare_dataset```) to pre-process the dataset. ([Here](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/wangad_connect_hku_hk/Eo2UGa6WTA5Gj0F1S6Xc9HABedf0FUVpNw8swUWl5-3kRg?e=UMBGmx) to download the samples of pre-processed files (passwd: ```cellseg```).)

- You may use ```prepare_datasets.py``` to pre-crop the 3D images if you find the training is slow.

- **\[IMPORTANT\] ```prepare_datasets.py``` generates a python dict (stored in ```dataset_info```) for each processed dataset. The dict contains file paths to each train and test files. All dataloading operations during training and testing depend on the dict. You should create your own dict. See dict samples in ```dataset_info```.**

## Train and test

Run ```train_HMS.py```, ```train_ATAS.py```, ```train_LRP.py```, and ```train_Ovules.py``` to train the model on the corresponding dataset. They were implemented by PyTorch. You can easily adjust the hyperparameters.

Run notebook ```test_HMS.ipynb```, ```test_ATAS.ipynb```, ```test_LRP.ipynb```, and ```test_Ovules.ipynb``` to test the model.

## Pretrained models

See folder ```output```.

## If you have more questions about the code

Please contact ```wangad@connect.hku.hk```.
