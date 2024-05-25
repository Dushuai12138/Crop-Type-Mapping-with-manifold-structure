# 👋 Crop-Type-Mapping-with-manifold-structure
This is the source code of our study ['Manifold Structure of Multispectral-spatial-temporal Remote Sensing Data in Crop Type Mapping based Temporal Feature Extractor'](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4762397)

The LSTM module helps mine smoother and lower dimensional manifold structure in EVI data. Temporal feature-based segmentation considers temporal feature separately before multispectral-spatial feature.

## This is the manifold in Hetao irrigation district.(Figs. g and h)
![HT_TSNE](https://github.com/Dushuai12138/Crop-Type-Mapping-with-manifold-structure/assets/116633147/eb643eaf-3b83-4749-8b16-934be4d68edc)


## This is the manifold in Northeast China.(Figs. g and h)
![NE_12bands_TSNE](https://github.com/Dushuai12138/Crop-Type-Mapping-with-manifold-structure/assets/116633147/93c9066a-4b34-4c9b-b6fc-e4dd7d3861a8)

## How to start
### 1. make datasets
```
    |_HT
        |_12bands
            |_    0_8510.tif
            ...
        |_EVI
            |_    0_8510.tif
            ...
        |_SegmentationClass
            |_    0_8510.tif
            ...
        |_ImageSets
            |_Segmentation
                |_   test.txt
                |_   train.txt
                |_   trainval.txt
                |_   val.txt
```
### 2.change some parameters in train_process.1_train.py, including
```
    VOCdevkit_path = os.path.join(r'J:\research\GEE\hetao_classification', place)  # base path of datasets
    models = ['TFBS']         # choose models, TFBS
    band = 'EVI'              # choose datasets, including 'EVI' and '12bands'   
    places = ['2020HT']       # choose aoi, including '2020HT' and 'NEofCHINA'
    training = True           # start training
    transferlearning = False  # pre-training from other datasets
    get_miou = True           # caculate accuracy metrics
    prediction = False        # generate prediction map in whole aoi
    lstm_outputses = [32]     # outputs of LSTM module in TFBS, here we modified it to 32. In different datasets, like EVI dataset, it could be smaller.
    input_features = 1        # inputs of LSTM module, or number of band per month. 1 or 12
```
### 3. run train_process.1_train.py

## TFBS model structure
reference

[Yang, L., Huang, R., Huang, J., Lin, T., Wang, L., Mijiti, R., Wei, P., Tang, C., Shao, J., Li, Q., & Du, X. (2021). Semantic Segmentation Based on Temporal Features: Learning of Temporal–Spatial Information From Time-Series SAR Images for Paddy Rice Mapping. IEEE Transactions on Geoscience and Remote Sensing, 60, 1–16. Q1. https://doi.org/10.1109/TGRS.2021.3099522](https://ieeexplore.ieee.org/document/9506988/)

[TFBS source code in keras](https://github.com/younglimpo/TFBSmodel)

[our modified TFBS source code in pytorch](https://github.com/Dushuai12138/Crop-Type-Mapping-with-manifold-structure/blob/main/nets/TFBS.py)
