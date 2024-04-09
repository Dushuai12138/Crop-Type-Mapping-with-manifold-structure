# Crop-Type-Mapping-with-manifold-structure
This is the source code of our study ['Manifold Structure of Multispectral-spatial-temporal Remote Sensing Data in Crop Type Mapping based Temporal Feature Extractor'](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4762397)

The LSTM module helps mine smoother and lower dimensional manifold structure in EVI data. Temporal feature-based segmentation considers temporal feature separately before multispectral-spatial feature.

### This is the manifold in Hetao irrigation district.
![HT_TSNE](https://github.com/Dushuai12138/Crop-Type-Mapping-with-manifold-structure/assets/116633147/4527c678-3e0a-47cd-ba1c-2ecaf764f8db)

### This is the manifold in Northeast China.
![NE_12bands_TSNE](https://github.com/Dushuai12138/Crop-Type-Mapping-with-manifold-structure/assets/116633147/2c37981b-f0d5-412b-9605-06c18b4762a7)


### TFBS model structure
reference

[Yang, L., Huang, R., Huang, J., Lin, T., Wang, L., Mijiti, R., Wei, P., Tang, C., Shao, J., Li, Q., & Du, X. (2021). Semantic Segmentation Based on Temporal Features: Learning of Temporal–Spatial Information From Time-Series SAR Images for Paddy Rice Mapping. IEEE Transactions on Geoscience and Remote Sensing, 60, 1–16. Q1. https://doi.org/10.1109/TGRS.2021.3099522](https://ieeexplore.ieee.org/document/9506988/)

[TFBS source code in keras](https://github.com/younglimpo/TFBSmodel)

[modified TFBS source code in pytorch](https://github.com/Dushuai12138/Crop-Type-Mapping-with-manifold-structure/blob/main/nets/segformer.py)
