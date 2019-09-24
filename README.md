# Molecule classification

This project aims to detect automatically molecule and transcriptions site involved in the first step of developpement of drosophilia embryo.
Such information can be extracted from microscop images such as this one:

![microscop image](docs/image_embryo.JPG)


This package implements several tools requested to achieve this classification:
- Image filtering and scaling
- Region extraction
- Blob detection
- Contours detection
- 3D region aggregation
- Region features extraction
- Unsupervised classification benchmarker
- File manager and viewer
- Performance monitoring

A standard pipeline has been implemented in demo.py 
![pipeline image](docs/pipeline.JPG)


### Exemples
#### Region detection
Region detection performed on cells.
![region image](docs/region_detection.png)

#### Cell contour detection
This detection is used as feature of classification algorithm. Region are tagged regarding the cell they are embedded in or not. 
![Cell contours](docs/cells_contours_detection.png)

#### Various data exploration prior to classification
Region intensity exploration along the z axis
![z_distribution](docs/z_distribution.JPG)

Intensity / area ratio
![ratio_mean_area](docs/data_visualisation.JPG)

Distribution filtering. This part enables classification of Transcription site
![distribution_filtering](docs/quantile_filtering.JPG)

Features distribution analysis
![features_distribution](docs/Distribution_comparison.JPG)

#### Unsupervised classification benchmark
A benchmark tool has been implemented to identify which classification algorithm presents the best results.
The following classification was handled on the raw features (prior to distribution filtering)
![benchmark](docs/multiclustering.png)
 

  