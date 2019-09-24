# Molecule classification

This project aims to detect automatically molecule and transcriptions site involved in the first step of developpement of drosophilia embryo.
Such information can be extracted from microscop images such as this one:

![microscop image](https://github.com/Galsor/Protein_image_processing/blob/master/docs/image_embryo.JPG?raw=true)


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
![region image](docs/region_detection.png)

#### Cell contour detection
![Cell contours](docs/cells_contours_detection.png)



  