import logging
from scripts.file_viewer import MultiLayerViewer
from scripts.performance_monitoring import Timer
import scripts.file_manager as fm

import numpy as np
import pandas as pd



class RegionFrame:

    def __init__(self, df_regions):
        region_id = 1
        regions_ids = []
        self.regions = {}
        for index, region in df_regions.iterrows():
            region3D = Region3D(region, region_id, 0)
            regions_ids.append(region_id)
            self.regions[region_id] = region3D
            region_id += 1
        self.map = {0: regions_ids}

    def get_region3D(self, region_id):
        return self.regions[region_id]

    def get_last_id(self):
        try:
            id = max(self.regions.keys())
        except ValueError:
            id = 0
        return id

    def get_last_layer_id(self):
        return max(self.map.keys())

    def get_map(self, index):
        return self.map[index]

    def get_last_map(self):
        return self.map[self.get_last_layer_id()]

    def get_regions3D_in_layer(self, index):
        regions3D = {}
        for i in self.get_map(index):
            region3D = self.get_region3D(i)
            regions3D[i] = region3D
        return regions3D

    def get_regions3D_in_last_layer(self):
        return self.get_regions3D_in_layer(self.get_last_layer_id())

    def get_regions_in_layer(self, layer_id):
        regions3D = self.get_regions3D_in_layer(layer_id)
        regions = {}
        for region_id, region3D in regions3D.items():
            regions[region_id] = region3D.get_region(layer_id)
        return regions

    def get_regions_in_last_layer(self):
        return self.get_regions_in_layer(self.get_last_layer_id())

    def enrich_region3D(self, couples):
        partial_map = []
        for key, item in couples.items():
            region = item
            region3D = self.get_region3D(key)
            region3D.add_layer(region)
            region3D_id = region3D.get_id()
            partial_map.append(region3D_id)
        logging.info("regions enriched : {}".format(len(couples)))
        return partial_map

    def populate_region3D(self, new_regions):
        partial_map = []
        new_layer_id = self.get_last_layer_id() + 1
        for region in new_regions:
            region_id = self.get_last_id() + 1
            partial_map.append(region_id)
            region3D = Region3D(region, region_id, new_layer_id)
            self.regions[region_id] = region3D
        logging.info("regions created in layer {0} : {1}".format(new_layer_id, len(new_regions)))
        return partial_map

    def update_map(self, existing_ids, new_ids):
        map_id = self.get_last_layer_id() + 1
        self.map[map_id] = existing_ids + new_ids

    def get_amount_of_regions3D(self):
        return len(self.regions)

    def extract_features(self):
        df = pd.DataFrame()
        for r in self.regions.values():
            features = r.extract_features()
            df = df.append(features, ignore_index=True)
        return df

# Region 3D are implemented to work only with data dictionnaries containing the following keys :
# 'coords', 'intensity_image', 'max_intensity', 'min_intensity', 'mean_intensity', 'centroid-0', 'centroid-1'
class Region3D:
    def __init__(self, region, id, layer_id):
        self.id = id
        data = {key: [value] for key, value in region.items()}
        self.layers = pd.DataFrame(data, index=[layer_id])

    def add_layer(self, region):
        layer_index = max(self.layers.index.values) + 1
        region = region.rename(layer_index)
        self.layers = self.layers.append(region)

    def get_region(self, layer_index):
        return self.layers.loc[layer_index]

    def get_id(self):
        return self.id

    # Features extraction for classification
    def get_depth(self):
        return len(self.layers.index)

    def get_area(self):
        area = 0
        for i, r in self.layers.iterrows():
            area += r['area']
        return area

    def get_coords(self):
        coords_3D = []
        for layer_id, r in self.layers.iterrows():
            layer_coords = [[coord[0], coord[1], layer_id] for coord in r['coords']]
            coords_3D += layer_coords
        return coords_3D

    def get_layers(self):
        return list(self.layers.keys())

    def get_total_intensity(self):
        total_intensity = 0
        for layer_id, r in self.layers.iterrows():
            total_intensity += np.sum(r['intensity_image'])
        return total_intensity

    def get_mean_intensity(self):
        #TODO : change with mean_intensity = total_intensity / total_area
        means = []
        for layer_id, r in self.layers.iterrows():
            means.append(r['mean_intensity'])
        mean = np.mean(means)
        return mean

    def get_max_intensity(self):
        maxs = []
        for layer_id, r in self.layers.iterrows():
            maxs.append(r['max_intensity'])
        max = np.max(maxs)
        return max

    def get_min_intensity(self):
        mins = []
        for layer_id, r in self.layers.iterrows():
            mins.append(r['min_intensity'])
        min = np.min(mins)
        return min

    def get_centroid_3D(self):
        centroids = self.get_local_centroids()
        x = int(np.sum([c[0] for c in centroids]) / float(len(centroids)))
        y = int(np.sum([c[1] for c in centroids]) / float(len(centroids)))
        z = np.mean(self.layers.index.values).astype(int)
        return x, y, z

    def get_local_centroids(self):
        centroids = []
        for layer_id, r in self.layers.iterrows():
            centroids.append((r['centroid-0'], r['centroid-1']))
        return centroids

    def get_extent(self):
        """Ratio of pixels in the region to pixels in the total bounding box. Computed as area / (rows * cols * height)"""
        coords = np.array(self.get_coords())
        rows = np.amax(coords[:, 0]) - np.amin(coords[:, 0]) + 1
        cols = np.amax(coords[:, 1]) - np.amin(coords[:, 1]) + 1
        height = np.amax(coords[:, 2]) - np.amin(coords[:, 2]) + 1
        extent = self.get_area() / np.prod([rows, cols, height])
        return extent

    def extract_features(self):
        ids = self.id
        depth = self.get_depth()
        areas = self.get_area()
        total_intensity = self.get_total_intensity()
        mean_intensity = self.get_mean_intensity()
        max_intensity = self.get_max_intensity()
        min_intensity = self.get_min_intensity()
        x, y, z = self.get_centroid_3D()
        extent = self.get_extent()

        features = {"id": ids,
                    "depth": depth,
                    "area": areas,
                    "total_intensity": total_intensity,
                    "mean_intensity": mean_intensity,
                    "max_intensity": max_intensity,
                    "min_intensity": min_intensity,
                    "centroid_x": x,
                    "centroid_y": y,
                    "centroid_z": z,
                    "extent": extent
                    }
        return features
    # Todo : add the solidity of the image as feature of Region3D



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    timer = Timer()
    folder_path = "C:\\Users\\Antoine\\PycharmProjects\\Protein_image_processing\\data\\"
    file = "C10DsRedlessxYw_emb11_Center_Out.tif"
    file_path = folder_path + file
    tiff = fm.get_tiff_file(11)

    viewer = MultiLayerViewer(tiff, channel=2)
    #blobs, rscl_img = blob_extraction(viewer.get_images())
    #print(blobs)
    viewer.plot_imgs()

    viewer.show()






