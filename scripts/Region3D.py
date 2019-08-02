from scripts.Tiff_processing import region_properties, label_filter, overlaped_regions
from skimage import io
import numpy as np
import pandas as pd


class RegionFrame:

    def __init__(self, regions_properties):
        region_id = 1
        regions_ids = []
        self.regions = {}
        for region in regions_properties:
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
        print("regions enriched : {}".format(len(couples)))
        return partial_map

    def populate_region3D(self, new_regions):
        partial_map = []
        new_layer_id = self.get_last_layer_id() + 1
        for region in new_regions:
            region_id = self.get_last_id() + 1
            partial_map.append(region_id)
            region3D = Region3D(region, region_id, new_layer_id)
            self.regions[region_id] = region3D
        print("regions created in layer {0} : {1}".format(new_layer_id, len(new_regions)))
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
            df.append(features, ignore_index=True)
        df = df.set_index(["id"])
        return df


class Region3D:
    def __init__(self, region, id, layer_id):
        self.id = id
        self.layers = {layer_id: region}

    def add_layer(self, region):
        layer_index = max(self.layers.keys()) + 1
        self.layers[layer_index] = region

    def get_region(self, layer_index):
        return self.layers[layer_index]

    def get_id(self):
        return self.id

    # Features extraction for classification
    def get_area(self):
        area = 0
        for r in self.layers.values():
            area += r.area
        return area

    def get_coords(self):
        coords_3D = []
        for layer_id, r in self.layers.items():
            layer_coords = [[coord[0], coord[1], layer_id] for coord in r.coords]
            coords_3D.append(layer_coords)
        return coords_3D

    def get_total_intensity(self):
        total_intensity = 0
        for r in self.layers.values():
            total_intensity += np.sum(r.intensity_image)
        return total_intensity

    def get_mean_intensity(self):
        means = []
        for r in self.layers.values():
            means.append(r.mean_intensity)
        mean = np.mean(means)
        return mean

    def get_max_intensity(self):
        maxs = []
        for r in self.layers.values():
            maxs.append(r.max_intensity)
        max = np.max(maxs)
        return max

    def get_min_intensity(self):
        mins = []
        for r in self.layers.values():
            mins.append(r.min_intensity)
        min = np.min(mins)
        return min

    def get_centroid_3D(self):
        centroids = self.get_local_centroids()
        x = int(np.sum([c[0] for c in centroids]) / float(len(centroids)))
        y = int(np.sum([c[1] for c in centroids]) / float(len(centroids)))
        z = np.mean(list(self.layers.keys())).astype(int)
        return (x, y, z)

    def get_local_centroids(self):
        centroids = []
        for r in self.layers.values():
            centroids.append(r.centroid)
        return centroids

    def get_equivalent_sphere(self):
        # TODO : return radius and centroids coordinate

        pass

    def get_extent(self):
        """Ratio of pixels in the region to pixels in the total bounding box. Computed as area / (rows * cols * height)"""
        coords = np.array(self.get_coords())
        rows = np.amax(coords[:, :, 0]) - np.amin(coords[:, :, 0]) + 1
        cols = np.amax(coords[:, :, 1]) - np.amin(coords[:, :, 1]) + 1
        height = np.amax(coords[:, :, 2]) - np.amin(coords[:, :, 2]) + 1
        extent = self.get_area() / np.prod([rows, cols, height])
        return extent

    def extract_features(self):
        return {"id ": self.id,
                "area": self.get_area(),
                "total_intensity": self.get_total_intensity(),
                "mean_intensity": self.get_mean_intensity(),
                "max_intensity": self.get_max_intensity(),
                "min_intensity": self.get_min_intensity(),
                "centroid_3D": self.get_centroid_3D(),
                "extent": self.get_extent()
                }


def extract_regions(tiff_file, channel=0):
    tiff = io.imread(tiff_file)

    if not isinstance(channel, int):
        raise TypeError("Wrong type for channel value")

    try:
        ch = tiff[:, :, :, channel]
    except Exception as e:
        raise e

    init = True
    for layer, img in enumerate(ch):
        print("_" * 80)
        print("Layer {}".format(layer))
        print("_" * 80)
        regions, df_properties = region_properties(label_filter(img)[0], img, min_area=1)
        if init and regions:
            rf = RegionFrame(regions)
            init = False
            prev_img = img
        elif not init:
            region_dict = rf.get_regions_in_last_layer()
            matched_regions, new_regions_matched_ids = overlaped_regions(img, regions, prev_img, region_dict)
            existing_regions_map = rf.enrich_region3D(matched_regions)
            new_regions = [region for idx, region in enumerate(regions) if idx not in new_regions_matched_ids]
            new_regions_map = rf.populate_region3D(new_regions)
            rf.update_map(existing_regions_map, new_regions_map)
            prev_img = img

    print("Total amount of regions detected {}".format(rf.get_amount_of_regions3D()))
    return rf


def test_pipeline():
    file_path_template = "C:\\Users\\Antoine\\PycharmProjects\\Protein_image_processing\\data\\train\\" \
                         "C10DsRedlessxYw_emb{}_Center_Out.tif"
    EMBRYOS = {1: (77, 24221), 7: (82, 23002), 8: (71, 15262), 10: (92, 23074)}

    try:
        results = {}
        for embryo in EMBRYOS.keys():
            file_path = file_path_template.format(embryo)
            results[embryo] = extract_regions(file_path).get_amount_of_regions3D()
        for embryo, r in results.items():
            expected = EMBRYOS[embryo][0] + EMBRYOS[embryo][1]
            print(" {} (expected {} ) proteins detected for embryo {}").format(r, expected, embryo)

    except Exception as e:
        raise e


if __name__ == '__main__':
    file_path = "C:\\Users\\Antoine\\PycharmProjects\\Protein_image_processing\\data\\" \
                "C10DsRedlessxYw_emb11_Center_Out.tif"
    rf = extract_regions(file_path)
    df = rf.extract_features()
    print(df.head())
