import logging
from scripts.file_viewer import MultiLayerViewer
from scripts.performance_monitoring import Timer
import scripts.file_manager as fm

import numpy as np
import pandas as pd


class RegionFrame:
    """ A RegionFrame included every Region3D extracted from an image.

    Parameters
    --------
    regions: dict
        Dictionnary containing all Region3D extracted in a tiff file. Keys refers to the id of the Region3D and Values refer to the Region3D object
    map: dict
        Dictionnary containing all the Region3D ids sorted by layer index. Keys refers to the layer index and Values refer to a list of Region3D ids. This attribute facilitate the recovery of the regions included in a given layer of the file.
    cells: ndarray
        Labelled image identifying the cells of the embryos
    """

    def __init__(self, df_regions, cells = None):
        region_id = 1
        regions_ids = []
        self.regions = {}
        if cells is not None:
            self.cells = cells
        for index, region in df_regions.iterrows():
            region3D = Region3D(region, region_id, 0)
            regions_ids.append(region_id)
            self.regions[region_id] = region3D
            region_id += 1
        self.map = {0: regions_ids}

    def get_region3D(self, region_id):
        return self.regions[region_id]

    def get_last_id(self):
        """ Returns the last region id recorded in the RegionFrame.

        :return: int
            Region3D id.
        """
        try:
            id = max(self.regions.keys())
        except ValueError:
            id = 0
        return id

    def get_last_layer_id(self):
        """ Returns the id of the last layer recorded in the RegionFrame

        :return: int
            Last layer id
        """
        return max(self.map.keys())

    def get_map(self, index):
        """ Return the map of a specfic layer.

        :param index: int
            Layer index
        :return: array
            List of Region3D ids included in the given layer.
        """
        return self.map[index]

    def get_last_map(self):
        """ Return the map (list of Region3D ids) of the last recorded layer.

        :return: array
            List of Region3D ids included in the last layer.
        """
        return self.map[self.get_last_layer_id()]

    def get_regions3D_in_layer(self, index):
        """ Return a dictionary of all regions 3D included in a layer.

        :param index: int
            Layer id
        :return: dict
            With Keys as region id and Values as the related Region3D object.
        """
        regions3D = {}
        for i in self.get_map(index):
            region3D = self.get_region3D(i)
            regions3D[i] = region3D
        return regions3D

    def get_regions3D_in_last_layer(self):
        """ Return a dictionary of all regions 3D included in the last layer.

            :return: dict
                With Keys as region id and Values as the related Region3D object.
        """
        return self.get_regions3D_in_layer(self.get_last_layer_id())

    def get_regions_in_layer(self, layer_id):
        """ Return a dictionary including with all the regions of a given layer. Each region is mapped with it related Region3D id.
        This function might be usefull for overlaping detection where only the last region (and not the full Region3D) is needed.

        :param layer_id: int
            Layer id
        :return: dict
            With Keys as Region3D id and Values as its related region.
        """
        regions3D = self.get_regions3D_in_layer(layer_id)
        regions = {}
        for region_id, region3D in regions3D.items():
            regions[region_id] = region3D.get_region(layer_id)
        return regions

    def get_regions_in_last_layer(self):
        """ Return a dictionary including with all the regions of the last layer. Each region is mapped with it related Region3D id.
        This function might be usefull for overlaping detection where only the last region (and not the full Region3D) is needed.

        :return: dict
            With Keys as Region3D id and Values as its related region.
        """
        return self.get_regions_in_layer(self.get_last_layer_id())

    def are_in_cell(self, df_region):
        pass

    def enrich_region3D(self, couples):
        """ Add new regions to the related Region3D.

        :param couples: dict
            Dictionary describing a map of {key: Region3D id, value: region}
        :return: array
            List of the Region3D ids which have been enriched with a new region.
        """
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
        """ Create new Region3D from unmapped regions.

        :param new_regions: array
            List of regions which haven't been mapped to existing Region3D.
        :return: array
            List of Region3D ids newly created.
        """
        partial_map = []
        new_layer_id = self.get_last_layer_id() + 1
        for id_region, region in new_regions.iterrows():
            region3D_id = self.get_last_id() + 1
            partial_map.append(region3D_id)
            region3D = Region3D(region, region3D_id, new_layer_id)
            self.regions[region3D_id] = region3D
        logging.info("regions created in layer {0} : {1}".format(new_layer_id, len(new_regions)))
        return partial_map

    def update_map(self, existing_ids, new_ids):
        """ Add a new layer to the maps by merging existing ids (enriched) and new ids (populated).

        :param existing_ids: array
            List of the Region3D ids which have been enriched with a new region.
        :param new_ids:
            List of Region3D ids newly created.
        """
        map_id = self.get_last_layer_id() + 1
        self.map[map_id] = existing_ids + new_ids

    def get_amount_of_regions3D(self):
        """ Return the total amount of Region3D in a RegionFrame
        """
        return len(self.regions)

    def extract_features(self):
        """ Extract features from all Region3D included in a RegionFrame.

        :return: pandas.DataFrame
            DataFrame with region as row and features as columns.

        Note: For further information on features types, please refer to Region3D.extract_features
        """
        df = pd.DataFrame()
        total = len(self.regions.values())
        i = 0
        for r in self.regions.values():
            prog = (i/total*100).round()
            if prog % 10==0:
                logging.info("progression {}%, {}/{} regions".format(prog, i, total))
            features = r.extract_features()
            df = df.append(features, ignore_index=True)
            i+=1
        df.set_index('id')
        return df


# Region 3D are implemented to work only with data dictionnaries containing the following keys :
# 'coords', 'intensity_image', 'max_intensity', 'min_intensity', 'mean_intensity', 'centroid-0', 'centroid-1'
class Region3D:
    """ Collection of regions mapped over the z axis.

    Parameters
    --------
    id: int
        Id of the Region3D
    layers: pandas.DataFrame
        DataFrame where rows are the layer id and columns are region feature
    cell: int
        Label id of the cell in which the region is included. 0 if the region is out of cells boundaries
    """

    def __init__(self, region, id, layer_id, cell=0):
        self.id = id
        data = {key: [value] for key, value in region.items()}
        self.layers = pd.DataFrame(data, index=[layer_id])

    def add_layer(self, region):
        """ Add a layer to the Region3D.

        :param region: pandas.Series
            Serie including all features of the region to add.
        """
        layer_index = max(self.layers.index.values) + 1
        region = region.rename(layer_index)
        self.layers = self.layers.append(region)

    def get_region(self, layer_index):
        """Return region from a given layer index

        :param layer_index: int
            Layer index
        :return: pandas.Series
            Series including the features of the related region
        """
        return self.layers.loc[layer_index]

    def get_id(self):
        """ Return the id of the Region3D

        :return: int
            Region3D id
        """
        return self.id

    def get_cell(self):
        cells = []
        for i, r in self.layers.iterrows():
            cells.append(r['cell'])
        unique = np.unique(np.array(cells))
        cell = unique[0]
        if len(unique) > 1:
            logging.debug("More than one cell contains the region {}".format(self.id))
        return cell


    # Features extraction for classification
    def get_depth(self):
        """ Return the amount of layer included in the Region3D.

        :return: int
            Depth of the Region3D
        """
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
        mean = self.get_total_intensity() / self.get_area()
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
        y = int(np.sum([c[0] for c in centroids]) / float(len(centroids)))
        x = int(np.sum([c[1] for c in centroids]) / float(len(centroids)))
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

    def get_convex_area(self):
        """Sum of the area surrounded by the smallest hull polygon of each layer"""
        c_area = 0
        for i, r in self.layers.iterrows():
            c_area += r['convex_area']
        return c_area

    def get_solidity(self):
        """Ratio of pixels in the region to pixels in the convex hull polygon. Computed as area / convex_area"""

        return self.get_area() / self.get_convex_area()

    def extract_features(self):
        """ Return the computed features of the Region3D. These features includes:
            - "id": id of the Region3D,
            - "depth": depth of the Region3D,
            - "area": Total area covered by all regions of the Region3D,
            - "total_intensity": Sum of the intensities from all regions of the Region3D,
            - "mean_intensity": Ratio of total intensity over total area,
            - "max_intensity": maximum intensity over each region,
            - "min_intensity": minimum intensity over each region,
            - "centroid_x": x coordinate of the Region3D centroid,
            - "centroid_y": y coordinate of the Region3D centroid,
            - "centroid_z": z coordinate of the Region3D centroid,
            - "extent": Ratio of pixels in the region to pixels in the total bounding box.
            - "solidity": Ratio of the region area with the area of the smallest hull polygon that that surround the region
            - "in_cell": True is the region is included in a cell
        :return: dict
            Dictionnary including Region3D's features.
        """
        # Todo : add the solidity of the image as feature of Region3D

        ids = self.id
        depth = self.get_depth()
        areas = self.get_area()
        total_intensity = self.get_total_intensity()
        mean_intensity = self.get_mean_intensity()
        max_intensity = self.get_max_intensity()
        min_intensity = self.get_min_intensity()
        x, y, z = self.get_centroid_3D()
        extent = self.get_extent()
        solidity = self.get_solidity()
        in_cell = self.get_cell()

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
                    "extent": extent,
                    "solidity": solidity,
                    "cell": in_cell
                    }
        return features
