from scripts.Tiff_processing import region_properties, label_filter, overlaped_regions
from skimage import io


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
    return rf.get_amount_of_regions3D()


if __name__ == '__main__':
    file_path = "C:\\Users\\Antoine\\PycharmProjects\\Protein_image_processing\\data\\" \
                "C10DsRedlessxYw_emb11_Center_Out.tif"
    file_path_template = "C:\\Users\\Antoine\\PycharmProjects\\Protein_image_processing\\data\\train\\" \
                         "C10DsRedlessxYw_emb{}_Center_Out.tif"
    EMBRYOS = {1: (77, 24221), 7: (82, 23002), 8: (71, 15262), 10: (92, 23074)}

    try:
        results = {}
        for embryo in EMBRYOS.keys():
            file_path = file_path_template.format(embryo)
            results[embryo] = extract_regions(file_path)
        for embryo, r in results.items():
            expected = EMBRYOS[embryo][0] + EMBRYOS[embryo][1]
            print(" {} (expected {} ) proteins detected for embryo {}").format(r, expected, embryo)

    except Exception as e:
        raise e
