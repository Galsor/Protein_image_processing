
from scripts.Tiff_processing import region_properties, label_filter
from skimage import io



class RegionFrame:

    def __init__(self, regions_properties):
        region_id = 1
        regions_ids =[]
        self.regions = {}
        for region in regions_properties:
            Region3D(region, region_id)
            regions_ids.append(region_id)
            self.regions[region_id] = region
            region_id += 1
        self.layers = { 0: regions_ids}

    def get_region3D(self, region_id ):
        return self.regions[region_id]

    def get_layer(self, index):
        return self.layers[index]

    def get_last_layer(self):
        return self.layers[max(self.layers.keys())]

    def enrich_region3D(self, couples):
        #TODO to be implemented
        pass

    def populate_region3D(self, new_regions):
        #Todo to be implemented
        pass


class Region3D:
    def __init__(self, region, id):
        self.id = id
        self.layers = {1: region}

    def add_layer(self, region):
        layer_index = max(self.layers.keys())+1
        self.layers[layer_index] = region

    def get_region(self, layer_index):
        return self.layers[layer_index]



def extract_regions(tiff_file, channel):
    tiff = io.imread(tiff_file)

    if not isinstance(channel, int):
        raise TypeError("Wrong type for channel value")

    try :
        ch1 = tiff[:, :, :, channel]
    except Exception as e:
        raise e

    init = True
    for img in tiff :
        regions, df_properties = region_properties(label_filter(img)[0], img, min_area= 4 )
        if init:
            rf = RegionFrame(regions)
            init=False
        else:
            #TODO:
            # - compare images
            # - add layer with the existing ones
            # - populate with the other

    return rf


if __name__ == '__main__' :
    file_path = "C:\\Users\\Antoine\\PycharmProjects\\Protein_image_processing\\data\\" \
                "C10DsRedlessxYw_emb11_Center_Out.tif"
    try :
        extract_regions(file_path)
    except Exception as e:
        raise e