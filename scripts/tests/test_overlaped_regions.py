from unittest import TestCase

from scripts.tiff_processing import region_properties, label_filter, overlaped_regions
import numpy as np

class TestOverlaped_regions(TestCase):
    def setUp(self):
        im1 = np.zeros([20, 20]).astype(np.int_)
        # create one region
        im1[2:5, 2:5] = 500
        # Crete another region
        im1[10:18, 10:18] = 1000
        df1 = region_properties(label_filter(im1)[0])

        prev_im = np.zeros([20, 20]).astype(np.int_)
        #create one region partially overlaping
        prev_im[8:12, 8:12] = 700
        df_prev = region_properties(label_filter(prev_im)[0])
        self.existing_regions_map, self.new_regions_matched_ids = overlaped_regions(im1, df1, prev_im, df_prev)

    def test_overlaped_regions(self):
        self.assertTrue(len(self.existing_regions_map) == 1)
        self.assertTrue(len(self.new_regions_matched_ids) == 1)


