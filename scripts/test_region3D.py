from unittest import TestCase, main
from scripts.Region3D import Region3D
import numpy as np
from skimage.measure import regionprops

class TestRegion3D(TestCase):
    def setUp(self):
        img = np.zeros([20, 20]).astype(np.int_)
        img[5:15, 5:15] = 5
        bin = img.copy()
        bin[5:15, 5:15] = 1
        region = regionprops(bin, img)[0]
        self.r = Region3D(region, 0, 0)
        self.result_features = {'id': 0,
                       'area': 100,
                       'total_intensity': 500,
                       'mean_intensity': 5.0,
                       'max_intensity': 5,
                       'min_intensity': 5,
                       'centroid_3D': (9, 9, 0),
                       'extent': 1.0}


    def test_add_layer(self):
        #TODO
        self.fail()

    def test_get_region(self):
        #TODO
        self.fail()

    def test_get_id(self):
        self.assertTrue(self.r.get_id() == self.result_features['id'])

    def test_get_area(self):
        self.assertTrue(self.r.get_area() == self.result_features['area'])

    def test_get_coords(self):
        #TODO
        self.fail()

    def test_get_total_intensity(self):
        total_intensity = self.r.get_total_intensity()
        self.assertTrue(total_intensity == self.result_features['total_intensity'])

    def test_get_mean_intensity(self):
        self.assertTrue(self.r.get_mean_intensity() == self.result_features['mean_intensity'])

    def test_get_max_intensity(self):
        self.assertTrue(self.r.get_max_intensity() == self.result_features['max_intensity'])

    def test_get_min_intensity(self):
        self.assertTrue(self.r.get_min_intensity() == self.result_features['min_intensity'])

    def test_get_centroid_3D(self):
        self.assertTrue(self.r.get_centroid_3D() == self.result_features['centroid_3D'])

    def test_get_local_centroids(self):
        #TODO
        self.fail()

    def test_get_equivalent_sphere(self):
        #TODO
        self.fail()

    def test_get_extent(self):
        self.assertTrue(self.r.get_extent() == self.result_features['extent'])

    def test_extract_features(self):
        self.assertEqual(self.r.extract_features() == self.result_features)

if __name__ == '__main__':
    main()