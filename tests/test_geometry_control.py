import unittest

import pandas as pd

from image_geometry_control import fit_control


class GeometryControlTests(unittest.TestCase):
    def test_nontraining_features_and_labels_do_not_affect_fit(self):
        frame=pd.DataFrame(dict(split=['train']*4+['test']*2,
            label=[0,0,1,1,0,1],width=[100,110,200,210,1000,2000],height=[100,110,200,210,1000,2000]))
        _,first=fit_control(frame)
        changed=frame.copy();changed.loc[changed.split=='test',['width','height']]=1000000
        changed.loc[changed.split=='test','label']=[1,0]
        _,second=fit_control(changed)
        self.assertEqual(first,second)
        self.assertEqual(first['threshold'],.5)

    def test_invalid_training_dimensions_are_rejected(self):
        frame=pd.DataFrame(dict(split=['train','train'],label=[0,1],width=[0,200],height=[100,200]))
        with self.assertRaisesRegex(ValueError,'Invalid image dimensions'):
            fit_control(frame)


if __name__=='__main__':
    unittest.main()
