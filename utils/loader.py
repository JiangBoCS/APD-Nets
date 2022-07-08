import os

from dataset import DataLoaderTrain, DataLoaderVal, DataLoaderTest, DataLoaderTestSR
#def get_training_data(rgb_dir, img_options):
#    assert os.path.exists(rgb_dir)
#    return DataLoaderTrain(rgb_dir, img_options, None)
    
def get_training_data(gt_dir, input_dir, img_options):
    assert os.path.exists(gt_dir)
    assert os.path.exists(input_dir)
    return DataLoaderTrain(gt_dir, input_dir, img_options, None)

def get_validation_data(gt_dir, input_dir, img_options):
    assert os.path.exists(gt_dir)
    assert os.path.exists(input_dir)
    return DataLoaderVal(gt_dir, input_dir, img_options, None)


def get_test_data(gt_dir, input_dir):
    assert os.path.exists(gt_dir)
    assert os.path.exists(input_dir)
    return DataLoaderTest(gt_dir, input_dir, None)


def get_test_data_SR(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderTestSR(rgb_dir, None)