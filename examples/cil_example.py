import numpy as np
import cil.framework
import tensorboardX
from datetime import datetime

import img_quality_cil_stir as imgq

def MSE(x,y):
    """ mean squared error between two numpy arrays
    """
    return ((x-y)**2).mean()

def MAE(x,y):
    """ mean absolute error between two numpy arrays
    """
    return np.abs(x-y).mean()

def PSNR(x, y, scale = None):
    """ peak signal to noise ratio between two numpy arrays x and y
        y is considered to be the reference array and the default scale
        needed for the PSNR is assumed to be the max of this array
    """
  
    mse = ((x-y)**2).mean()
  
    if scale == None:
        scale = y.max()
  
    return 10*np.log10((scale**2) / mse)

#---------------------------------------------------------------------------------------------

# seed the random generator to setup random test image
np.random.seed(1)
# image dimension / shape of test images
image_shape = (30,40,50)
# setup standard image geometry
image_geom = cil.framework.ImageGeometry(voxel_num_x  = image_shape[2], 
                                         voxel_num_y  = image_shape[1],
                                         voxel_num_z  = image_shape[0],
                                         voxel_size_x = 2.78,
                                         voxel_size_y = 2.98,
                                         voxel_size_z = 3.12)
# setup a test and reference image
test_image = cil.framework.ImageData(array = np.random.rand(*image_shape), geometry = image_geom)
ref_image  = cil.framework.ImageData(array = np.random.rand(*image_shape), geometry = image_geom) 
# setup a 2 binary ROI images
roi_image_dict = {}
mask_1 = np.zeros(image_shape)
mask_1[1:,:-1,0] = 1
roi_image_dict['roi_1'] = cil.framework.ImageData(array = mask_1, geometry = image_geom)
mask_2 = np.zeros(image_shape)
mask_2[0,:-1,1:] = 1
roi_image_dict['roi_2'] = cil.framework.ImageData(array = mask_2, geometry = image_geom)

# create a tensorboardX summary writer
dt_string = datetime.now().strftime("%Y%m%d-%H%M%S")
tb_summary_writer = tensorboardX.SummaryWriter(f'runs/exp-{dt_string}')
# instanciate ImageQualityCallback
img_qual_callback = imgq.ImageQualityCallback(ref_image, tb_summary_writer,
                                              roi_mask_dict = roi_image_dict,
                                              metrics_dict = {'MSE':MSE, 'MAE':MAE, 'PSNR':PSNR},
                                              statistics_dict = {'MEAN': (lambda x: x.mean()),
                                                                 'STDDEV': (lambda x: x.std()),
                                                                 'MAX': (lambda x: x.max()),
                                                                 'COM': (lambda x: np.array([3,2,1]))},
                                              post_smoothing_fwhms_mm_list = [5., 8.])
img_qual_callback.eval(1, 1, test_image)
img_qual_callback.eval(2, 1, test_image*1.1)
img_qual_callback.eval(3, 1, test_image*0.9)
img_qual_callback.eval(4, 1, test_image*1.2)
tb_summary_writer.close()
