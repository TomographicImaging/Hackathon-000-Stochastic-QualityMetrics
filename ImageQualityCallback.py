#TODO: - multi resolution approach ("smoothing before calcuating metrics)
#      - getter and setter methods
#      - how to save / log results (e.g. csv file, tensorboard file ...)

import numpy as np
import cil.framework

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


class ImageQualityCallback:
    r"""

    Parameters
    ----------

    reference_image: ImageData
      containing the reference image used to calculate the metrics

    roi_mask_dict : dictionary of ImageData objects
      list containing one binary ImageData object for every ROI to be
      evaluated. Voxels with values 1 are considered part of the ROI
      and voxels with value 0 are not.
      Dimension of the ROI mask images must be the same as the dimension of
      the reference image.
      
    metrics_dict : dictionary of lambda functions f(x,y) mapping
      two 1-dimensional numpy arrays x and y to a scalar value.
      x and y can be the voxel values of the whole images or the values of
      voxels in a ROI such that the metric can be computed on the whole
      images and optionally in the ROIs separately.

      E.g. f(x,y) could be MSE(x,y), PSNR(x,y), MAE(x,y)

    statistics_dict : dictionary of lambda functions f(x) mapping a 
      1-dimensional numpy array x to a scalar value.
      E.g. mean(x), std_deviation(x) that calculate global and / or
      ROI mean and standard deviations.
      


    """
    def __init__(self, reference_image, roi_mask_dict = None, metrics_dict = None,
                       statistics_dict = None):
    
        # the reference image
        self.reference_image = reference_image

        self.roi_indices_dict = {}

        for key, value in roi_mask_dict.items():
          self.roi_indices_dict[key] = np.where(roi_mask_dict[key] == 1)

        self.metrics_dict = metrics_dict

        self.statistics_dict = statistics_dict

    def eval(self, iteration, last_cost , test_image):
        r""" Callback function called by CIL algorithm that calculates global and local
            metrics and measures

        Parameters
        ----------

        test_image : ImageData
          test image where metrics and measure should be computed on

        """
        
        # (1) calculate global metrics and statistics
        global_metrics    = {}
        global_statistics = {}

        for metric_name, metric in self.metrics_dict.items():
            global_metrics[metric_name] = metric(test_image.as_array().ravel(), 
                                                 self.reference_image.as_array().ravel())

        for statistic_name, statistic in self.statistics_dict.items():
            global_statistics[statistic_name] = statistic(test_image.as_array().ravel())
       
  
        print('\nglobal metrics')
        print(global_metrics)
        print('\nglobal statistics')
        print(global_statistics)

        # (2) caluclate local metrics and statistics
        all_roi_metrics    = {}
        all_roi_statistics = {}

        for roi_name, roi_inds in self.roi_indices_dict.items():
          roi_metrics    = {}
          roi_statistics = {}

          for metric_name, metric in self.metrics_dict.items():
              roi_metrics[metric_name] = metric(test_image.as_array()[roi_inds].ravel(), 
                                                self.reference_image.as_array()[roi_inds].ravel())

          all_roi_metrics[roi_name] = roi_metrics

          for statistic_name, statistic in self.statistics_dict.items():
              roi_statistics[statistic_name] = statistic(test_image.as_array()[roi_inds].ravel())

          all_roi_statistics[roi_name] = roi_statistics

        print('\nROI metrics')
        print(all_roi_metrics)
        print('\nROI statistics')
        print(all_roi_statistics)

        # (3) save / log results
        
#------------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    # seed the random generator to setup random test image
    np.random.seed(1)

    # image dimension / shape of test images
    image_shape = (3,4,5)

    # setup standard image geometry
    image_geom = cil.framework.ImageGeometry(voxel_num_x = image_shape[2], 
                                             voxel_num_y = image_shape[1],
                                             voxel_num_z = image_shape[0])


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


    # instanciate ImageQualityCallback
    img_qual_callback = ImageQualityCallback(ref_image, roi_mask_dict = roi_image_dict,
                                             metrics_dict = {'MSE':MSE, 'MAE':MAE, 'PSNR':PSNR},
                                             statistics_dict = {'MEAN': (lambda x: x.mean()),
                                                                'STDDEV': (lambda x: x.std())})

    img_qual_callback.eval(1, 1, test_image)
