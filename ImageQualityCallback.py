#TODO: - multi resolution approach ("smoothing before calcuating metrics) / function to apply
#        to test / reference image data before calculating stuff
#      - getter and setter methods

# Discussion Tue afternoon:
# - implement metrics / statistics that work on ImageData?
# - handle non-scalar return values
# - add pre-processing function that is called before calculating metrics / stats

import numpy as np
import cil.framework
import tensorboardX

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

    tb_summary_writer ; tensorboardX SummaryWriter
      summary writer used to log results to tensorboard event files

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
    def __init__(self, reference_image, tb_summary_writer, roi_mask_dict = None, metrics_dict = None,
                       statistics_dict = None):
    
        # the reference image
        self.reference_image = reference_image
        self.reference_image_array = self.reference_image.as_array()

        # tensorboard summary writer
        self.tb_summary_writer = tb_summary_writer

        self.roi_indices_dict = {}

        for key, value in roi_mask_dict.items():
          self.roi_indices_dict[key] = np.where(roi_mask_dict[key].as_array() == 1)

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
       
        # get numpy arrays behing test ImageData
        test_image_array = test_image.as_array()

        # (1) calculate global metrics and statistics
        global_metrics    = {}
        global_statistics = {}

        for metric_name, metric in self.metrics_dict.items():
            global_metrics[metric_name] = metric(test_image_array.ravel(), 
                                                 self.reference_image_array.ravel())

        for statistic_name, statistic in self.statistics_dict.items():
            global_statistics[statistic_name] = statistic(test_image_array.ravel())
       
        self.tb_summary_writer.add_scalars('global_metrics', global_metrics, iteration)
        self.tb_summary_writer.add_scalars('global_statistics', global_statistics, iteration)
  
        # (2) caluclate local metrics and statistics

        for roi_name, roi_inds in self.roi_indices_dict.items():
            roi_metrics    = {}
            roi_statistics = {}

            for metric_name, metric in self.metrics_dict.items():
                roi_metrics[metric_name] = metric(test_image_array[roi_inds], 
                                                  self.reference_image_array[roi_inds])

            for statistic_name, statistic in self.statistics_dict.items():
                roi_statistics[statistic_name] = statistic(test_image_array[roi_inds])

            self.tb_summary_writer.add_scalars(f'{roi_name}_metrics', roi_metrics, iteration)
            self.tb_summary_writer.add_scalars(f'{roi_name}_statistics', roi_statistics, iteration)

        # (3) log the value of the cost function
        self.tb_summary_writer.add_scalar('cost', last_cost, iteration)
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


    # create a tensorboardX summary writer
    from datetime import datetime
    dt_string = datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_summary_writer = tensorboardX.SummaryWriter(f'runs/exp-{dt_string}')

    # instanciate ImageQualityCallback
    img_qual_callback = ImageQualityCallback(ref_image, tb_summary_writer,
                                             roi_mask_dict = roi_image_dict,
                                             metrics_dict = {'MSE':MSE, 'MAE':MAE, 'PSNR':PSNR},
                                             statistics_dict = {'MEAN': (lambda x: x.mean()),
                                                                'STDDEV': (lambda x: x.std())})

    img_qual_callback.eval(1, 1, test_image)
    img_qual_callback.eval(2, 1, test_image)
    img_qual_callback.eval(3, 1, test_image)
    img_qual_callback.eval(4, 1, test_image)

    tb_summary_writer.close()
