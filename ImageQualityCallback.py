#TODO: - getter and setter methods

# Discussion Tue afternoon:
# - implement metrics / statistics that work on ImageData?
# - handle non-scalar return values
# - add pre-processing function that is called before calculating metrics / stats

import numpy as np
import cil.framework
import tensorboardX

from scipy.ndimage import gaussian_filter

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

    reference_image: CIL or SIRF ImageData
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

    statistics_dict : list of floats
      Containing FWHMs (mm) of Gaussian post smoothing kernels to be applied
      before calculating all metrics and statistics.
      Default [5., 8.]


    """
    def __init__(self, reference_image, 
                       tb_summary_writer, 
                       roi_mask_dict   = None,
                       metrics_dict    = None,
                       statistics_dict = None,
                       post_smoothing_fwhms_mm_list = [5., 8.]):
    
        # the reference image
        self.reference_image = reference_image

        # tensorboard summary writer
        self.tb_summary_writer = tb_summary_writer

        self.roi_indices_dict = {}

        if roi_mask_dict is not None:
            for key, value in roi_mask_dict.items():
                self.roi_indices_dict[key] = np.where(roi_mask_dict[key].as_array() == 1)
        else:
            self.roi_indices_dict = None

        self.metrics_dict = metrics_dict

        self.statistics_dict = statistics_dict

        self.post_smoothing_fwhms_mm_list = post_smoothing_fwhms_mm_list
        if 0 not in self.post_smoothing_fwhms_mm_list:
            self.post_smoothing_fwhms_mm_list.insert(0,0)

        self.voxel_size_mm = (ref_image.geometry.voxel_size_z,
                              ref_image.geometry.voxel_size_y,
                              ref_image.geometry.voxel_size_x)

    def eval(self, iteration, last_cost , test_image):
        r""" Callback function called by CIL algorithm that calculates global and local
             metrics and measures

        Parameters
        ----------

        iteration : int
          current iteration

        last_cost : float
          current value of objective function

        test_image : CIL or SIRF ImageData
          test image where metrics and measure should be computed on

        """
       
        # get numpy arrays behind test ImageData
        test_image_array      = test_image.as_array()
        reference_image_array = self.reference_image.as_array()

        for post_smoothing_fwhm_mm in self.post_smoothing_fwhms_mm_list:
            if post_smoothing_fwhm_mm > 0:
                ps_str = f'_{post_smoothing_fwhm_mm}mm_smoothed'
            else:
                ps_str = ''


            if post_smoothing_fwhm_mm > 0:
                sig = post_smoothing_fwhm_mm / (2.35*np.array(self.voxel_size_mm))
                test_image_array_ps      = gaussian_filter(test_image_array, sig)
                reference_image_array_ps = gaussian_filter(reference_image_array, sig)
            else:
                test_image_array_ps      = test_image_array
                reference_image_array_ps = reference_image_array

            # (1) calculate global metrics and statistics
            global_metrics    = {}
            global_statistics = {}

            if self.metrics_dict is not None:
                for metric_name, metric in self.metrics_dict.items():
                    global_metrics[metric_name] = metric(test_image_array_ps.ravel(), 
                                                         reference_image_array_ps.ravel())
                self.tb_summary_writer.add_scalars(f'global_metrics{ps_str}', global_metrics, iteration)

            if self.statistics_dict is not None:
                for statistic_name, statistic in self.statistics_dict.items():
                    global_statistics[statistic_name] = statistic(test_image_array_ps.ravel())
                self.tb_summary_writer.add_scalars(f'global_statistics{ps_str}', global_statistics, iteration)
  
            # (2) caluclate local metrics and statistics
            if self.roi_indices_dict is not None:
                for roi_name, roi_inds in self.roi_indices_dict.items():
                    roi_metrics    = {}
                    roi_statistics = {}

                    if self.metrics_dict is not None:
                        for metric_name, metric in self.metrics_dict.items():
                            roi_metrics[metric_name] = metric(test_image_array_ps[roi_inds], 
                                                              reference_image_array_ps[roi_inds])
                        self.tb_summary_writer.add_scalars(f'{roi_name}_metrics{ps_str}', roi_metrics, iteration)

                    if self.statistics_dict is not None:
                        for statistic_name, statistic in self.statistics_dict.items():
                            roi_statistics[statistic_name] = statistic(test_image_array_ps[roi_inds])
                        self.tb_summary_writer.add_scalars(f'{roi_name}_statistics{ps_str}', roi_statistics, iteration)

            # (3) log the value of the cost function
            self.tb_summary_writer.add_scalar('cost', last_cost, iteration)
#------------------------------------------------------------------------------------------------------


if __name__ == '__main__':

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
    from datetime import datetime
    dt_string = datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_summary_writer = tensorboardX.SummaryWriter(f'runs/exp-{dt_string}')

    # instanciate ImageQualityCallback
    img_qual_callback = ImageQualityCallback(ref_image, tb_summary_writer,
                                             roi_mask_dict = roi_image_dict,
                                             metrics_dict = {'MSE':MSE, 'MAE':MAE, 'PSNR':PSNR},
                                             statistics_dict = {'MEAN': (lambda x: x.mean()),
                                                                'STDDEV': (lambda x: x.std()),
                                                                'MAX': (lambda x: x.max())})

    img_qual_callback.eval(1, 1, test_image)
    img_qual_callback.eval(2, 1, test_image*1.1)
    img_qual_callback.eval(3, 1, test_image*0.9)
    img_qual_callback.eval(4, 1, test_image*1.2)

    tb_summary_writer.close()
