import numpy as np
from scipy.ndimage import gaussian_filter

class ImageQualityCallback:
    r"""

    Parameters
    ----------

    reference_image: CIL or STIR ImageData
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
      two 1-dimensional numpy arrays x and y to a scalar value or a
      numpy.ndarray.
      x and y can be the voxel values of the whole images or the values of
      voxels in a ROI such that the metric can be computed on the whole
      images and optionally in the ROIs separately.

      E.g. f(x,y) could be MSE(x,y), PSNR(x,y), MAE(x,y)

      If the return value is an nd.array, the results will be saved
      as separate scalar values in the tensorboard results
      (one for each value in the array)

    statistics_dict : dictionary of lambda functions f(x) mapping a 
      1-dimensional numpy array x to a scalar value or a numpy.ndarray.
      E.g. mean(x), std_deviation(x) that calculate global and / or
      ROI mean and standard deviations.

      E.g. f(x) could be x.mean()

      If the return value is an nd.array, the results will be saved
      as separate scalar values in the tensorboard results
      (one for each value in the array)

    post_smoothing_fwhms_mm_list : list of floats
      Containing FWHMs (mm) of Gaussian post smoothing kernels to be applied
      before calculating all metrics and statistics.
      If you don't want any of the post-smoothed metrics, pass an empty list.
      Default []
    
    voxel_size_mm : tuple of floats
      Gives the voxel size in mm (z, y, x).


    """
    def __init__(self, reference_image, 
                       tb_summary_writer, 
                       roi_mask_dict   = None,
                       metrics_dict    = None,
                       statistics_dict = None,
                       post_smoothing_fwhms_mm_list = []):
    
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

        # get the voxel sizes from the input reference image
        # since STIR and CIL use different way to store the voxel sizes
        # we test for the attributes voxel_sizes (STIR) and geometry.voxel_size_x (CIL
        if hasattr(reference_image, 'voxel_sizes'):
            # STIR image
            self.voxel_size_mm = reference_image.voxel_sizes()
        elif hasattr(reference_image, 'geometry'):
            if hasattr(reference_image.geometry, 'voxel_size_x'):
                # CIL image
                self.voxel_size_mm = (reference_image.geometry.voxel_size_z,
                                      reference_image.geometry.voxel_size_y,
                                      reference_image.geometry.voxel_size_x)
            else:
                NotImplementedError
        else:
            NotImplementedError
            
    def eval(self, iteration, last_cost , test_image):
        r""" Callback function called by CIL algorithm that calculates global and local
             metrics and measures.
             The input arguments are fixed by the callback from the CIL algorithm class.

        Parameters
        ----------

        iteration : int
          current iteration

        last_cost : float
          current value of objective function

        test_image : CIL or SIRF ImageData
          test image where metrics and measure should be computed on

        """
       
        # (0) log the value of the cost function
        self.tb_summary_writer.add_scalar('cost', last_cost, iteration)

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
            if self.metrics_dict is not None:
                for metric_name, metric in self.metrics_dict.items():
                    met = metric(test_image_array_ps.ravel(), reference_image_array_ps.ravel())
                    # check if metric is scalar or vector valued
                    # for the 2nd case, we save each scalar value separately in the dict
                    if isinstance(met, np.ndarray):
                        for im, m in enumerate(met.ravel()):
                            self.tb_summary_writer.add_scalar(f'Global_{metric_name}_{im}{ps_str}',  m, iteration)
                    else:
                        self.tb_summary_writer.add_scalar(f'Global_{metric_name}{ps_str}',  met, iteration)

            if self.statistics_dict is not None:
                for statistic_name, statistic in self.statistics_dict.items():
                    stat = statistic(test_image_array_ps.ravel())
                    # check if statistic is scalar or vector valued
                    # for the 2nd case, we save each scalar value separately in the dict
                    if isinstance(stat, np.ndarray):
                        for ist, st in enumerate(stat.ravel()):
                            self.tb_summary_writer.add_scalar(f'Local_{statistic_name}_{ist}{ps_str}',  st, iteration)
                    else:
                        self.tb_summary_writer.add_scalar(f'Global_{statistic_name}{ps_str}',  stat, iteration)
  
            # (2) caluclate local metrics and statistics
            if self.roi_indices_dict is not None:
                for roi_name, roi_inds in self.roi_indices_dict.items():

                    if self.metrics_dict is not None:
                        for metric_name, metric in self.metrics_dict.items():
                            roi_met = metric(test_image_array_ps[roi_inds], reference_image_array_ps[roi_inds])
                            # check if metric is scalar or vector valued
                            # for the 2nd case, we save each scalar value separately in the dict
                            if isinstance(met, np.ndarray):
                                for im, m in enumerate(roi_met.ravel()):
                                    self.tb_summary_writer.add_scalar(f'Local_{roi_name}_{metric_name}_{im}{ps_str}',  m, iteration)
                            else:
                                self.tb_summary_writer.add_scalar(f'Local_{roi_name}_{metric_name}{ps_str}',  roi_met, iteration)

                    if self.statistics_dict is not None:
                        for statistic_name, statistic in self.statistics_dict.items():
                            roi_stat = statistic(test_image_array_ps[roi_inds])
                            # check if statistic is scalar or vector valued
                            # for the 2nd case, we save each scalar value separately in the dict
                            if isinstance(roi_stat, np.ndarray):
                                for ist, st in enumerate(roi_stat.ravel()):
                                    self.tb_summary_writer.add_scalar(f'Local_{roi_name}_{statistic_name}_{st}{ps_str}',  st, iteration)
                            else:
                                self.tb_summary_writer.add_scalar(f'Local_{roi_name}_{statistic_name}{ps_str}',  roi_stat, iteration)
