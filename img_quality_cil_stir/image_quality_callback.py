from cil.optimisation.utilities.callbacks import Callback
import numpy as np
from scipy.ndimage import gaussian_filter

class GaussianFilter():
    '''Gaussian filter to be applied to the input image array
    
    Parameters
    ----------
    post_smoothing_fwhm_mm : float
      full width at half maximum of the Gaussian filter in mm
    voxel_size_mm : tuple of floats
      Gives the voxel size in mm. Currently only homogeneous voxel sizes are supported.'''
    
    def __init__(self, post_smoothing_fwhm_mm: float, voxel_size_mm: float):
        self.post_smoothing_fwhm_mm = post_smoothing_fwhm_mm
        if post_smoothing_fwhm_mm < 0:
            raise ValueError('post_smoothing_fwhm_mm must be a positive number')
        self.voxel_size_mm = voxel_size_mm

    def __call__(self, image_array):
        '''Apply Gaussian filter to the input image array
        
        Parameters
        ----------
        image_array : numpy.ndarray
          input image array
          
        Returns
        -------
        numpy.ndarray of the same shape as the input image array
        '''
        sig = self.post_smoothing_fwhm_mm / (2.35*np.array(self.voxel_size_mm))
        return gaussian_filter(image_array, sig)
        

class ImageQualityCallback(Callback):
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

    filter : dict of filters
      Filter to be applied before calculating all metrics and statistics.
      The key of the dict is the name of the filter which will be used to identify
      the results in the tensorboard event files.
      If you don't want any of the post-smoothed metrics, pass an empty dict.
      Default {}
    
    voxel_size_mm : tuple of floats
      Gives the voxel size in mm (z, y, x).


    """
    def __init__(self, reference_image, 
                       tb_summary_writer, 
                       roi_mask_dict   = None,
                       metrics_dict    = None,
                       statistics_dict = None,
                       filter={}):
        
    
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

        self.filter = filter
        # get the voxel sizes from the input reference image
        # since STIR and CIL use different way to store the voxel sizes
        # we test for the attributes voxel_sizes (STIR) and geometry.voxel_size_x (CIL
        if hasattr(reference_image, 'voxel_sizes'):
            # STIR image
            self.voxel_size_mm = reference_image.voxel_sizes()
        elif hasattr(reference_image, 'geometry'):
            if hasattr(reference_image.geometry, 'voxel_size_x'):
                # CIL image
                if reference_image.ndim == 3:
                    self.voxel_size_mm = (reference_image.geometry.voxel_size_z,
                                        reference_image.geometry.voxel_size_y,
                                        reference_image.geometry.voxel_size_x)
                elif reference_image.ndim == 2:
                    self.voxel_size_mm = (reference_image.geometry.voxel_size_y,
                                        reference_image.geometry.voxel_size_x)
                else:
                    raise ValueError(
                        f'This {self.__class__.__name__} handles only 2D or 3D images, got a {reference_image.ndim}D image.'
                        )

            else:
                NotImplementedError
        else:
            NotImplementedError
            
    def __call__(self, algorithm):
        iteration = algorithm.iteration
        if iteration % algorithm.update_objective_interval != 0 and iteration != algorithm.max_iteration:
            return
        last_cost = algorithm.get_last_objective(return_all=False)
        # evaluate the test image
        test_image = algorithm.x
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

        # for post_smoothing_fwhm_mm in self.post_smoothing_fwhms_mm_list:
        if self.filter == {}:
            filters = {'':None}
        else:
            filters = self.filter
        for filter_name, filter in filters.items():
            ps_str = f'{filter_name}'

            if filter is None:
                test_image_array_ps      = test_image_array
                reference_image_array_ps = reference_image_array
            else:
                test_image_array_ps      = filter(test_image_array)
                reference_image_array_ps = filter(reference_image_array)

            # (1) calculate global metrics and statistics
            if self.metrics_dict is not None:
                for metric_name, metric in self.metrics_dict.items():
                    met = metric(reference_image_array_ps.ravel(), test_image_array_ps.ravel())
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
                            roi_met = metric(reference_image_array_ps[roi_inds], test_image_array_ps[roi_inds])
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
    
    def set_filters(self, filters):
        self.filter = filters
