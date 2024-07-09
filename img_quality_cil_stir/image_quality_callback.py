from numbers import Number

import numpy as np
from scipy.ndimage import gaussian_filter

from cil.optimisation.utilities.callbacks import Callback


class GaussianFilter:
    """Gaussian filter to be applied to the input image array

    Parameters
    ----------
    post_smoothing_fwhm_mm : float
      full width at half maximum of the Gaussian filter in mm
    voxel_size_mm : tuple of floats
      Gives the voxel size in mm. Currently only homogeneous voxel sizes are supported.
    """
    def __init__(self, post_smoothing_fwhm_mm: float, voxel_size_mm: float):
        self.post_smoothing_fwhm_mm = post_smoothing_fwhm_mm
        if post_smoothing_fwhm_mm < 0:
            raise ValueError('post_smoothing_fwhm_mm must be a positive number')
        self.voxel_size_mm = voxel_size_mm

    def __call__(self, image_array):
        """Apply Gaussian filter to the input image array

        Parameters
        ----------
        image_array : numpy.ndarray
          input image array

        Returns
        -------
        numpy.ndarray of the same shape as the input image array
        """
        sig = self.post_smoothing_fwhm_mm / (2.35 * np.array(self.voxel_size_mm))
        return gaussian_filter(image_array, sig)


class ImageQualityCallback(Callback):
    """
    For use by `cil.optimisation.algorithms.Algorithm.run(callbacks=...)`
    to calculate global & local metrics.
    """
    def __init__(self, reference_image, tb_summary_writer, roi_mask_dict=None, metrics_dict=None, statistics_dict=None,
                 filter=None):
        """
        Parameters
        ----------
        reference_image : CIL or STIR ImageData
          containing the reference image used to calculate the metrics
        tb_summary_writer : tensorboardX SummaryWriter
          summary writer used to log results to tensorboard event files
        roi_mask_dict : dict[str, ImageData]
          One binary ImageData object (of same dimensions as the reference image)
          for every ROI to be evaluated. Voxels with values == 1 are considered part of the ROI.
        metrics_dict : dict[str, Callable] of named functions `f(y: 1darray, x: 1darray) -> scalar | ndarray`
          e.g. `{"MSE": skimage.metrics.mean_squared_error}`.
        statistics_dict : dict[str, Callable] of named functions `f(x: 1darray) -> scalar | ndarray`
          e.g. `{"mean": np.mean, "std": np.stdev}`.
        filter : dict[str, Callable] of named filters `f(x: ndarray) -> ndarray`
          Filter to be applied before calculating all metrics and statistics.
        """
        self.reference_image = reference_image
        self.tb_summary_writer = tb_summary_writer
        self.roi_indices = {}
        for key, value in (roi_mask_dict or {}).items():
            self.roi_indices[key] = np.where(value.as_array() == 1)
        self.metrics = metrics_dict or {}
        self.statistics = statistics_dict or {}
        self.filter = filter
        # get the voxel sizes from the input reference image
        # i.e. `voxel_sizes` (STIR) or `geometry.voxel_size_*` (CIL)
        if hasattr(reference_image, 'voxel_sizes'):
            # STIR image
            self.voxel_size_mm = reference_image.voxel_sizes()
        elif hasattr(reference_image, 'geometry'):
            if hasattr(reference_image.geometry, 'voxel_size_x'):
                # CIL image
                if reference_image.ndim == 3:
                    self.voxel_size_mm = (reference_image.geometry.voxel_size_z, reference_image.geometry.voxel_size_y,
                                          reference_image.geometry.voxel_size_x)
                elif reference_image.ndim == 2:
                    self.voxel_size_mm = (reference_image.geometry.voxel_size_y, reference_image.geometry.voxel_size_x)
                else:
                    raise IndexError(f"Expected 2D or 3D images; received {reference_image.ndim}D")
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    @property
    def filter(self):
        return self._filter

    @filter.setter
    def filter(self, filters):
        self._filter = filters or {'': None}

    def __call__(self, algorithm):
        iteration = algorithm.iteration
        if iteration % algorithm.update_objective_interval != 0 and iteration != algorithm.max_iteration:
            return
        objective = algorithm.get_last_objective(return_all=False)
        test_image = algorithm.x # CIL or SIRF ImageData

        # (0) objective value
        self.tb_summary_writer.add_scalar('objective', objective, iteration)

        test_im_arr, ref_im_arr = test_image.as_array(), self.reference_image.as_array()

        # for post_smoothing_fwhm_mm in self.post_smoothing_fwhms_mm_list:
        for filter_name, filter_func in self.filter.items():
            if filter_func is not None:
                test_im, ref_im = map(filter_func, (test_im_arr, ref_im_arr))

            # (1) global metrics & statistics
            self._log_metrics_stats(ref_im.ravel(), test_im.ravel(), "Global_", filter_name, iteration)

            # (2) local metrics & statistics
            for roi_name, roi_inds in self.roi_indices.items():
                self._log_metrics_stats(ref_im[roi_inds], test_im[roi_inds], f"Local_{roi_name}_", filter_name,
                                        iteration)

    def _log_metrics_stats(self, ref_im, test_im, prefix, suffix, iteration):
        for metric_name, metric in self.metrics.items():
            met = metric(ref_im, test_im)
            if isinstance(met, Number): # scalar
                self.tb_summary_writer.add_scalar(f'{prefix}{metric_name}{suffix}', met, iteration)
            else:                       # vector
                for im, m in enumerate(met.ravel()):
                    self.tb_summary_writer.add_scalar(f'{prefix}{metric_name}_{im}{suffix}', m, iteration)

        for statistic_name, statistic in self.statistics.items():
            stat = statistic(test_im)
            if isinstance(stat, Number): # scalar
                self.tb_summary_writer.add_scalar(f'{prefix}{statistic_name}{suffix}', stat, iteration)
            else:                        # vector
                for ist, st in enumerate(stat.ravel()):
                    self.tb_summary_writer.add_scalar(f'{prefix}{statistic_name}_{ist}{suffix}', st, iteration)
