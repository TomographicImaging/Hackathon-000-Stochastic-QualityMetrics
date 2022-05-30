import sirf.STIR as pet
import tensorboardX

from sirf.Utilities import examples_data_path
from datetime import datetime

test_image = pet.ImageData(examples_data_path('PET')+'/thorax_single_slice/emission.hv')
ref_image = pet.ImageData(examples_data_path('PET')+'/thorax_single_slice/emission.hv')
roi_image_dict = {}

# create a tensorboardX summary writer
dt_string = datetime.now().strftime("%Y%m%d-%H%M%S")
tb_summary_writer = tensorboardX.SummaryWriter(f'runs/exp-{dt_string}')
# instanciate ImageQualityCallback
img_qual_callback = ImageQualityCallback(ref_image, tb_summary_writer,
                                         roi_mask_dict = roi_image_dict,
                                         metrics_dict = {'MSE':MSE, 'MAE':MAE, 'PSNR':PSNR},
                                         statistics_dict = {'MEAN': (lambda x: x.mean()),
                                                            'STDDEV': (lambda x: x.std()),
                                                            'MAX': (lambda x: x.max()),
                                                            'COM': (lambda x: np.array([3,2,1]))},
                                         post_smoothing_fwhms_mm_list = [5., 8.])
img_qual_callback.eval(2, 1, test_image*1.1)
img_qual_callback.eval(3, 1, test_image*0.9)
img_qual_callback.eval(4, 1, test_image*1.2)
tb_summary_writer.close()
