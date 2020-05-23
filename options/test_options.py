from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        #  Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')

        # 为了计算IS增加的参数
        parser.add_argument('--config', type=str, default='configs/edges2handbags_folder',
                            help='Path to the config file.')
        parser.add_argument('--input_folder', type=str, help="input image folder")
        parser.add_argument('--output_folder', type=str, help="output image folder")
        parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
        parser.add_argument('--a2b', type=int, help="1 for a2b and others for b2a", default=1)
        parser.add_argument('--seed', type=int, default=1, help="random seed")
        parser.add_argument('--num_style', type=int, default=10, help="number of styles to sample")
        parser.add_argument('--synchronized', action='store_true', help="whether use synchronized style code or not")
        parser.add_argument('--output_only', action='store_true', help="whether use synchronized style code or not")
        parser.add_argument('--output_path', type=str, default='.',
                            help="path for logs, checkpoints, and VGG model weight")
        parser.add_argument('--trainer', type=str, default='InstaGAN', help="InstaGAN") # 改为instagan
        parser.add_argument('--compute_IS', action='store_true', default=True,
                            help="whether to compute Inception Score or not")
        parser.add_argument('--compute_fid', action='store_true', default=True,
                            help="whether to compute fid Score or not")
        parser.add_argument('--compute_CIS', action='store_true', default=True,
                            help="whether to compute Conditional Inception Score or not")
        parser.add_argument('--inception_a', type=str, default='.',
                            help="path to the pretrained inception network for domain A")
        parser.add_argument('--inception_b', type=str, default='.',
                            help="path to the pretrained inception network for domain B")
        parser.add_argument('--cuda', type=bool, default=True, help="")
        parser.set_defaults(model='test')
        # To avoid cropping, the loadSize should be the same as fineSize
        parser.set_defaults(loadSize=parser.get_default('fineSize'))
        self.isTrain = False
        return parser
