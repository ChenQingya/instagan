import importlib
import torch.utils.data
from data.base_data_loader import BaseDataLoader
from data.base_dataset import BaseDataset

# find_dataset_using_name函数：返回类，这里是Class UnalignedSegDataset
def find_dataset_using_name(dataset_name):
    # Given the option --dataset_mode [datasetname]
    # the file "data/datasetname_dataset.py"
    # will be imported.
    # eg."dataset_name" is "unaligned_seg"
    dataset_filename = "data." + dataset_name + "_dataset"
    # eg. "dataset_filename" is "data.unaligned_seg_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    # In the file, the class called DatasetNameDataset() will
    # be instantiated. It has to be a subclass of BaseDataset,
    # and it is case-insensitive.
    dataset = None
    # eg. "target_dataset_name" is "unalignedsegdataset"
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    # dict.items() : return (keys, values). eg."name,cls" is "unalignedsegdataset, cls",cls 在这里表示类UnalignedSegDataset
    # 类的静态函数、类函数、普通函数、全局变量以及一些内置的属性都是放在类__dict__里的

    for name, cls in datasetlib.__dict__.items():   # 此处datasetlib.__dict__.items(),返回：dict_items([('__name__', 'data.unaligned_seg_dataset'), ('__doc__', None),省略}，
                                                    # 其中有一项为('UnalignedSegDataset', <class 'data.unaligned_seg_dataset.UnalignedSegDataset'>)
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):    # 保证cls是类UnalignedSegDataset，所以cls的类型必须是BaseDataset的子类
            dataset = cls

    if dataset is None:
        print("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))
        exit(0)

    # 返回数据集对应的类，如Class UnalignedSegDataset
    return dataset

# 获取数据集的option，根据创建数据集的类（比如Class UnalignedSegDataset），调用类函数modify_commandline_options，获得options
def get_option_setter(dataset_name):
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options

# 创建数据集
def create_dataset(opt):
    # dataset：数据集的类名
    dataset = find_dataset_using_name(opt.dataset_mode)
    # 依据类名，创建数据集实例，真正数据集开始生成（包含domainA和domainB的图片和其segs）
    instance = dataset()
    instance.initialize(opt)
    print("dataset [%s] was created" % (instance.name()))
    return instance


def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader()
    data_loader.initialize(opt)
    return data_loader


# Wrapper class of Dataset class that performs
# multi-threaded data loading
class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        # self.dataset是实例，因为create_dataset返回数据集实例
        self.dataset = create_dataset(opt)
        # dataloader，加载数据，通过enumerate()枚举返回的data
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))   #num_workers:线程数目，dataloader多线程

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        # Loading Batched and Non-Batched Data,Automatic batching (default).
        # When batch_size (default 1) is not None,
        # the data loader yields batched samples instead of individual samples.
        # refer:https://pytorch.org/docs/stable/data.html#automatic-batching-default
        for i, data in enumerate(self.dataloader):  # 当执行enumerate(dataloader),是Multi-process data loading。refer：https://pytorch.org/docs/stable/data.html#multi-process-data-loading
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            # yield means 产出
            yield data
