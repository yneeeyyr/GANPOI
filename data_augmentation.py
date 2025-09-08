import random
import copy
import itertools

class CombinatorialEnumerate(object):
    #给定M种数据增强方法，通过连续调用这些方法，可以生成C(M，2)对增强样本，用于多视图对比学习。
    """Given M type of augmentations, and a original sequence, successively call \
    the augmentation 2*C(M, 2) times can generate total C(M, 2) augmentaion pairs. 
    In another word, the augmentation method pattern will repeat after every 2*C(M, 2) calls.
    
    For example, M = 3, the argumentation methods to be called are in following order: 
    a1, a2, a1, a3, a2, a3. Which formed three pair-wise augmentations:
    (a1, a2), (a1, a3), (a2, a3) for multi-view contrastive learning.
    """
    #item_similarity_model提供物品相似性的模型，用于增强方法，insert_rate插入操作的概率，max_insert_num_per_pos每个位置最多插入的物品数量，substitute_rate替换操作的概率，n_views对比视图的数量
    def __init__(self, tao=0.2, gamma=0.7, beta=0.2, \
                item_similarity_model=None, insert_rate=0.3, \
                max_insert_num_per_pos=3, substitute_rate=0.3, n_views=5):
        #初始化数据增强方法
        self.data_augmentation_methods = [Crop(tao=tao), Mask(gamma=gamma), Reorder(beta=gamma), 
                            Insert(item_similarity_model, insert_rate=insert_rate, 
                                max_insert_num_per_pos=max_insert_num_per_pos),
                            Substitute(item_similarity_model, substitute_rate=substitute_rate)]
        self.n_views = n_views
        # length of the list == C(M, 2)
        self.augmentation_idx_list = self.__get_augmentation_idx_order() #数据增强方法的索引顺序，由__get_augmentation_idx_order方法生成
        self.total_augmentation_samples = len(self.augmentation_idx_list) #计算并存储数据增强组合的总数，就是增强方法索引列表的长度
        self.cur_augmentation_idx_of_idx = 0 #当前使用的数据增强方法的索引的索引。用于在多个增强方法中循环选择当前使用的方法。

    #生成数据增强方法的索引顺序，用于按组合顺序应用增强方法。
    def __get_augmentation_idx_order(self):
        augmentation_idx_list = [] #空列表，用于存储数据增强方法的索引顺序
        for (view_1, view_2) in itertools.combinations([i for i in range(self.n_views)], 2): #生成所有可能的组合对，每种组合包含两个不同的视图索引。[i for i in range(self.n_views)]表示所有可用的视图列表
            #将每对组合中的两个视图索引依次添加到augmentation_idx_list中
            augmentation_idx_list.append(view_1)
            augmentation_idx_list.append(view_2)
        return augmentation_idx_list #返回生成的增强方法索引列表

    #根据当前的索引从多个数据增强方法中选择一个，对输入序列进行增强，并更新索引，以便下次调用时使用不同的方法
    def __call__(self, sequence): #sequence需要增强的输入序列
        augmentation_idx = self.augmentation_idx_list[self.cur_augmentation_idx_of_idx] #从 augmentation_idx_list 中获取当前增强方法的索引
        augment_method = self.data_augmentation_methods[augmentation_idx] #从data_augmentation_methods中选择对应的增强方法。augment_method是一个具体的数据增强方法
        # keep the index of index in range(0, C(M,2))
        self.cur_augmentation_idx_of_idx += 1 #更新 ur_augmentation_idx_of_idx使其指向下一个增强方法的索引
        self.cur_augmentation_idx_of_idx = self.cur_augmentation_idx_of_idx % self.total_augmentation_samples #通过取模操作确保cur_augmentation_idx_of_idx始终在augmentation_idx_list的有效范围内循环
        # print(augment_method.__class__.__name__)
        return augment_method(sequence) #调用选定的数据增强方法对输入序列进行增强

#根据输入序列的长度随机选择一种数据增强方法，并应用于输入的序列。根据给定的阈值augment_threshold区分长序列和短序列，分别使用不同的增强方法。该类在每次调用时随机选择一种数据增强方法应用于输入序列
class Random(object):
    """Randomly pick one data augmentation type every time call"""
    def __init__(self, tao=0.2, gamma=0.7, beta=0.2, \
                item_similarity_model=None, insert_rate=0.3, \
                max_insert_num_per_pos=3, substitute_rate=0.3,\
                augment_threshold=-1,
                augment_type_for_short='SIM'): #augment_threshold用于区分短序列和长序列的阈值。如果为-1，表示不区分长短序列，统一处理.augment_type_for_short定义短序列时使用的增强方法类型，SIM替换插入遮掩
        self.augment_threshold = augment_threshold
        self.augment_type_for_short = augment_type_for_short
        if self.augment_threshold == -1: #当augment_threshold=-1时，初始化统一的增强方法列表，裁剪，遮掩，重排序，插入和替代
            self.data_augmentation_methods = [Crop(tao=tao), Mask(gamma=gamma), Reorder(beta=beta), 
                                Insert(item_similarity_model, insert_rate=insert_rate, 
                                    max_insert_num_per_pos=max_insert_num_per_pos),
                                Substitute(item_similarity_model, substitute_rate=substitute_rate)]
            print("Total augmentation numbers: ", len(self.data_augmentation_methods))
        elif self.augment_threshold > 0: #如果augment_threshold大于0，表示需要区分长短序列
            print("short sequence augment type:", self.augment_type_for_short)#根据augment_type_for_short的值初始化短序列增强方法
            if self.augment_type_for_short == 'SI': #SI表示短序列设置插入和替代两种增强方法
                self.short_seq_data_aug_methods = [Insert(item_similarity_model, insert_rate=insert_rate, 
                                        max_insert_num_per_pos=max_insert_num_per_pos, 
                                        augment_threshold=self.augment_threshold),
                                    Substitute(item_similarity_model, substitute_rate=substitute_rate)]
            elif self.augment_type_for_short == 'SIM': #SIM表示短序列设置替代，插入和遮掩三种数据增强
                self.short_seq_data_aug_methods = [Insert(item_similarity_model, insert_rate=insert_rate, 
                                        max_insert_num_per_pos=max_insert_num_per_pos, 
                                        augment_threshold=self.augment_threshold),
                                    Substitute(item_similarity_model, substitute_rate=substitute_rate),
                                    Mask(gamma=gamma)]

            elif self.augment_type_for_short == 'SIR':#SIR表示替代，插入和重排序
                self.short_seq_data_aug_methods = [Insert(item_similarity_model, insert_rate=insert_rate, 
                                        max_insert_num_per_pos=max_insert_num_per_pos, 
                                        augment_threshold=self.augment_threshold),
                                    Substitute(item_similarity_model, substitute_rate=substitute_rate),
                                    Reorder(beta=gamma)]
            elif self.augment_type_for_short == 'SIC': #SIC表示替代插入和裁剪
                self.short_seq_data_aug_methods = [Insert(item_similarity_model, insert_rate=insert_rate, 
                                        max_insert_num_per_pos=max_insert_num_per_pos, 
                                        augment_threshold=self.augment_threshold),
                                    Substitute(item_similarity_model, substitute_rate=substitute_rate),
                                    Crop(tao=tao)]
            elif self.augment_type_for_short == 'SIMR': #SIMR表示替，插入，遮掩，重排序
                self.short_seq_data_aug_methods = [Insert(item_similarity_model, insert_rate=insert_rate, 
                                        max_insert_num_per_pos=max_insert_num_per_pos, 
                                        augment_threshold=self.augment_threshold),
                                    Substitute(item_similarity_model, substitute_rate=substitute_rate),
                                    Mask(gamma=gamma), Reorder(beta=gamma)]
            elif self.augment_type_for_short == 'SIMC': #SIMC表示替代，插入，遮掩，裁剪
                self.short_seq_data_aug_methods = [Insert(item_similarity_model, insert_rate=insert_rate, 
                                        max_insert_num_per_pos=max_insert_num_per_pos, 
                                        augment_threshold=self.augment_threshold),
                                    Substitute(item_similarity_model, substitute_rate=substitute_rate),
                                    Mask(gamma=gamma), Crop(tao=tao)]
            elif self.augment_type_for_short == 'SIRC': #SIRC表示替代，插入，重排序，裁剪
                self.short_seq_data_aug_methods = [Insert(item_similarity_model, insert_rate=insert_rate, 
                                        max_insert_num_per_pos=max_insert_num_per_pos, 
                                        augment_threshold=self.augment_threshold),
                                    Substitute(item_similarity_model, substitute_rate=substitute_rate),
                                    Reorder(beta=gamma), Crop(tao=tao)]
            else: #替代。插入。裁剪，遮掩，重排序
                print("all aug set for short sequences")
                self.short_seq_data_aug_methods = [Insert(item_similarity_model, insert_rate=insert_rate, 
                                        max_insert_num_per_pos=max_insert_num_per_pos, 
                                        augment_threshold=self.augment_threshold),
                                    Substitute(item_similarity_model, substitute_rate=substitute_rate),
                                   Crop(tao=tao), Mask(gamma=gamma), Reorder(beta=gamma)]                
            self.long_seq_data_aug_methods = [Insert(item_similarity_model, insert_rate=insert_rate, 
                                    max_insert_num_per_pos=max_insert_num_per_pos, 
                                    augment_threshold=self.augment_threshold),
                                Crop(tao=tao), Mask(gamma=gamma), Reorder(beta=gamma),
                                Substitute(item_similarity_model, substitute_rate=substitute_rate)] #初始化长序列的数据增强法
            print("Augmentation methods for Long sequences:", len(self.long_seq_data_aug_methods)) #打印可用方法
            print("Augmentation methods for short sequences:", len(self.short_seq_data_aug_methods))
        else: #如果augment_threshold小于-1，则抛出异常
            raise ValueError("Invalid data type.")

    def __call__(self, sequence): #调用数据增强操作
        if self.augment_threshold == -1: #如果augment_threshold为-1，则从data_augmentation_methods中随机选择一个方法并应用于sequence
            #randint generate int x in range: a <= x <= b
            augment_method_idx = random.randint(0, len(self.data_augmentation_methods)-1)
            augment_method = self.data_augmentation_methods[augment_method_idx]
            # print(augment_method.__class__.__name__) # debug usage
            return augment_method(sequence) #用选定的数据增强方法对输入序列进行处理，返回处理后的序列
        elif self.augment_threshold > 0: #根据输入序列的长度选择不同的增强方法
            seq_len = len(sequence) #计算输入序列的长度
            if seq_len > self.augment_threshold: #检查sequence的长度是否大于augment_threshold，如果是，则从long_seq_data_aug_methods中随机选择一个数据增强方法
                #randint generate int x in range: a <= x <= b
                augment_method_idx = random.randint(0, len(self.long_seq_data_aug_methods)-1)
                augment_method = self.long_seq_data_aug_methods[augment_method_idx]
                # print(augment_method.__class__.__name__) # debug usage
                return augment_method(sequence)
            elif seq_len <= self.augment_threshold: #如果sequence的长度小于或等于augment_threshold，则从short_seq_data_aug_methods中随机选择一个数据增强方法
                #randint generate int x in range: a <= x <= b
                augment_method_idx = random.randint(0, len(self.short_seq_data_aug_methods)-1)
                augment_method = self.short_seq_data_aug_methods[augment_method_idx]
                # print(augment_method.__class__.__name__) # debug usage
                return augment_method(sequence)                

#对比两个相似度模型的输出，并根据相似度得分选择最佳的推荐项
def _ensmeble_sim_models(top_k_one, top_k_two):
    # only support top k = 1 case so far
#     print("offline: ",top_k_one, "online: ", top_k_two)
    if top_k_one[0][1] >= top_k_two[0][1]:
        return [top_k_one[0][0]]
    else:
        return [top_k_two[0][0]]
    
class Insert(object): #在每次调用时在给定的序列中插入相似的项目
    """Insert similar items every time call"""
    #初始化
    def __init__(self, item_similarity_model, insert_rate=0.4, max_insert_num_per_pos=1,
            augment_threshold=14): #item_similarity_model项目相似度模型，max_insert_num_per_pos每个位置最大可插入的相似项目数量
        self.augment_threshold = augment_threshold
        if type(item_similarity_model) is list: #如果item_similarity_model是列表类型，表示使用了多个相似度模型。分别存储在item_sim_model_1，item_sim_model_2
            self.item_sim_model_1 = item_similarity_model[0]
            self.item_sim_model_2 = item_similarity_model[1]
            self.ensemble = True #表示使用了集成模型
        else: #如果不是列表类型，只存储一个模型，并设置ensemble为False
            self.item_similarity_model = item_similarity_model
            self.ensemble = False
        self.insert_rate = insert_rate #设置插入率
        self.max_insert_num_per_pos = max_insert_num_per_pos #每个位置最大可插入的相似项目数量
    
    #实现数据增强操作
    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence) #对输入序列进行深复制，避免原始序列被修改
        insert_nums = max(int(self.insert_rate*len(copied_sequence)), 1) #根据插入率计算需要插入的项目数量，最小为1
        insert_idx = random.sample([i for i in range(len(copied_sequence))], k = insert_nums) #随机选择insert_nums个索引位置，用于插入类似项目
        inserted_sequence = [] #用于存储插入后的序列
        for index, item in enumerate(copied_sequence):
            if index in insert_idx: #在insert_idx指定位置插入类似项目
                top_k = random.randint(1, max(1, int(self.max_insert_num_per_pos/insert_nums))) #随机确定插入的相似的项目数量
                if self.ensemble: #如果使用了集成模型，分别从两个模型中获取相似的项目
                    top_k_one = self.item_sim_model_1.most_similar(item,
                                            top_k=top_k, with_score=True)
                    top_k_two = self.item_sim_model_2.most_similar(item,
                                            top_k=top_k, with_score=True)
                    inserted_sequence += _ensmeble_sim_models(top_k_one, top_k_two) #调用_ensmeble_sim_models进行合并
                else: #没有使用集成模型，从单一模型中获取相似项目
                    inserted_sequence += self.item_similarity_model.most_similar(item,
                                            top_k=top_k)
            inserted_sequence += [item] #将插入后的项目加入inserted_sequence

        return inserted_sequence

#替代  将序列中的项目替换为相似的项目               
class Substitute(object):
    """Substitute with similar items"""
    def __init__(self, item_similarity_model, substitute_rate=0.1): #substitute_rate替换率
        if type(item_similarity_model) is list: ##如果item_similarity_model是列表类型，表示使用了多个相似度模型。分别存储在item_sim_model_1，item_sim_model_2
            self.item_sim_model_1 = item_similarity_model[0]
            self.item_sim_model_2 = item_similarity_model[1]
            self.ensemble = True ##表示使用了集成模型
        else: #如果不是列表类型，只存储一个模型，并设置ensemble为False
            self.item_similarity_model = item_similarity_model
            self.ensemble = False
        self.substitute_rate = substitute_rate

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence) #对输入序列进行深复制，避免原始序列被修改
        substitute_nums = max(int(self.substitute_rate*len(copied_sequence)), 1) #根据替代率计算需要替代的项目数量，最小为1
        substitute_idx = random.sample([i for i in range(len(copied_sequence))], k = substitute_nums) ##随机选择substitute_nums个索引位置，用于替代项目
        inserted_sequence = [] ##用于存储替代后的序列
        for index in substitute_idx:
            if self.ensemble:#如果使用了集成模型，分别从两个模型中获取相似的项目
                top_k_one = self.item_sim_model_1.most_similar(copied_sequence[index],
                                        with_score=True)
                top_k_two = self.item_sim_model_2.most_similar(copied_sequence[index],
                                        with_score=True)
                substitute_items = _ensmeble_sim_models(top_k_one, top_k_two) #调用_ensmeble_sim_models进行合并
                copied_sequence[index] = substitute_items[0] #替换原始项目
            else: #如果没有使用集成模型，从单一模型中获取最相似的项目，并替换原始项目
                copied_sequence[index] = copied_sequence[index] = self.item_similarity_model.most_similar(copied_sequence[index])[0]
        return copied_sequence

#裁剪 从原始序列中随机裁剪出一个子序列来生成新的序列。最终返回裁剪后的子序列
class Crop(object):
    """Randomly crop a subseq from the original sequence"""
    def __init__(self, tao=0.2): #tao裁剪比例因子
        self.tao = tao

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence) #对输入序列进行深复制，避免原始序列被修改
        sub_seq_length = int(self.tao*len(copied_sequence)) #计算需要裁剪出的子序列的长度
        #randint generate int x in range: a <= x <= b
        start_index = random.randint(0, len(copied_sequence)-sub_seq_length-1) #确定裁剪的起始位置。随机选择起始位置，使子序列可以完全包含在原始序列内
        if sub_seq_length<1: #序列长度非常短，直接返回包含一个元素的序列，该元素位于随机选择的起始位置
            return [copied_sequence[start_index]]
        else: #裁剪出从start_index开始，长度为sub_seq_length的子序列，并返回该子序列
            cropped_seq = copied_sequence[start_index:start_index+sub_seq_length]
            return cropped_seq

#遮掩   在输入序列中随机遮掩掉（替换为零）若干个元素来生成新的序列
class Mask(object):
    """Randomly mask k items given a sequence"""
    def __init__(self, gamma=0.7): #gamma遮掩比例因子。表示需要遮掩的元素数量相对于序列长度的比例。默认为0.7，即遮掩掉70%的序列元素
        self.gamma = gamma

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence) #对输入序列进行深复制，避免原始序列被修改
        mask_nums = int(self.gamma*len(copied_sequence)) #计算需要遮掩的元素数量
        mask = [0 for i in range(mask_nums)] #用于遮掩的值，这里是0，表示遮掩掉的元素被替换为0
        mask_idx = random.sample([i for i in range(len(copied_sequence))], k = mask_nums) #随机选择的要遮掩的元素的位置索引，长度为mask_nums。从copied_sequence随机选择mask_nums个位置
        for idx, mask_value in zip(mask_idx, mask):  #根据生成的索引位置，将对应位置的元素替换为0
            copied_sequence[idx] = mask_value
        return copied_sequence #返回遮掩后的序列

#重排序  随机打乱输入序列中的一个连续子序列来生成新的序列 
class Reorder(object):
    """Randomly shuffle a continuous sub-sequence"""
    def __init__(self, beta=0.2): #beta制子序列长度的比例因子
        self.beta = beta

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence) #对输入序列进行深复制，避免原始序列被修改
        sub_seq_length = int(self.beta*len(copied_sequence)) #需要打乱的子序列长度
        start_index = random.randint(0, len(copied_sequence)-sub_seq_length-1) #子序列的起始索引，随机生成
        sub_seq = copied_sequence[start_index:start_index+sub_seq_length] #提取的子序列，长度为sub_seq_length，起始位置start_index
        random.shuffle(sub_seq) #随机打乱子序列中的元素顺序
        reordered_seq = copied_sequence[:start_index] + sub_seq + \
                        copied_sequence[start_index+sub_seq_length:] #重排后的完整序列，由原始序列中未排序的部分加上打乱后的序列拼接而成 
        assert len(copied_sequence) == len(reordered_seq) #检查重排前后的序列长度是否一致
        return reordered_seq #返回重排后的序列

#测试不同的数据增强方法
if __name__ == '__main__':
    reorder = Reorder(beta=0.2) #创建实例
    sequence=[14052, 10908,  2776, 16243,  2726,  2961, 11962,  4672,  2224,
    5727,  4985,  9310,  2181,  3428,  4156, 16536,   180, 12044, 13700]
    rs = reorder(sequence)
    crop = Crop(tao=0.2)
    rs = crop(sequence)
    # rt = RandomType()
    # rs = rt(sequence)
    n_views = 5
    enum_type = CombinatorialEnumerateType(n_views=n_views)
    for i in range(40):
        if i == 20:
            print('-------')
        es = enum_type(sequence)