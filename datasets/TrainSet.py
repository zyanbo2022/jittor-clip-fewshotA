from .utils import Datum
import os
template = ['a photo of {}.']

def read_data(filepath):
    with open(filepath, 'r') as f:
        out = []
        for line in f.readlines():
            img_path, label = line.strip().split(' ')
            impath = os.path.join('caches', img_path)
            item = Datum(
            impath=impath,
            label=int(label)
            )
            out.append(item)
        return out
        
# def read_list(result):
#     out = []
#     for items in result:
#         impath, label = items
#         item = Datum(
#             impath=impath,
#             label=int(label)
#             )
#         out.append(item)
#     return out



class TrainSet():
    
    def __init__(self):
        self.template = template
        self.train_x = read_data('caches/sampled_output.txt')
        self.val = read_data('caches/val15267s.txt')
        self.test = read_data('caches/test.txt')
        
# class TrainSet():
    
#     def __init__(self):
#         self.template = template
#         self.train_x = read_data('caches/train.txt')
#         self.val = read_data('caches/val.txt')
#         self.test = read_data('caches/test.txt')
        
#         # super().__init__(train_x=train, val=val, test=test)
       
# class TrainSet_double():

#     def __init__(self):

#         self.template = template
#         trainx = read_data('caches/sampled_output.txt')
#         valx = read_data('caches/val15267s.txt')
#         if os.path.exists('caches/temp_labels.txt'):
#             trainx_double = read_data('caches/temp_labels.txt')
#             trainx_double_paths = set(item.impath for item in trainx_double)
#             self.train_x = trainx + trainx_double
#             # self.val =  [item for item in valx if item.impath not in trainx_double_paths]
#             self.val =valx
#         else :
#             self.train_x = trainx
#             self.val =valx
#         self.test = read_data('caches/test.txt')

class TrainSet_double():

    def __init__(self):

        self.template = template
        trainx = read_data('caches/train.txt')
        valx = read_data('caches/val15267s.txt')
        if os.path.exists('caches/temp_labels.txt'):
            trainx_double = read_data('caches/temp_labels.txt')
            trainx_double_paths = set(item.impath for item in trainx_double)
            self.train_x = trainx + trainx_double
            # self.val =  [item for item in valx if item.impath not in trainx_double_paths]
            self.val =valx
        else :
            self.train_x = trainx
            self.val =valx
        self.test = read_data('caches/test.txt')