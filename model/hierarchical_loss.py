'''Hierarchical Loss Network
'''

import pickle
import torch
import torch.nn as nn
from helper import read_meta


class HierarchicalLossNetwork:
    '''Logics to calculate the loss of the model.
    '''

    def __init__(self, metafile_path, hierarchical_labels, device='cpu', total_level=2, alpha=1, beta=0.8, p_loss=3):
        '''Param init.
        '''
        self.total_level = total_level
        self.alpha = alpha
        self.beta = beta
        self.p_loss = p_loss
        self.device = device
        self.level_one_labels, self.level_two_labels = read_meta(metafile=metafile_path)
        self.hierarchical_labels = hierarchical_labels
        self.numeric_hierarchy = self.words_to_indices()


    def words_to_indices(self):
        '''Convert the classes from words to indices.
        '''
        numeric_hierarchy = {}
        for k, v in self.hierarchical_labels.items():
            numeric_hierarchy[self.level_one_labels.index(k)] = [self.level_two_labels.index(i) for i in v]

        return numeric_hierarchy


    def check_hierarchy(self, current_level, previous_level):
        '''Check if the predicted class at level l is a children of the class predicted at level l-1 for the entire batch.
        '''

        #check using the dictionary whether the current level's prediction belongs to the superclass (prediction from the prev layer)
        bool_tensor = [not current_level[i] in self.numeric_hierarchy[previous_level[i].item()] for i in range(previous_level.size()[0])]

        return torch.FloatTensor(bool_tensor).to(self.device)


    def calculate_lloss(self, predictions, true_labels):
        '''Calculates the layer loss.
        '''

        lloss = 0
        for l in range(self.total_level):

            lloss += nn.CrossEntropyLoss()(predictions[l], true_labels[l])

        return self.alpha * lloss

    def calculate_dloss(self, predictions, true_labels):
        '''Calculate the dependence loss.
        '''

        dloss = 0
        for l in range(1, self.total_level):

            current_lvl_pred = torch.argmax(nn.Softmax(dim=1)(predictions[l]), dim=1)
            prev_lvl_pred = torch.argmax(nn.Softmax(dim=1)(predictions[l-1]), dim=1)

            D_l = self.check_hierarchy(current_lvl_pred, prev_lvl_pred)

            l_prev = torch.where(prev_lvl_pred == true_labels[l-1], torch.FloatTensor([0]).to(self.device), torch.FloatTensor([1]).to(self.device))
            l_curr = torch.where(current_lvl_pred == true_labels[l], torch.FloatTensor([0]).to(self.device), torch.FloatTensor([1]).to(self.device))

            dloss += torch.sum(torch.pow(self.p_loss, D_l*l_prev)*torch.pow(self.p_loss, D_l*l_curr) - 1)

        return self.beta * dloss
