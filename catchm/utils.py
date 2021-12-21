from gensim.models.callbacks import CallbackAny2Vec

def check_edgelist(edgelist):

    if not isinstance(edgelist, list):
        edgelist = list(edgelist)
    

class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1
