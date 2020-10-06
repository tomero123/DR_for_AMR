import os
from gensim.models.callbacks import CallbackAny2Vec


class EpochSaver(CallbackAny2Vec):
    '''Callback to save model after each epoch and show training parameters '''

    def __init__(self, savedir):
        self.savedir = savedir
        self.epoch = 1
        os.makedirs(self.savedir, exist_ok=True)

    def on_epoch_end(self, model):
        # loss = model.get_latest_training_loss()
        # print('Loss after epoch {}: {}'.format(self.epoch, loss))
        savepath = os.path.join(self.savedir, f"model_epoch_{self.epoch}.gz")
        model.save(savepath)
        print(
            "Epoch saved: {}".format(self.epoch),
            "Start next epoch ... ", sep="\n"
            )
        if os.path.isfile(os.path.join(self.savedir, f"model_epoch_{self.epoch - 1}.gz")):
            print("Previous model deleted ")
            os.remove(os.path.join(self.savedir, f"model_epoch_{self.epoch - 1}.gz"))
            os.remove(os.path.join(self.savedir, f"model_epoch_{self.epoch - 1}.gz.trainables.syn1neg.npz"))
            os.remove(os.path.join(self.savedir, f"model_epoch_{self.epoch - 1}.gz.wv.vectors.npz"))
        self.epoch += 1