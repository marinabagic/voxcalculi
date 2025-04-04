import os

from torchtext.data import Field

from process_dm_math import DATASET_TARGET_DIR
from utils.jit_dataset import JitDataset, JitIterator

TRAIN_FILE_NAME = "extrapolate"
EVAL_FILE_NAME = "interpolate"
TEST_FILE_NAME = "extrapolate"

XY_FILE_ENDING = ".xy"
INDEX_FILE_ENDING = ".indexes_pt"
VOCAB_FILE_ENDING = ".vocab"

def is_jit_data_available(module_name):
    folder = os.path.join(DATASET_TARGET_DIR, module_name)
    fn_vocab = os.path.join(folder, TRAIN_FILE_NAME) + VOCAB_FILE_ENDING
    return os.path.exists(fn_vocab)

class JitDataLoader:
  def __init__(self, module_name, file_name, batch_size, is_train, device, log,
               vocab=None):
    self.module_name = module_name

    split_chars = lambda x: list(x)

    source = Field(tokenize=split_chars,
                   init_token='<sos>',
                   eos_token='<eos>',
                   batch_first=True)

    target = Field(tokenize=split_chars,
                   init_token='<sos>',
                   eos_token='<eos>',
                   batch_first=True)

    log("Loading JIT datasets ...")
    folder = os.path.join(DATASET_TARGET_DIR, module_name)

    dataset = JitDataset(path=os.path.join(folder, file_name),
                         exts=(XY_FILE_ENDING, INDEX_FILE_ENDING),
                         fields=(source, target))

    if vocab is None:
      log("Building vocab ...")
      fn_vocab = os.path.join(folder, TRAIN_FILE_NAME) + VOCAB_FILE_ENDING
      with open(fn_vocab, "r") as vfile:
          vocab_text = vfile.read()

      source.build_vocab([vocab_text])
      target.vocab = source.vocab
    else:
      target.vocab = vocab
      source.vocab = vocab

    log("Creating iterators ...")
    if is_train:
      iterator = JitIterator(dataset=dataset,
                             batch_size=batch_size,
                             train=False,
                             repeat=True,
                             shuffle=False,
                             device=device)
    else:
      iterator = JitIterator(dataset=dataset,
                             batch_size=batch_size,
                             train=False,
                             repeat=False,
                             shuffle=False,
                             device=device)

    self.dataset = dataset
    self.iterator = iterator
    self.source = source
    self.target = target

  def encode(self, str_list):
    return self.source.process(str_list)

  def decode(self, batch, remove_pad=False):
    itos = self.source.vocab.itos.copy()
    if remove_pad:
      itos[1] = ""
    str_list = ["".join([itos[idx] for idx in row]) for row in batch.tolist()]
    return str_list

   

