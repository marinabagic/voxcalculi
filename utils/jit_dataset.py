import os
import sys
import time

import torch
from torchtext.data import Dataset, Iterator, Batch


class JitDataset(Dataset):

    def __init__(self, path, exts, fields, **kwargs):

        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]

        xy_path, index_path = tuple(os.path.expanduser(path + x) for x in exts)

        examples = []

        if not os.path.exists(xy_path):
            print("Error: cannot find XY file: {}".format(xy_path))
            sys.exit(1)

        self.xy_path = xy_path

        print("  loading index examples from: {}".format(index_path))
        started = time.time()
        examples = torch.load(index_path)
        elapsed = time.time() - started

        print("  built {:,d} examples ({:.2f} secs)\n".format(len(examples), elapsed))

        super(JitDataset, self).__init__(examples, fields, **kwargs)
    
    @classmethod
    def splits(cls, exts, fields, path=None, root='.data',
               train='train', validation='val', test='test', **kwargs):
        """Create dataset objects for splits of a TranslationDataset.
        """
        if path is None:
            path = cls.download(root)

        train_data = None if train is None else cls(
            os.path.join(path, train), exts, fields, **kwargs)

        val_data = None if validation is None else cls(
            os.path.join(path, validation), exts, fields, **kwargs)

        test_data = None if test is None else cls(
            os.path.join(path, test), exts, fields, **kwargs)

        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)

class MyExample(object): super

class JitIterator(Iterator):
    def __init__(self, dataset, batch_size, sort_key=None, device=None,
                 batch_size_fn=None, train=True,
                 repeat=False, shuffle=None, sort=None,
                 sort_within_batch=None):

        self.xy_path = dataset.xy_path
        self.xy_file = open(self.xy_path, "r")
    
        super(JitIterator, self).__init__(dataset, batch_size, sort_key, device,
            batch_size_fn, train, repeat, shuffle, sort, sort_within_batch)

    def convert_minibatch(self, minibatch):
        examples = []

        for index in minibatch:
            self.xy_file.seek(index)
            text = self.xy_file.readline().replace("\n", "")

            src, trg = text.split("\t")
            example = MyExample()
            example.src = src
            example.trg = trg
            examples.append(example)

        return examples

    def __iter__(self):
        report_every = 1000
        
        start_wall = time.time()
        total_convert = 0
        total_both = 0

        while True:
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1

                start = time.time()
                minibatch = self.convert_minibatch(minibatch)

                total_convert += (time.time() - start)

                if self.sort_within_batch:
                    if self.sort:
                        minibatch.reverse()
                    else:
                        minibatch.sort(key=self.sort_key, reverse=True)

                mini = Batch(minibatch, self.dataset, self.device)

                total_both += (time.time() - start)

                if idx and report_every and idx % report_every == 0:
                    wall_elapsed = time.time() - start_wall

                    percent = 100*total_convert/wall_elapsed
                    print("{} minibatch CONVERT: total={:.2f}, wall={:.2f}, overhead={:.2f} %".format(report_every,
                        total_convert, wall_elapsed, percent))

                    percent = 100*total_both/wall_elapsed
                    print("{} minibatch CONVERT+PROCESS: total={:.2f}, wall={:.2f}, overhead={:.2f} %".format(report_every,
                        total_both, wall_elapsed, percent))

                    total_both = 0
                    total_convert = 0
                    start_wall = time.time()

                yield mini
            if not self.repeat:
                return
