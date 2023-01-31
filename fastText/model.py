import torch
import torch.nn as nn
import torch.nn.functional as F
from dictionary import Dictionary, BUCKET
from utils.metrics import conf_matrix
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
from utils.metrics import accuracy
import torch.multiprocessing as mp
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np


def train(rank, model, loader):
    torch.manual_seed(0 + rank)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1, momentum=0.9)
    pid = os.getpid()
    for epoch in range(0, 1):
        for i, batch in enumerate(loader):
            optimizer.zero_grad()
            x, gt = batch[0], batch[1]
            output = model(x)
            loss = criterion(output, gt)
            loss.backward()
            optimizer.step()
            # scheduler.step()
            acc = accuracy(output, gt)
            print(f'iter : {i}/{len(loader)} loss - {loss.item()} acc - {acc} pid {pid} ')


class FastTextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, numb_classes, pad_index):
        super(FastTextClassifier, self).__init__()
        self.emb_dim = embedding_dim
        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim, mode='mean', padding_idx=pad_index)
        self.head = nn.Linear(embedding_dim, numb_classes, bias=False)
        self.__init_weights()

    def __init_weights(self):
        nn.init.constant_(self.head.weight, val=0.0)
        nn.init.uniform(self.embedding.weight, -1 / self.emb_dim, 1 / self.emb_dim)

    def forward(self, indexes):
        emb = self.embedding(indexes)
        return F.log_softmax(self.head(emb), dim=-1)


class FastText:
    def __init__(self, lines, gt, numb_classes, embedding_dim=100, max_len=1024, log=True):
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.log_dir = os.path.join(self.root_dir, 'runs', f'exp_fasttext2_maxlen1024_lr1_scheduler_no_bigram')
        self.log = log
        self.text = lines
        self.labels = gt
        self.embedding_dim = embedding_dim
        self.number_of_classes = numb_classes
        self.dictionary = Dictionary()
        print('Preprocessing data...')
        self.processed_text = [self.dictionary.readWord(t) for t in self.text]
        self.max_len = max_len
        print('Build vocab')
        self.dictionary.readFromFile(self.processed_text)
        print(f'Vocab created. Size = {len(self.dictionary.words_)}')
        self.pad_index = len(self.dictionary.words_) + BUCKET
        self.train_dataset = self.__prepare_samples(self.processed_text, self.labels)
        self.test_dataset = None
        self.train_dataloader = None
        self.test_dataloader = None
        self.__init_model()
        self.loss = nn.NLLLoss()
        if self.log:
            self.writer = SummaryWriter(self.log_dir)
        else:
            self.writer = None

    def __init_model(self):
        self.model = FastTextClassifier(vocab_size=len(self.dictionary.words_) + BUCKET + 1,
                                        embedding_dim=self.embedding_dim,
                                        numb_classes=self.number_of_classes,
                                        pad_index=self.pad_index)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1, momentum=0.9)

    def __prepare_samples(self, processed_text, labels):
        inputs_ids = []
        for line in processed_text:
            tokens = line.split()
            ids = torch.LongTensor(self.dictionary.getLine(tokens))
            if len(ids) > self.max_len:
                ids = ids[:self.max_len]
            else:
                ids = torch.concat([ids, torch.LongTensor([self.pad_index] * (self.max_len - len(ids)))])
            inputs_ids.append(ids)

        inputs_ids = torch.vstack(inputs_ids)
        inputs_ids = torch.LongTensor(inputs_ids)
        labels = torch.LongTensor(labels.astype(int))
        dataset = TensorDataset(inputs_ids, labels)
        return dataset

    def load_test_data(self, test_text, test_labels):
        print('Processing test data...')
        processed_text = [self.dictionary.readWord(t) for t in test_text]
        self.test_dataset = self.__prepare_samples(processed_text, test_labels)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=128, shuffle=True, drop_last=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=128, drop_last=False)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=1,
                                                             steps_per_epoch=len(self.train_dataloader),
                                                             pct_start=0.0, anneal_strategy="linear", epochs=5)

    def fit(self):
        print('Start train...')
        for epoch in range(5):
            print(f'Processed epoch - {epoch}')
            for i, batch in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                x, gt = batch[0], batch[1]
                output = self.model(x)
                loss = self.loss(output, gt)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                acc = accuracy(output, gt)
                print(f'iter : {i}/{len(self.train_dataloader)} loss - {loss.item()} acc - {acc} ')
                if self.writer is not None:
                    self.writer.add_scalar('Batch/batch_loss', loss, epoch * len(self.train_dataloader) + i)
                    self.writer.add_scalar('Batch/batch_acc', acc, epoch * len(self.train_dataloader) + i)
            self.eval(epoch, train=True)
            self.eval(epoch, train=False)

    def eval(self, epoch, train):
        self.model.eval()
        total_accuracy = []
        total_loss = []
        preds = []
        labels = []
        if train:
            dataloader = self.train_dataloader
        else:
            dataloader = self.test_dataloader
        with torch.no_grad():
            for batch in dataloader:
                x, gt = batch[0], batch[1]
                output = self.model(x)
                loss = self.loss(output, gt)
                preds.extend(output)
                labels.extend(gt)
                sum_accuracy = accuracy(output, gt).item() * gt.size(0)
                total_accuracy.append(sum_accuracy)
                sum_loss = loss.item() * gt.size(0)
                total_loss.append(sum_loss)
        print(f'Dataset type: {"Train" if train else "Test"}')
        print(f'epoch: {epoch + 1}, total accuracy: {np.sum(total_accuracy) * 100 / len(labels)}%, '
              f'total loss: {np.sum(total_loss) / len(labels)} ')
        if self.writer is not None:
            self.writer.add_scalar(f'{"Train" if train else "Test"}/total_loss', np.sum(total_loss) / len(labels),
                                   epoch)
            self.writer.add_scalar(f'{"Train" if train else "Test"}/total_accuracy',
                                   np.sum(total_accuracy) / len(labels), epoch)

        if epoch == 4:
            preds = torch.vstack(preds)
            labels = torch.stack(labels)

            conf_matrix(preds, labels)

        return np.sum(total_accuracy) * 100 / len(labels)

    def fit_multiprocessing(self, batch_size):
        mp.set_start_method('spawn', force=True)
        self.model.share_memory()
        processes = []
        num_processes = 6
        for rank in range(num_processes):
            loader = DataLoader(self.train_dataset, sampler=DistributedSampler(dataset=self.train_dataset,
                                                                               num_replicas=num_processes, rank=rank),
                                batch_size=batch_size, drop_last=True)

            p = mp.Process(target=train, args=(rank, self.model, loader))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
