# fastText-PyTorch
Custom PyTorch implementation of fastText algorithm


## Result
- Model training on AG's corpus of news articles [link](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html).

- Batch train loss / accuracy:
![p1](result/bigram/batch_loss_acc.jpg)

- Test loss / accuracy:
![p2](result/bigram/test_loss_acc.jpg)

- Train confusion matrix:
![p3](result/bigram/conf_m_train.jpg)

- Test confusion matrix:
![p4](result/bigram/conf_m_test.jpg)

## Influence of batch size on training time in multiprocessing.
![p5](result/multiproc_time.jpg)

