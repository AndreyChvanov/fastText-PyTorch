import pandas as pd
from fastText.model import FastText


def main():
    import time
    df = pd.read_csv('../data/train.csv')
    text = df.loc[::].Description
    labels = df.loc[:, 'Class Index'].values - 1
    df_test = pd.read_csv('../data/test.csv')
    text_test = df_test.loc[:, :].Description
    labels_test = df_test.loc[:, 'Class Index'].values - 1
    model = FastText(lines=text, gt=labels, numb_classes=4)
    model.load_test_data(test_text=text_test, test_labels=labels_test)
    start = time.time()
    for bs in [32, 64, 128]:
        start = time.time()
        # model.fit()
        model.fit_multiprocessing(bs)
        print(f'batch_size = {bs}final time', time.time() - start)

if __name__ == '__main__':
    main()
