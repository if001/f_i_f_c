import pandas as pd
import matplotlib.pyplot as plt


def main():
    csv_file = "./training_log.csv"
    csv_file = "./test.csv"
    df = pd.read_csv(csv_file, sep=',',
                     encoding='utf-8-sig', engine='python')
    df.set_index('epoch', inplace=True)

    # plt.figure()
    df.plot()
    plt.show()


if __name__ == "__main__":
    main()
