import pandas as pd
import matplotlib.pyplot as plt
import sys


def main():
    csv_file = "./training_log.csv"
    csv_file = sys.argv[-1]

    df = pd.read_csv(csv_file, sep=',',
                     encoding='utf-8-sig', engine='python')
    df.set_index('epoch', inplace=True)
    print(df)
    # plt.figure()
    df.plot()
    plt.show()


if __name__ == "__main__":
    main()
