import time
from concurrent.futures import ProcessPoolExecutor


def func_2(x, y):
    for n in range(3):
        time.sleep(1)
        print(f"func_2 - {n} ({x}, {y})")
    return f"result - ({x}, {y})"


def main():
    print("start")
    # プロセスプールを作成し、最大4つのプロセスで並列処理を実行
    with ProcessPoolExecutor(max_workers=4) as executor:
        # func_2を並列に実行
        # 引数はタプルで渡す
        # mapは引数を順番に渡すので、func_2の引数は2つ必要
        # ここでは、func_2に対して2つのリストを渡している
        results = executor.map(func_2, ["A", "B", "C", "D"], ["V", "X", "Y", "Z"])

    print(list(results))
    print("end")


if __name__ == "__main__":
    main()
