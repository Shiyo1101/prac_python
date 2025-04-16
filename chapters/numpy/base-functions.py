# Numpyの基礎的な関数（モジュール）を学ぶ
# 参考URL: https://www.kikagaku.co.jp/kikagaku-blog/numpy-base/#i-13
import numpy as np


# 1次元配列の作成
def create_1d_array():
    arr = np.array([1, 2, 3, 4, 5])
    print("1D Array:", arr)


# 2次元配列の作成
def create_2d_array():
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    print("2D Array:\n", arr)


# 3次元配列の作成
def create_3d_array():
    arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    print("3D Array:\n", arr)


# 配列に要素を追加 append()
def append_to_array(arr, value):
    # 配列が1次元であることを確認
    if arr.ndim != 1:
        raise ValueError("Input array must be 1D.")  # エラーを出力

    # np.append()を使用して配列に新しい要素を追加
    new_arr = np.append(arr, value)
    print("Appended Array:", new_arr)

    return new_arr


# 配列の結合 concatenate()
def concatenate_arrays(arr1, arr2):
    # np.concatenate()を使用して2つの配列を結合
    # axis=0は行方向に結合することを意味する
    concatenated_arr = np.concatenate((arr1, arr2), axis=0)
    print("Concatenated Array:\n", concatenated_arr)

    return concatenated_arr


# 条件に基づいて値を選択する where()
def where_example():
    arr = np.array([1, 2, 3, 4, 5])

    # 条件を定義: 配列の要素が3より大きいかどうか
    condition = arr > 3

    # np.where()を使用して条件に基づいて値を選択
    # 条件がTrueの場合は元の値を使用し、Falseの場合は-1を使用
    result = np.where(condition, arr, -1)

    print("Where Example Result:", result)


# 指定した範囲の値を持つ配列を作成 arange()
def create_arange_array(start, stop, step):
    # np.arange()を使用して指定した範囲の値を持つ配列を作成
    arr = np.arange(start, stop, step)
    print("Arange Array:", arr)

    return arr


# 指定した範囲に従って等間隔の値を持つ配列を作成 linspace()
def create_linspace_array(start, stop, num):
    # np.linspace()を使用して指定した範囲に従って等間隔の値を持つ配列を作成
    arr = np.linspace(start, stop, num)
    print("Linspace Array:", arr)

    return arr


# 配列の要素の平均値 mean()
def mean_of_array(arr):
    # np.mean()を使用して配列の平均値を計算
    mean_value = np.mean(arr)
    print("Mean Value:", mean_value)

    return mean_value


# 配列の要素の合計 sum()
def sum_of_array(arr):
    # np.sum()を使用して配列の合計を計算
    sum_value = np.sum(arr)
    print("Sum Value:", sum_value)

    return sum_value


# 配列の要素の最大値 max()
def max_of_array(arr):
    # np.max()を使用して配列の最大値を計算
    max_value = np.max(arr)
    print("Max Value:", max_value)

    return max_value


# 配列の要素の最小値 min()
def min_of_array(arr):
    # np.min()を使用して配列の最小値を計算
    min_value = np.min(arr)
    print("Min Value:", min_value)

    return min_value


# 配列の形状を変更 reshape()
def reshape_array():
    # 2次元配列（2行3列）を作成
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    print("Original Array:\n", arr)

    # np.reshape()を使用して形状を変更
    # ここでは3行2列に変更
    reshaped_arr = arr.reshape(3, 2)
    print("Reshaped Array:\n", reshaped_arr)


# 配列の転置 transpose()
def transpose_array():
    # 2次元配列（2行3列）を作成
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    print("Original Array:\n", arr)

    # np.transpose()を使用して転置
    transposed_arr = np.transpose(arr)
    print("Transposed Array:\n", transposed_arr)


# 配列同士の内積 dot()
def dot_product(arr1, arr2):
    # np.dot()を使用して内積を計算
    dot_result = np.dot(arr1, arr2)
    print("Dot Product Result:", dot_result)

    return dot_result


# ランダムな値を持つ配列の作成 random()
def random_array(length):
    # np.random.rand()を使用してランダムな値を持つ配列を作成
    arr = np.random.rand(length)
    print("Random Array:\n", arr)

    return arr


# 配列の要素をソート sort()
def sort_array(arr):
    # np.sort()を使用して配列をソート
    sorted_arr = np.sort(arr)
    print("Sorted Array:", sorted_arr)

    return sorted_arr


# 配列の要素を逆順にする flip()
def flip_array(arr):
    # np.flip()を使用して配列を逆順にする
    flipped_arr = np.flip(arr)
    print("Flipped Array:", flipped_arr)

    return flipped_arr


def main():
    where_example()


if __name__ == "__main__":
    main()
