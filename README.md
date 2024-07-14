# imi-assignment-2

知能機械情報学レポート課題2

## 実行環境
- Windows 11
- Python 3.12.0 (@ rye 0.36.0)

## 実行方法
1. rye等で仮想環境を作成
```
$ cd imi_assignment_2
$ rye sync                           # ryeが使用可能な場合
$ pip install -r requirements.lock   # その他
```
2. スクリプトを実行
```
# ryeが使用可能な場合
$ rye run python ./test/test_adaboost.py
$ rye run python ./test/test_kernel_k_means.py

# その他
$ python ./test/test_adaboost.py
$ python ./test/test_kernel_k_means.py
```

# プログラムについて
- `decision_stump.py`：決定株の実装
- `adaboost.py`：AdaBoostの実装
- `kernel.py`：カーネル関数の実装
- `kernel_k_means.py`：Kernel K-meansの実装