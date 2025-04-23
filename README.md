# プロジェクト名

このプロジェクトは、XOR 問題を解くためのシンプルなニューラルネットワークの実装です。

## 必要条件

このプロジェクトを実行するには、以下のソフトウェアが必要です:

- Python 3.8 以上
- pip（Python パッケージ管理ツール）
- Git

## セットアップ手順

以下の手順にしたがって、このプロジェクトをセットアップしてください。

### 1. リポジトリをクローンする

まず、Git を使用してこのリポジトリをローカル環境にクローンします。

```bash
git clone https://github.com/your-username/your-repository.git
```

### 2. プロジェクトディレクトリに移動する

クローンしたリポジトリのディレクトリに移動します。

```bash
cd prac-python
```

### 3. 必要なパケージをインストールする

```bash
uv sync
```

仮想環境を有効化します。

- macOS/Linux:

  ```bash
  source venv/bin/activate
  ```

- Windows:

  ```bash
  venv\Scripts\activate
  ```

### 4. 仮想環境に入る

プロジェクトで必要な Python パッケージをインストールします。

```bash
pip install -r requirements.txt
```

### 5. モジュール検索パスを設定する

プロジェクトルートにある `common` パッケージを使用するために、`PYTHONPATH` を設定します。

- macOS/Linux:

  ```bash
  export PYTHONPATH=$(pwd):$PYTHONPATH
  ```

- Windows:

  ```bash
  set PYTHONPATH=%cd%;%PYTHONPATH%
  ```

### 6. スクリプトを実行する

以下のコマンドでスクリプトを実行し、ニューラルネットワークの学習を開始します。

```bash
python chapters/neural-network/basic-xor-nn.py
```

## プロジェクトの構成

```
your-repository/
├── chapters/
│   └── neural-network/
│       └── basic-xor-nn.py  # XOR 問題を解くニューラルネットワークの実装
├── common/
│   └── functions.py         # シグモイド関数などの共通関数
├── requirements.txt         # 必要な Python パッケージ
└── README.md                # このファイル
```

## 注意事項

- 学習の進捗はコンソールに出力されます。
- 学習後の結果はスクリプトの最後に表示されます。

## ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。詳細は [LICENSE](LICENSE) ファイルをご覧ください。
