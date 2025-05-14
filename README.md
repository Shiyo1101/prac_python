# Python 学習 〜CNN まで理解しよう〜

## 必要条件

このプロジェクトを実行するには、以下のソフトウェアが必要です:

- [uv（Python パッケージ管理ツール）](https://docs.astral.sh/uv/guides/install-python/)
- [Git](https://git-scm.com/)

## 注意事項

windows の場合、WSL 環境での実行を推奨します。

## セットアップ手順

以下の手順にしたがって、このプロジェクトをセットアップしてください。

### 1. リポジトリをクローンする

まず、Git を使用してこのリポジトリをローカル環境にクローンします。

```bash
git clone https://github.com/Shiyo1101/practice_python.git
```

### 2. プロジェクトディレクトリに移動する

クローンしたリポジトリのディレクトリに移動します。

```bash
cd practice_python
```

### 3. 必要なパケージを同期する

```bash
uv sync
```

### 4. 仮想環境に入る

- macOS/Linux:

  ```bash
  . .venv/bin/activate
  ```

- Windows:

  ```bash
  .venv\Scripts\activate
  ```

### 5. モジュール検索パスを設定する

プロジェクトルートにある `common` パッケージなどを使用するために、`PYTHONPATH` を設定します。

- macOS/Linux:

  ```bash
  export PYTHONPATH=$(pwd):$PYTHONPATH
  ```

- Windows (PowerShell):

  ```powershell
  $env:PYTHONPATH="$PWD;$env:PYTHONPATH"
  ```

- Windows (コマンドプロンプト):

  ```cmd
  set PYTHONPATH=%cd%;%PYTHONPATH%
  ```

### 6. python ファイルを実行する

```bash
uv run main.py
```

## プロジェクトの構成

```
practice-python/
├── chapters/
│   └── neural-network/      # ニューラルネットワークの勉強
│       └── basic-xor-nn.py
│   └── numpy/               # numpyの勉強
│   └── opencv/              # opencvの勉強
├── common/
│   └── functions.py         # シグモイド関数などの共通関数
├── images/                  # mnistデータセット
├── images/                  # 画像
├── models/                  # 作成したモデル
```

## （おまけ）uv の基本コマンド説明

`uv` は Python プロジェクトの依存関係管理やスクリプト実行を簡単にするツールです。以下に、よく使われる基本コマンドを説明します。
詳細な説明は[こちら](https://zenn.dev/turing_motors/articles/594fbef42a36ee)から

### 1. `uv sync`

プロジェクトの依存関係をインストールまたは更新します。`requirements.txt` や `pyproject.toml` に基づいて必要なパッケージをインストールします。

```bash
uv sync
```

### 2. `uv run <script>`

指定した Python スクリプトを実行します。仮想環境が自動的に有効化され、`PYTHONPATH` も適切に設定されます。

```bash
uv run main.py
```

### 3. `uv add <package>`

プロジェクトにパッケージの追加を行うことができます。

```bash
uv add numpy
```

バージョンを指定することも可能です

```bash
uv add "numpy==1.26.4"
```

### 4. `uv remove <package>`

プロジェクトからパッケージを取り除くことができます。

```bash
uv remove numpy
```

### 5. `uv venv`

仮想環境を作成します。`uv` はプロジェクトごとに仮想環境を管理します。

```bash
uv venv
```

### 6. `uv which`

仮想環境内の Python 実行ファイルのパスを表示します。

```bash
uv which python
```

### 7. `uv clean`

仮想環境やキャッシュを削除して、プロジェクトをクリーンな状態に戻します。

```bash
uv clean
```

### 8. `uv doctor`

プロジェクトの設定や依存関係に問題がないかを診断します。

```bash
uv doctor
```
