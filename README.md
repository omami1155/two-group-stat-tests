# streamlit-stat-tests

CSVをアップロードして、以下の5検定を一括実行するStreamlitアプリです。

- Shapiro-Wilk検定
- Levene検定
- Student t-test
- Welch t-test
- Mann-Whitney U検定

## 対応CSV形式

```csv
group_A,group_B
12.3,10.2
11.8,9.8
13.1,10.9
,11.1
```

- 2列以上のCSVを読み込み、比較したい2列をアプリ上で選択します。
- 数値に変換できない値は自動的に除外されます。

## ローカル実行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## GitHub → Streamlit Community Cloud 公開手順

1. このフォルダー一式をGitHubリポジトリにアップロード
2. Streamlit Community Cloud にGitHub連携
3. 対象リポジトリを選択
4. Main file path を `app.py` に設定
5. Deploy

## ファイル構成

```text
streamlit-stat-tests/
├── app.py
├── requirements.txt
└── README.md
```
