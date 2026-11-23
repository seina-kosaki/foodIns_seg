# `src/analyze.py` 解析結果可視化仕様

## 1. 目的・範囲

保存済みSAM3推論runの評価結果を、条件別に確認し、`no_bbox` と `bbox` の比較を可視化する。入力となる推論DBを変更せず、指定runの解析成果物だけを生成する。

- **AI開発:** はい。既存SAM3 APIによるインスタンスセグメンテーションの推論結果を評価・可視化する。学習は対象外。
- **フロントエンド:** No。
- **バックエンド/API:** 不要。HTTP API、DBサービスを追加しない。
- **実行言語:** Pythonのローカルスクリプト。
- **対象外:** SAM3の再推論、独自学習、推論DBスキーマの変更、外部サービス連携。

`bbox` はGT bboxを与える条件付き評価であり、`no_bbox` の通常のプロンプト推論と区別して表示する。

## 2. 入出力・設定

### 入力

- 設定: `configs/settings_sam3.yaml`。解析対象runは `analysis_run_id` で明示する。
- DB: `results/{analysis_run_id}/inference.sqlite3`（実際の親ディレクトリは `output.results_dir` に従う）。
- GT、予測、カテゴリ、画像情報および評価条件: DB内の推論結果と設定スナップショットを使う。解析時に元アノテーションから再推論・再構成しない。
- しきい値: IoU/AP用 `evaluation.iou_thresholds`、表示用 `evaluation.report_thresholds`、P/R/F1用 `evaluation.pr_confidence_threshold` は、推論時DBの設定スナップショットを使用する。report thresholdsはIoU thresholdsに含まれる値とする。表示用の `output` 設定（ラベル、フォント、dpi、サンプル数等）は既存設定ファイルに従う。

現行設定値はIoU thresholdsが0.50～0.95の0.05刻み、report thresholdsが`[0.5, 0.6, 0.7, 0.8, 0.9]`、P/R/F1 confidence thresholdが0.5。仕様上は設定可能値を使い、これらの値を固定値として扱わない。

### 出力

`results/{run_id}/analysis/` 以下に生成する。解析再実行時は既存 `analysis/` を置き換え、旧成果物を別版として残さない。画像ファイルが利用できない場合はoverlayを生成できない旨を通知する。DB不在・破損・非対応スキーマ・不正なしきい値は明確なエラーとする。

## 3. 表示要件

### 3.1 条件別成果物（`no_bbox` / `bbox`）

各実行モードのディレクトリに、その条件だけの `metrics.csv`、カテゴリ別グラフ、overlayを出力する。CSVはUTF-8 BOM付きとし、bbox/mask別、カテゴリ別・全体のAP（各IoU閾値およびAP50–95/AP50/AP75）と、各report thresholdでのTP/FP/FN・適合率・再現率・F1を含める。全体値はGTがあるカテゴリのmacro平均（カテゴリごとのAP平均、mAP）とし、GTなしカテゴリを除外する。カテゴリ数が不均衡でも各カテゴリを等しい重みで評価できるため、macro平均には意味がある。補助として全GTの同一重みマイクロ平均も出力する。カテゴリ名は`output.label_ja`を優先する。

条件別カテゴリグラフはmask用とbbox用を別画像にする。既存どおり、各画像あたり最大20カテゴリ、同じページ番号・カテゴリ範囲で分割し、`category_metrics_mask_001.png`／`category_metrics_bbox_001.png`形式とする。左パネルにAP50–95/AP50/AP75の横棒、右パネルにカテゴリ×（適合率・再現率・F1 × 各report threshold）のヒートマップを示す。ヒートマップの各セルに数値を表示し、凡例と棒グラフの重なりをなくす。

### 3.2 Summaryとcomparison（A/B比較）

run直下の既存 `comparison_summary.png` を、既存の2×2構成で生成する。

1. mask AP対IoU threshold
2. bbox AP対IoU threshold
3. mask全体macro P/R/F1対report threshold
4. bbox全体macro P/R/F1対report threshold

両モードがrunにある場合、グラフ上で `no_bbox` と `bbox` を識別してA/B比較する。各軸に設定済みの全thresholdを表示し、P/R/F1 confidence thresholdの実値と、bboxがGT bbox入力の条件付き評価であることを注記する。片方のmodeだけのrunでは、存在するmodeの値だけを出し、存在しないmodeの値を作らない。

### 3.3 Overlays

各modeの `overlays/` に既存の3パネル画像（生画像、bboxのみ、maskのみ）を出力し、TP/FP/FNを識別可能にする。生画像には判定結果を重ねない。日本語カテゴリラベル・`output.dpi`・フォント設定を反映し、必要な日本語グリフを表示できるフォントがない場合は生成失敗を通知する。

サンプル総数、worst/best選択は `output.visualization_samples` と `output.worst_best_samples` に従う。既存実装の選択基準（mask IoU、confidence閾値による対応付け）を維持する。現行設定はサンプル合計20、worst/best上限10であり、mode間に分配する。

### 3.4 per-GT・遷移・判定テキスト
GTインスタンス単位の結果を `per_gt.csv` に出力する。各行は`annotation_id`、画像・カテゴリ、サイズ区分、密集度、bbox/maskの最良IoU、マスク判定状態、成功率@0.50/@0.75を含む。bbox modeでは入力GT bbox由来の予測との入力元IoUも記録する。

両modeがある場合は`comparison_gt.csv`と`transition_matrix.csv`を生成し、成功/低IoU/見逃しの3×3遷移を比較する。遷移しきい値は既存YAMLの`comparison.comparison_iou_threshold`を使用する。散布図・遷移ヒートマップ・比較サマリも生成する。

判定テキストはマスク再現率差を優先し、次にbbox/mask AP差、適合率差、差なしの順で設定されたしきい値とともに自動生成する。文言は断定を避け「可能性がある」を含める。

### 3.5 既存画像の置換

再解析で `analysis/` を再生成し、同じ成果物パスを置き換える。A/B比較画像は既存名 `comparison_summary.png` を使用する。新たな並行版や別名画像は作成しない。

## 4. 実装責務

- `src/analyze.py`: 指定runとSQLiteを読み、DB内の評価条件で指標を集計し、条件別CSV・可視化を呼び出す。SAM3を初期化しない。
- `src/sam3_inference/evaluation.py`: 既存のbbox/mask指標・対応付けを再利用する。
- `src/sam3_inference/reporting.py`: 既存のCSV、summary、カテゴリグラフ、overlay出力を再利用・拡張する。
- Hydraは導入しない。既存のYAML設定方式を維持する。

## 5. 契約・制約

- 外部公開APIや外部プロジェクト間DB契約の変更は本仕様の対象外。解析出力は本リポジトリ内成果物であり、推論SQLiteの契約・スキーマは変更しない。
- 現在 `docs/contracts/` は存在しない。外部契約に変更が生じる場合は、その時点で既存の `docs/contracts/` 方針を確認し、該当契約文書を更新対象として別途合意する。現時点で同ディレクトリや契約文書を新設する要求はない。
- 対象データは既存runのDBと画像のみ。既存しきい値・評価意味・カテゴリ表示数（最大20カテゴリ/画像）を変更しない。

## 6. 受け入れ条件

1. CLI引数やフロントエンド/APIを追加せず、既存設定方式で指定runを解析できる。
2. 指標と評価条件は推論DBの設定スナップショットを使い、SAM3を再実行しない。
3. 条件別のCSV、mask/bbox別カテゴリグラフ、3パネルoverlayを生成し、既存カテゴリ表示数の上限20を維持する。
4. `comparison_summary.png`に既存2×2の全指標を出し、両modeが存在するときA/B比較できる。bbox条件付き評価とP/R/F1 confidence thresholdが識別できる。
5. サンプル選択、既存日本語ラベル・フォント・dpi設定を反映し、必要フォントがない場合や元画像がない場合に状態を通知する。
6. 再解析時に既存 `analysis/` と同名の成果物を置き換え、古い画像を残さない。
7. DB不在・破損・非対応スキーマ・不正なしきい値を明確に通知する。
8. per-GT、遷移、判定テキストの詳細を含め、合意した設定と成果物を満たす。

## 7. 未解決事項

現時点で未解決事項はない。
