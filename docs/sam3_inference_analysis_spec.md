# SAM3 推論・解析分離仕様

## 1. 目的とスコープ

既存の SAM3 食品インスタンスセグメンテーション処理を、推論データの生成・保存と、保存済みデータの評価・可視化に分離する。1回の推論実行で `results/run_XXXXX/` を作り、各画像の処理が終わるたびに解析に必要な情報を SQLite に保存する。解析は指定された run の SQLite を読み、指標・レポートを生成する。

- **AI開発:** はい。既存 SAM3 API を使った推論、評価、可視化が対象。問題種別はインスタンスセグメンテーション。
- **学習:** 対象外。独自学習・学習設定・学習ログ・学習曲線・Early Stopping・学習率スケジューラは追加しない。
- **フロントエンド:** No。
- **バックエンド:** 不要。
- **HTTP API:** 不要。
- **実行言語:** Python のローカル実行。run指定を含むCLI引数は使わず、run-id・解析runはコードまたは設定ファイルで設定する。
- **対象外:** SAM3 API自体の変更、データセットやアノテーションの変更、学習、外部サービス連携。

`no_bbox` は画像に存在するGTカテゴリのプロンプトのみを用いる。`bbox` はGT bboxとカテゴリを入力する条件付き評価で、通常の検出性能とは区別する。

## 2. 入力・設定

推論・解析の既定設定ファイルは `configs/settings_sam3.yaml`。相対パスはプロジェクトルート基準とする。少なくとも以下を使用する。

- COCOアノテーションと画像ディレクトリ、`include_crowd`、`image_ids`、`image_limit`
- 推論confidenceしきい値、プロンプトテンプレート・カテゴリ別上書き
- 実行モード、AP用IoUしきい値、レポート用IoUしきい値、P/R/F1用confidenceしきい値
- 結果ルート、グラフ・overlayの設定、日本語ラベル・フォント

設定ファイル内の値を記録して解析の再現性を保つ。解析時のAP/P/R/F1条件は推論時設定を既定とし、画像表示・ラベル等の設定には解析時設定を利用する。解析だけでSAM3を呼び出してはならない。

## 3. `main_infer.py`（推論）

### 3.1 エントリポイントとrun設定

推論エントリポイントは `src/main_infer.py` を実行する。設定ファイルは `configs/settings_sam3.yaml`をコードまたは既存起動構成で指定する。`main.py`は薄い`main_infer.py`互換ラッパーとして残し、従来起動経路から推論処理へ委譲する。

run-id、結果ルート、解析対象runはすべてコードまたはconfigで設定する。CLI引数は使用しない。run-idはコードまたはconfigで指定した値を使用し、新規runでは既存runと重複しないことを確認する。再実行時は既存runへ追記せず、必ず新しいrunを作る。

### 3.2 処理と保存

1. 設定とCOCOデータを読み込み、既存どおり画像選択とcrowd設定を適用する。
2. `Sam3Segmenter`（`src/sam3_inference/inference_categories.py`）で設定されたモードを推論する。`no_bbox` は画像内のGTカテゴリを重複なく使い、`bbox` は各GT bboxとカテゴリを対応させる。bboxモードは各GT bboxにつき最大1予測。モード・画像・カテゴリごとの予測上限は既存処理どおりconfidence上位100件。
3. COCO XYWH bboxを画像内にclipしたXYXYとして扱う。GT・予測maskは画像寸法を検証し、COCO圧縮RLEにする。
4. **各画像について、選択された全モードの推論が終わった時点で、その画像の情報をSQLiteの1トランザクションで書き込みcommitする。** 全画像の推論が終わるまで予測をメモリに貯めて一括保存する方式にはしない。画像単位で途中までの結果を保持し、異常終了時にもcommit済み画像を失わない。
5. `run_info.state`は正常完了時に完了、異常終了時に不完全として記録する。途中で異常終了したrunは再実行可能な状態として再開せず、不完全runとして残す。同一runへの追記は行わず、再実行は新しいrunとして行う。
6. 画像を読み込めない場合は既存挙動に合わせて警告してスキップする。成功数・スキップ数を実行情報に記録する。スキップ画像は評価のGT母集団に含めない。DB書き込みに失敗した場合、当該画像の部分レコードを残さない。
7. 推論中にはAP計算・metrics CSV・グラフ・overlayを生成しない。推論結果の保存後に別途 `analyze.py` を実行する。

### 3.3 SQLite保存仕様

既定保存先は `results/run_XXXXX/inference.sqlite3`。SQLiteを本プロジェクト内の推論－解析間データ契約とし、COCO compressed RLEの`counts`はASCII文字列を含むJSON文字列として格納する。各maskは少なくとも`size=[height,width]`と`counts`を持つ。bboxはJSON配列のXYXY。推論DBは解析に必要な全データを持ち、解析時に元COCO JSONからGTを再構成しない。

| テーブル | 主な列・内容 |
|---|---|
| `run_info` | `run_id`、スキーマ版、開始時刻、設定JSON（使用した閾値を含む）、アノテーション・画像ディレクトリ、モード、状態（完了／不完全）、実行デバイス、成功画像数、スキップ画像数 |
| `images` | `image_id`（主キー）、画像パス、幅、高さ、処理状態、画像処理時刻 |
| `categories` | `category_id`（主キー）、カテゴリ名 |
| `ground_truths` | GT/annotation ID、`image_id`、`category_id`、bbox XYXY、COCO圧縮RLE mask、必要に応じ`iscrowd` |
| `predictions` | 予測ID、`image_id`、mode、`category_id`・カテゴリ名、prompt、confidence、bbox XYXY、入力元GT bbox（なければNULL）、COCO圧縮RLE mask |

`run_info`の設定スナップショットは少なくともSAM3 confidence threshold、P/R/F1 confidence threshold、IoU/report thresholds、crowd・画像選択条件を復元可能にする。各画像の`processed_at`はその画像について推論・DB保存する際の時刻（ISO 8601）を記録する。未処理画像の行やスキーマ版の具体表現は実装時に決定するが、推論済み画像・GT・予測の関連とmodeを一意に復元できること。

出力例:

```text
results/run_00001/
  inference.sqlite3
  run_info.json
  analysis/                 # analyze.pyが生成
```

## 4. `analyze.py`（解析・可視化）

### 4.1 エントリポイント・run選択

解析エントリポイントは `src/analyze.py` を実行する。解析対象runはコードまたはconfigで設定し、解析時にCLI引数から変更しない。指定runは`results/run_XXXXX/`内の既存DBを特定する。最新runの暗黙選択は行わない。

### 4.2 読み込みと出力

指定run内の`inference.sqlite3`、設定スナップショット、予測・GTの圧縮RLEを読み、`src/sam3_inference/evaluation.py`相当の既存指標処理を適用する。画像overlayには記録された画像パスの画像を使うため、当該画像にアクセスできない場合はoverlayを生成できない旨を通知する。DBの不足・破損・非対応スキーマは明示的にエラーにする。

出力は `results/run_XXXXX/analysis/` 以下とし、既存の評価内容を引き継ぐ。再解析時はこの同じ`analysis/`配下を再生成・上書きし、過去解析版を別ディレクトリに保持しない。

```text
analysis/
  comparison_summary.png
  no_bbox/
    metrics.csv
    category_metrics_mask_001.png
    category_metrics_bbox_001.png
    overlays/...
  bbox/
    metrics.csv
    category_metrics_mask_001.png
    category_metrics_bbox_001.png
    overlays/...
```

- モードごとのmetrics CSVに、bbox/mask別、カテゴリ別と全体のAP（各IoU thresholdおよびAP50–95/AP50/AP75）・P/R/F1（各report threshold）を出力する。CSVはUTF-8 BOM付き、日本語列見出し。カテゴリ名は`output.label_ja`を優先する。全体値はGTが1件以上あるカテゴリのmacro平均とし、GTなしカテゴリを除外する。
- APはconfidence順位を用い、P/R/F1用confidence thresholdで予測を切り捨てない。APはCOCO互換101点補間。P/R/F1は指定confidence以上を対象とし、画像内・カテゴリ一致・confidence降順のgreedy 1対1対応でbbox IoU/mask IoUを別々に判定する。
- `comparison_summary.png`は既存の2×2比較（mask AP、bbox AP、mask全体P/R/F1、bbox全体P/R/F1）。各パネルに設定された全閾値を表示し、bbox条件付き評価とP/R/F1 confidence thresholdを明記する。
- カテゴリグラフはmask/bboxを別ファイルにし、各画像最大20カテゴリ、同じページ番号・カテゴリ範囲、AP50–95/AP50/AP75とP/R/F1の棒グラフを維持する。ファイル名は`category_metrics_mask_001.png`／`category_metrics_bbox_001.png`形式。
- overlayは既存設定のサンプル数・worst/best選択に従い、(a)生画像、(b) bboxのみ、(c) maskのみの3パネルを配置し、TP/FP/FNを識別可能にする。日本語フォントのグリフを確認し、利用可能なフォントがない場合は画像生成失敗を通知する。`output.dpi`と`output.label_ja`を反映する。
- SQLiteが予測の永続的な正本であり、`predictions.json`互換出力は行わない。

### 4.3 既存の評価条件

既存設定の値を引き継ぐ。現行`settings_sam3.yaml`のIoU thresholdsは0.50〜0.95の0.05刻み、report thresholdsは`[0.5, 0.6, 0.7, 0.8, 0.9]`、P/R/F1 confidence thresholdは0.5。設定変更時はDBに保存した実値を解析に使い、現在の設定値で過去runの評価条件を黙って置き換えない。

## 5. 既存ファイルへの変更

| パス | 変更案 |
|---|---|
| `src/main_infer.py` | 設定ロード、画像選択、SAM3推論、画像単位のSQLite保存、実行情報を担当。評価・レポート処理を持たない。 |
| `src/analyze.py` | 新規。指定runのDB読み込み、指標計算、CSV・画像成果物の生成を担当。SAM3を初期化しない。 |
| `src/main.py` | `main_infer.py`への薄い互換ラッパーとして残し、従来の起動方法を壊さず推論処理へ委譲する。 |
| `src/sam3_inference/inference_categories.py` | 既存のSAM3推論APIを再利用。分離目的で必要な最小限の調整に留める。 |
| `src/sam3_inference/evaluation.py` | 解析で再利用。DBから復元したRLEと同じ評価意味を保つ。 |
| `src/sam3_inference/reporting.py` | 解析側から再利用し、出力先を`analysis/`にする。 |
| `configs/settings_sam3.yaml` | 既存設定を維持。必要なら結果DBのファイル名等を追記するが、合意なしに既存値・評価条件を変更しない。 |
| `results/run_XXXXX/` | 推論DB・run情報と解析成果物の共通単位。画像単位commitにより、解析は完了済み画像分のDBを対象にする。 |

`docs/contracts/`は現時点で存在せず、外部プロジェクト間のAPI/DB契約も確認できない。HTTP APIはなく、上記SQLiteは本リポジトリ内の推論・解析間契約のため、新たな外部契約文書は作成しない。将来、外部利用者がSQLite形式に依存することが合意された場合は、既存方針に沿って契約仕様を別途定義する。

## 6. 制約・互換性

- SAM3は既存の`Sam3Segmenter`/公式SAM3推論APIを利用し、独自モデル学習を加えない。
- 既存の`no_bbox`/`bbox`条件、カテゴリ選択、GT bbox入力、100件上限、指標・可視化仕様を維持する。実際に使った設定値をDBに保存する。
- 推論と解析は独立して再実行可能とし、解析中にSAM3推論を再実行しない。
- 推論成果物と解析成果物を同じrunに紐付ける。DBの書き込みは画像単位でcommitする。
- この分離によりSQLiteファイルが追加される。RLEは圧縮形式だが、画像数・mask数に応じたDB容量増加がある。

## 7. 受け入れ条件

1. Pythonで推論と解析を別々に実行でき、CLI引数を使わず、run-id・解析runをコードまたはconfigで設定できる。フロントエンド・HTTP API・バックエンドを追加しない。
2. 推論結果は未使用の`results/run_XXXXX/`に保存され、再実行は新しいrunになる。
3. 1画像分の全mode推論後にその画像のGT・予測・メタデータをSQLiteへcommitし、全画像完了後の一括保存を行わない。
4. DBから画像ID/パス/幅/高さ、処理時刻、mode、カテゴリ、confidence、予測bbox・mask、正解bbox・mask、使用thresholdを復元できる。maskはCOCO圧縮RLEである。
5. 画像を処理した後、DBにその画像に対応する全GTと全modeの予測があり、画像単位のDB transactionが部分書き込みを残さない。
6. `no_bbox`と`bbox`の既存推論条件、confidence順位の100件上限およびGT bbox条件付き評価の意味を維持する。
7. `analyze.py`は指定runのSQLiteだけを読み、元アノテーションを使った再推論なしで解析できる。
8. 解析側が既存のbbox/mask AP、P/R/F1、カテゴリ別・全体集計を再現し、GTなしカテゴリの全体macro平均除外を維持する。
9. 解析側がmode別metrics CSV、条件比較summary、カテゴリ別mask/bboxグラフ、3パネルoverlayを出力し、既存の閾値・サンプル・日本語フォント要件を維持する。
10. 推論側でAP/metrics/可視化を行わず、解析側でSAM3を初期化しない。
11. run不在・DB破損・非対応スキーマを明確に通知する。画像欠落時はスキップ数を記録し、スキップ画像をGT母集団から除外し、overlay不能を明確に扱う。
12. `main.py`は従来の起動経路を維持する薄い`main_infer.py`互換ラッパーである。
13. 解析再実行時は指定run配下の`analysis/`成果物を再生成・上書きする。
14. 異常終了runは不完全として記録される。画像単位commit済みデータは保持されるが、同一runへ追記せず再実行は新runになる。
15. `inference.sqlite3`が予測・推論成果物の正本であり、`predictions.json`互換出力を行わない。

## 8. 未解決事項

なし。
