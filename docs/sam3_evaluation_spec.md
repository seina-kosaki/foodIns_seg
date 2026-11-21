# SAM3 食事データ評価仕様

## 1. 目的

既存の SAM3 Segmenter を用い、食事データのインスタンスセグメンテーションをローカル Python 環境で評価する。

- 評価モード: `no_bbox`（カテゴリプロンプトのみ）と `bbox`（GT bbox + カテゴリ）。
- 出力: bbox AP/mask AP、カテゴリ別・全体指標、Precision/Recall/F1、可視化成果物。
- 学習は行わない。
- `bbox` は正解 bbox を与える条件付き評価であり、通常の物体検出器の検出性能とは別に扱う。

## 2. 対象フェーズと区分

- 対象フェーズ: AI評価のみ（推論・評価）。
- 問題種別: インスタンスセグメンテーション。
- モデル/API: 既存 SAM3 `Segmenter/API` を利用する。
- 独自学習: 行わない。
- 実行形態: Python ローカルスクリプト。
- フロントエンド: 不要。
- バックエンド/API: 不要。
- 学習固有の設定、ログ、Early Stopping、学習率スケジューラ、学習曲線は対象外。

## 3. 入力

設定は `configs/settings_sam3.yaml` を唯一の設定ソースとする。

| 入力 | パスまたは内容 |
|---|---|
| COCOアノテーション | `datasets/annotations/Test.json` |
| 評価画像 | `datasets/images/test` |
| 評価モード | `no_bbox`、`bbox` |
| プロンプトテンプレート | `prompt.template` |
| カテゴリ別プロンプト | `prompt.overrides` |
| 日本語ラベル・フォント | `output.label_ja`、`output.font_path` |

COCO形式の `images`、`annotations`、`categories` を使用する。

対象画像の選択は `dataset.image_ids` と `dataset.image_limit` に従う。`dataset.include_crowd` は `iscrowd=1` アノテーションの評価対象への含有/除外を制御する。

## 4. 推論条件

### 4.1 no_bbox

- 各画像で、その画像のアノテーションに含まれるカテゴリを重複なく抽出する。
- 抽出したカテゴリのみをプロンプトとして推論する。
- データセット全カテゴリを問い合わせない。
- 未注釈カテゴリへの誤検出と、提示カテゴリ外へのカテゴリ誤認は評価対象外とする。

### 4.2 bbox

- 各 GT アノテーションの COCO XYWH bbox を SAM3 の XYXY に変換する。
- 各 GT bbox と対応するカテゴリを一組として推論に入力する。
- 既存推論器の規則に従い、各 GT bbox に対して最大1件を予測する。
- 正解情報を与える条件付き評価であり、通常の検出性能としては扱わない。

両モードは同一の対象画像・設定・SAM3モデルで実行する。

## 5. 推論と予測上限

- 推論器の confidence threshold の既定値は `0.0` とし、設定可能とする。
- SAM3推論は画像とモードごとに1回行う。
- 推論結果をメモリ上で保持し、APとP/R/F1は同一予測結果から算出する。
- no_bboxでは、画像・カテゴリごとに confidence 上位100件までを評価候補とする。
- bbox条件では、各 GT bbox あたり最大1件とする。
- maskはディスクへ保存しない。評価と可視化に必要な間だけメモリで保持する。
- JSON等の永続予測データにはmaskを含めない。
- 永続予測レコードにはモード、画像ID、カテゴリ、プロンプト、confidence、予測bbox、必要に応じて入力元bboxを含める。

## 6. 評価方法

### 6.1 AP

- bbox APとmask APは別々に算出する。
- bboxは予測bboxとGT bboxのIoU、maskは予測maskとGT maskのIoUを使用する。
- IoU閾値は0.50〜0.95を0.05刻み（合計10点）とする。
- AP50–95は10閾値の平均。
- AP50はIoU 0.50、AP75はIoU 0.75。
- COCO互換の101点補間を使用する。
- confidenceで予測を切り捨てない。推論器thresholdによって得られた候補全体をAPへ使う。

### 6.2 Precision / Recall / F1

- P/R/F1評価用のconfidence threshold既定値は`0.5`とし、設定可能とする。
- candidateは0.5以上のみを対象とする。
- bboxとmaskを別々に算出する。
- 設定可能なIoU閾値ごとに出力する。
- 既定の個別表示閾値は0.50、0.75とする。

### 6.3 対応付け

- 対応付けは画像内かつカテゴリ一致を条件に行う。
- confidence降順で貪欲に1対1対応付けを行う。
- bbox AP/mask APおよびP/R/F1で、対応するbbox IoUまたはmask IoUが閾値以上ならTPとする。
- 同一予測を複数GTへ対応させない。
- 同一GTを複数予測へ再利用しない。
- 閾値に一致しない予測をFP、一致しないGTをFNとする。

### 6.4 集計

- カテゴリ別APとP/R/F1を出力する。
- 全体集計は、GTが1件以上存在するカテゴリだけを対象としたカテゴリ平均とする。
- GTのないカテゴリは全体平均から除外する。
- 通常集計では `supplementary_exclude_categories` の影響を受けない。
- 補足集計では除外カテゴリを除いた全体指標を併記する。

補足除外設定の現在値は `other ingredients` とする。このカテゴリは補足集計のみに除外し、通常集計には含める。

## 7. 出力成果物

実行ごとに未使用の連番ディレクトリ `results/run_00001`、`run_00002`…を作成する。

- `predictions.json`: 永続予測レコード。maskは含めない。
- `run_info.json`: 入力パス、モード、対象・処理画像数、予測数、推論confidence threshold、P/R/F1 confidence threshold、実行デバイス等。
- 指標CSV:
  - モード、bbox/mask、全体・カテゴリ別、AP50–95/AP50/AP75、P/R/F1等を保持する。
  - カテゴリ名は `label_ja` を優先し、未設定カテゴリは英語名を使用する。
  - ファイルはUTF-8 BOM付きで保存する。
- 表画像:
  - 全体指標、カテゴリ別指標、bbox有無比較を表またはグラフとして示す。
  - `output.dpi`、日本語ラベル、`output.font_path`を利用する。
- overlay画像:
  - GTと予測のmaskおよびbboxを重ねる。
  - TP、FP、FNを色分けし、凡例またはラベルで識別する。

## 8. 可視化サンプル

- 可視化サンプルは合計20枚とする。
- mask IoUが低い事例から10件（worst）、mask IoUが高い事例から10件（best）を選ぶ。
- 各サンプルでGT/予測のmaskとbboxを重ねる。
- TP/FP/FNをそれぞれ識別可能な色と凡例で表示する。
- 日本語ラベルは `label_ja` を使用し、フォントは `font_path` を優先する。フォント未指定時は利用可能な日本語フォントを検出する。
- overlay画像は`output.dpi`を適用する。

## 9. 制約

- データ、設定、モデル以外の外部サービスやAPIを使用しない。
- AP算出時はP/R/F1用thresholdで予測を切り捨てない。
- P/R/F1算出時はconfidence 0.5以上の予測だけを使用する。
- maskを永続予測ファイルへ含めず、評価時と可視化時に必要な範囲でメモリ上に保持する。
- 通常集計と補足集計を混在させない。
- `bbox` モードは条件付き評価として明示する。

## 10. 受け入れ条件

1. `configs/settings_sam3.yaml` のみを設定入力としてローカル実行できる。
2. `no_bbox` と `bbox` を実行し、両モードの指標を比較できる。
3. no_bboxでは各画像に注釈されたカテゴリのみを問い合わせる。
4. bboxでは各GT bboxとカテゴリを推論に渡し、各GTから最大1予測を得る。
5. 推論器confidence threshold既定値は0.0、P/R/F1のconfidence threshold既定値は0.5である。
6. SAM3推論をモード・画像ごとに1回ずつ実行し、同一予測からAPとP/R/F1を算出できる。
7. bbox AP/mask APをそれぞれCOCO互換101点補間で算出し、AP50–95/AP50/AP75を出力できる。
8. カテゴリ別指標を出力し、GTのないカテゴリを全体カテゴリ平均から除外できる。
9. confidence降順・カテゴリ一致・画像内・貪欲な1対1対応付けを適用できる。
10. no_bbox予測は画像・カテゴリごと上位100件、bbox予測は各GTあたり最大1件に制限できる。
11. maskをディスクへ保存せず、永続予測JSONへmaskを含めない。
12. 通常集計には`other ingredients`を含め、補足集計のみ除外できる。
13. CSVをUTF-8 BOM付きで出力し、表画像とoverlay画像に日本語ラベル・設定フォントを適用できる。
14. mask IoUのworst/best各10件、合計20件のoverlayサンプルを出力できる。
15. GT/予測のmaskとbboxを重ね、TP/FP/FNを識別できる。

## 11. 未解決事項

なし。
