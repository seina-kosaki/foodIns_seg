# SAM3 食事データ評価仕様

## 1. 目的

既存の SAM3 Segmenter を用い、食事データのインスタンスセグメンテーションをローカル Python 環境で評価する。

- 評価モード: `no_bbox`（カテゴリプロンプトのみ）と `bbox`（GT bbox + カテゴリ）。
- 出力: bbox AP/mask AP、カテゴリ別・全体指標、Precision/Recall/F1、条件別の詳細成果物、条件比較画像。
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
| 評価閾値とモード | `evaluation.modes`、`evaluation.iou_thresholds`、`evaluation.report_thresholds`、`evaluation.pr_confidence_threshold` |

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
- 既存推論器の規則に従い、各 GT bbox から最大1件の予測を得る。
- 各 GT bboxから得た予測を、画像・カテゴリごとにまとめる。
- 同一画像・カテゴリに予測が100件を超える場合はconfidence上位100件へ制限する。
- 正解情報を与える条件付き評価であり、通常の検出性能としては扱わない。

両モードは同一の対象画像・設定・SAM3モデルで実行する。

## 5. 推論と予測上限

- 推論器の confidence threshold の既定値は `0.0` とし、設定可能とする。
- SAM3推論は画像とモードごとに1回行う。
- 推論結果をメモリ上で保持し、APとP/R/F1は同一予測結果から算出する。
- 各モードとも、画像・カテゴリごとに confidence 上位100件までを評価候補とする。
- no_bboxでは候補に対して上記の上位100件制限を直接適用する。
- bboxではまず各 GT bboxから最大1件を得て、その後、同一画像・カテゴリの予測が100件を超える場合に上位100件へ制限する。
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
- `other ingredients` を含む全カテゴリを通常どおり集計し、カテゴリの除外設定や補足集計は設けない。

## 7. 出力成果物

実行ごとに未使用の連番の親ディレクトリ `results/run_00001`、`run_00002`…を作成する。各評価条件の詳細成果物は条件別ディレクトリに分け、条件間比較の成果物は親ディレクトリに置く。

```text
results/run_00001/
  run_info.json
  comparison_summary.png
  no_bbox/
    metrics.csv
    predictions.json
    category_metrics_mask_001.png
    category_metrics_bbox_001.png
    category_metrics_mask_002.png
    category_metrics_bbox_002.png
    ...
    overlays/...
  bbox/
    metrics.csv
    predictions.json
    category_metrics_mask_001.png
    category_metrics_bbox_001.png
    category_metrics_mask_002.png
    category_metrics_bbox_002.png
    ...
    overlays/...
```

- 各条件ディレクトリには、その条件の指標CSV、永続予測JSON、overlay画像、カテゴリ別グラフ画像を保存する。条件名は `no_bbox` または `bbox` とする。
- `run_info.json` は親ディレクトリに置き、入力パス、実行モード、対象・処理画像数、予測数、推論confidence threshold、P/R/F1 confidence threshold、実行デバイス等を記録する。
- 親ディレクトリの `comparison_summary.png` は単一画像として、マスクAP、bbox AP、マスクの全体P/R/F1、bboxの全体P/R/F1を読みやすく配置する2×2構成を基本とする。
  - マスクAPパネル: x軸を `evaluation.iou_thresholds` の各値、y軸を該当IoU閾値ごとのマスクAPとし、`no_bbox` と `bbox` を線の色で比較する折れ線グラフとする。
  - bbox APパネル: 同様にx軸を `evaluation.iou_thresholds` の各値、y軸を該当IoU閾値ごとのbbox APとし、`no_bbox` と `bbox` を線の色で比較する折れ線グラフとする。
  - AP50-95/AP50/AP75の棒グラフは置換し、設定された `evaluation.iou_thresholds` の全点を上記の折れ線で表示する。設定に含まれる閾値を軸目盛または軸ラベルで確認できる形にする。
  - bboxモードの結果はGT bboxを与える条件付き評価である旨を図中に注記する。
  - マスクの全体P/R/F1パネルとbboxの全体P/R/F1パネルを別パネルとして配置し、各パネルで `evaluation.report_thresholds` をx軸に、該当閾値でのPrecision、Recall、F1をモード別の折れ線で比較する。
  - 各P/R/F1パネルで、P、R、F1を凡例または小パネルで識別できるようにし、`no_bbox` と `bbox` の線も凡例で区別する。各閾値点を目盛またはマーカーで示し、縦軸ラベルと目盛値を表示する。
  - 図中に実行時の `pr_confidence_threshold` の実際の値を明記する。`evaluation.report_thresholds` の全点も軸目盛または軸ラベルから確認できる形にする。
- 全カテゴリを一枚に詰めた表画像および数百行規模の全体サマリー表画像は出力しない。
- 指標CSV:
  - 条件ごとにCSVを保存し、全体・全カテゴリの評価対象・集計区分・評価閾値を識別できる形で、各`evaluation.iou_thresholds`ごとのAP、および各`report_thresholds`ごとの適合率・再現率・F1を正確な数値で保持する。
  - CSVには既存の集計仕様どおりbboxとマスクを区別し、各閾値、カテゴリ名、全体/カテゴリ別の数値を保持する。
  - CSVには既存の集計仕様上の要約値としてAP50–95、AP50、AP75も必要に応じて保持できるが、グラフの折れ線表示をそれら3値のみに置換しない。
  - 補足集計の区分・数値は出力しない。
  - 列見出しは日本語にする。
  - 評価対象の値は「bbox」「マスク」、集計区分の値は「全体」「カテゴリ別」とする。
  - カテゴリ名は `output.label_ja` を優先し、未設定カテゴリは英語名を使用する。
  - ファイルはUTF-8 BOM付きで保存する。
- カテゴリ別グラフ:
  - 条件別ディレクトリに保存し、mask用とbbox用の画像をそれぞれ別ファイルとして出力する。1枚の画像にmaskとbboxを混在させない。
  - カテゴリは同じ並び・カテゴリページ範囲で分割し、一画像あたりの表示カテゴリ数は最大20を維持する。mask用とbbox用は同じページ番号と同じカテゴリページ範囲を使用する。
  - ファイル名は `category_metrics_mask_001.png`、`category_metrics_bbox_001.png` のように形状を明示し、ページ番号を3桁連番（`001`、`002`、…）とする。
  - mask用画像にはmask APグラフとmaskのP/R/F1グラフのみを含める。bbox用画像にはbbox APグラフとbboxのP/R/F1グラフのみを含める。
  - カテゴリ別マスクAP/bbox APパネルでは、該当形状のAP50-95、AP50、AP75をカテゴリごとの棒グラフとして既存表示を維持する。
  - カテゴリ別P/R/F1パネルでは、該当形状の各`report_thresholds`の適合率・再現率・F1を棒グラフとして維持する。
  - カテゴリ別グラフを折れ線グラフへ変更せず、画像あたり最大20カテゴリの分割を維持する。
  - P/R/F1ではPrecision、Recall、F1を高コントラストかつ相互に識別可能な配色とする。`report_thresholds`が2つある場合、同一指標の各thresholdは同一色を使わず、同系色の濃淡または同等に識別可能な区別方法とする。
  - P/R/F1凡例には指標名（Precision、Recall、F1）とthreshold値の両方を含める。凡例だけで指標とthresholdを特定できるようにする。
  - 上記グラフは正確な全カテゴリ数値の参照元としてCSVを併用し、数値表示をグラフのみに頼らない。
  - `output.dpi`、日本語ラベル、`output.font_path`を利用する。
- overlay画像:
  - 条件別ディレクトリ内に保存する。
  - overlayのみディスクへ保存し、mask自体は保存しない。
  - 各overlayサンプルは`条件別overlays/`内の1枚の複合画像とし、横に次の3パネルを配置する:
    - (a)生画像。
    - (b)生画像上にGT/予測bboxだけを重ねた画像（maskは重ねない）。
    - (c)生画像上にGT/予測maskだけを重ねた画像（bboxは重ねない）。
  - 各パネルに明確なpanel title（例:「(a) 生画像」「(b) bbox」「(c) mask」）を付ける。
  - bboxパネルとmaskパネルではそれぞれTP/FP/FNを識別可能な色で表示する。
  - 各パネルでTP、FP、FNを区別する色と凡例またはラベルを保つ。
  - 生画像パネルには判定結果のbbox/maskを重ねず、bboxパネルとmaskパネルの役割を別々に判読できる構成とする。
  - 一枚の複合画像内に単一のcombined mask+bbox表現を使わず、上記の3パネル構成に置き換える。
  - 保存前に `output.label_ja` とカテゴリ名を含む必要グリフが表示可能な日本語フォントであることを確認する。
  - 表示可能なフォントが見つからない場合は実行を失敗として通知する。

## 8. 可視化サンプル

- 可視化サンプルは合計20枚とする。
- mask IoUが低い事例から10件（worst）、mask IoUが高い事例から10件（best）を選ぶ。
- 各サンプルは条件別の`overlays/`内に1枚の複合画像として保存し、横に(a)生画像、(b)GT/予測bboxのみ、(c)GT/予測maskのみの3パネルを配置する。
- bboxパネルとmaskパネルの両方でTP/FP/FNを識別可能な色と凡例またはラベルで表示する。
- 各パネルにpanel titleを付け、bboxとmaskを別々に判読できるようにする。
- 単一パネル内にbboxとmaskをまとめて重ねるcombined表現は使用しない。
- 日本語ラベルは `label_ja` を使用し、フォントは `font_path` を優先する。フォント未指定時は利用可能な日本語フォントを検出する。
- 日本語フォントのグリフを検査し、必要な日本語を表示できない場合は失敗として通知する。
- overlay画像は`output.dpi`を適用する。

## 9. 制約

- データ、設定、モデル以外の外部サービスやAPIを使用しない。
- AP算出時はP/R/F1用thresholdで予測を切り捨てない。
- P/R/F1算出時はconfidence 0.5以上の予測だけを使用する。
- maskを永続予測ファイルへ含めず、評価時と可視化時に必要な範囲でメモリ上に保持する。
- カテゴリ除外および補足集計は行わず、`other ingredients`を含む全カテゴリを通常集計に含める。
- `bbox` モードは条件付き評価として明示する。
- 日本語フォントのグリフ確認に失敗した場合は画像成果物の生成を失敗として扱い、実行へ通知する。

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
10. 各モードで画像・カテゴリごとconfidence上位100件に制限できる。bboxモードは各GT bboxから最大1件を得た後、同一画像・カテゴリが100件を超える場合に上位100件へ制限できる。
11. maskをディスクへ保存せず、永続予測JSONへmaskを含めない。
12. `other ingredients`を含む全カテゴリを通常集計に含め、カテゴリ除外設定および補足集計を設けない。
13. 条件ごとの親ディレクトリ配下に `no_bbox/` と `bbox/` を作り、各条件のCSV、予測JSON、overlay、カテゴリ別グラフを対応するディレクトリへ保存できる。条件比較成果物と `run_info.json` は親ディレクトリに置く。
14. `comparison_summary.png`を単一の統合画像として出力し、マスクAPとbbox APを別パネルの折れ線グラフとし、x軸に`evaluation.iou_thresholds`の全点、y軸に各閾値AP、線の色で`no_bbox`と`bbox`を比較できる。
    全体P/R/F1はマスクとbboxを別パネルとし、x軸に`evaluation.report_thresholds`の全点、y軸にPrecision/Recall/F1、モード別の折れ線で比較できる。閾値点と凡例を判読可能にし、`pr_confidence_threshold`の実値とbboxありはGT bboxを与える条件付き評価である旨を図中に明記できる。
15. 全カテゴリを一枚に詰めた表画像と全体サマリー表画像を出力せず、全カテゴリの正確な指標値を条件別CSVに保持できる。
16. カテゴリ別グラフはmask用とbbox用を別画像として出力し、ファイル名は`category_metrics_mask_001.png`と`category_metrics_bbox_001.png`の形式、ページ番号は3桁連番とする。mask用とbbox用は同じカテゴリページ範囲を同じページ番号で分割し、一画像あたり最大20カテゴリを維持する。
    mask用画像はmaskのAPグラフとP/R/F1グラフのみ、bbox用画像はbboxのAPグラフとP/R/F1グラフのみを含む。両形状を同一画像へ混在させず、カテゴリ別グラフを折れ線へ変更しない。
17. カテゴリ別P/R/F1はPrecision、Recall、F1を高コントラストかつ相互に識別可能な配色で表示し、同一指標の2つの`report_thresholds`は同系色の濃淡などで区別する。凡例は指標名とthreshold値の両方を特定できる表記とする。
18. CSVをUTF-8 BOM付きで出力し、列見出し、評価対象、集計区分を日本語化できる。カテゴリ名は`output.label_ja`を優先し、画像成果物に日本語ラベル・設定フォントを適用できる。
19. mask IoUのworst/best各10件、合計20件のoverlayサンプルを出力できる。
20. 各overlayサンプルを横に(a)生画像、(b)GT/予測bboxのみ、(c)GT/予測maskのみの3パネルで構成し、bboxパネルとmaskパネルそれぞれでTP/FP/FNを識別できる。各パネルにtitleと凡例またはラベルを付ける。
21. 条件別指標CSVに各`report_thresholds`ごとの適合率・再現率・F1、およびAP50–95/AP50/AP75を出力できる。
22. 日本語フォントのグリフを確認でき、表示可能なフォントがない場合は失敗を通知できる。

## 11. 未解決事項

なし。
