# Python 構成・GUI 設計

## 設計原則

- 読込 (`muaxis.io`) は GUI・解析ライブラリに依存しない。バッチ処理、Notebook、将来の別アプリでも同じ API を使用する。
- 数値データは常に `(energy, y, x)` の float 配列で扱い、8-bit 表示画像を解析値の代用にしない。
- GUI は状態を直接計算しない。すべての操作は `AnalysisSession` に対する明示的な command とし、処理完了時だけ画面を更新する。
- 元の強度データは読み取り専用とし、OD、登録済み画像、ROI、マップは派生データ・操作履歴として保存する。

## パッケージ構成

```text
src/muaxis/
  io/                 # 完了: UVSOR .hdr/.xim/i0.txt/drift.txt の読込
    stxm.py
  domain/             # 次: ScanStack を含むセッション、ROI、来歴
  processing/         # 次: OD、位置合わせ、ROI、クラスタ、マップ
  gui/                # PySide6: 画面、ViewModel、Worker
  plotting/           # pyqtgraph の画像・スペクトル描画アダプタ
```

`read_stxm_scan(path)` はヘッダの `ImageNNN_0` 番号で `.xim` を決定するため、Finder 上の並び順には依存しない。`i0.txt` と `drift.txt` は任意で、行数・形状がフレーム数と一致しない場合は例外を出す。source current 補正は reader の責務ではなく、処理層でユーザーが明示的に有効化する。

## GUI全体方針

旧 Xojo の Window1 に多くの操作が集中していたため、Python 版では **MainWindow をデータ管理と共通表示のハブ**に限定し、解析機能はモジュール単位の独立ウインドウに分ける。各ウインドウは同じ `AnalysisSession` を参照するが、計算は `processing` のサービスを呼び出し、ウインドウ間で配列や状態を直接共有しない。

共通ルール:

- `MainWindow` は常時1つ。データセット、現在のフレーム、表示レイヤ、セッション保存を管理する。
- 解析ウインドウは複数開けるが、同じ種類は dataset ごとに1つ（またはタブ化）とする。
- 各ウインドウに「入力データ」「処理設定」「実行」「結果レイヤ」「エクスポート」を置く。
- 長時間処理は worker で実行し、進捗・キャンセル・警告を共通 status bar に返す。
- 解析結果は MainWindow のレイヤ一覧へ登録し、他モジュールの入力に選択できる。

## ウインドウ構成

| Python ウインドウ | 対応する旧 Window | 責務 |
| --- | --- | --- |
| `MainWindow` | Window1 | ファイルを開く、scan table、画像/スペクトルの共通表示、表示レイヤ、session保存 |
| `RegistrationWindow` | Window2 | 基準フレーム、POC/相互相関/手動シフト、品質指標、補正適用 |
| `MultivariateWindow` | 新設（Window5/6の一部を整理） | SVD/PCA、score/loading、寄与率、低ランク再構成、ノイズ/残差 map、相候補 |
| `SegmentationWindow` | Window5 | ヒストグラム、閾値、形態学処理、連結成分、クラスタラベルと領域表 |
| `FittingWindow` | Window2/4/6 の参照処理を再編 | 参照スペクトル、R²照合、NNLS、非線形ピークフィット、残差/信頼度 |
| `PeakMapWindow` | Window7 | ピーク feature、事前相分布、RGB/多チャネルマップ、統計 |
| `PhaseMapWindow` | Window4 | 相候補/相同定、unknown/mixed、相別固定色、オーバーレイ |
| `ExportWindow` | 各 Window の保存処理 | 画像、CSV、スペクトル、session、レポートを一括出力 |

旧 Window6 に散在していた表示画像・LUT・閾値操作は `MainWindow` の Image Display dock と `PeakMapWindow` / `PhaseMapWindow` へ移し、画像表示と解析設定を分離する。旧 Window4 の map 計算は `PeakMapWindow` と `PhaseMapWindow` に分ける。

## MainWindow

```text
┌ Toolbar: Open | Save session | Export | Undo/Redo ────────────────────────┐
├ Scan table ──────┬ Image workspace ─────────────────┬ Analysis dock ─────┤
│ energy / time    │ [Raw] [I0] [OD] [Registered]      │ I0 / OD settings   │
│ current / status │                                      │ named ROI list      │
│ frame selection  │ image + LUT + histogram + scale bar │ registration action │
│                  │ energy slider / frame playback       │ export action       │
├──────────────────┴──────────────────────────────────────┴─────────────────┤
│ Spectrum workspace: ROI / selected-pixel / reference spectra             │
└ Status: dataset, processing state, progress, warnings ────────────────────┘
```

MainWindow の主要操作は「開く」「表示レイヤを選ぶ」「現在エネルギーを変える」「解析ウインドウを開く」「sessionを保存」の5つに限定する。I0、登録、フィット、相同定の詳細パラメータは MainWindow に置かない。

## MultivariateWindow（SVD/PCA）

SVD/PCA は独立したウインドウにし、単独の相同定にも補助処理にも使えるようにする。

```text
入力: raw / OD / registered OD / ROI / pixel matrix
設定: energy range / mask / center or z-score / component count / method
表示: singular values・寄与率、loading spectra、score map、residual map
操作: low-rank reconstruction を layer 登録、denoise dataset を派生作成
相同定: score/loading を reference と照合、候補・信頼度・unknown を出力
```

低ランク再構成は新しい派生データとして作り、元の OD は変更しない。ノイズ除去を有効にした場合、以後のピークマップ/フィット/相マップには「どの component 数と前処理を使ったか」を表示する。

## モジュールウインドウ間のデータ連携

```text
MainWindow
  ├─ RegistrationWindow ── registered stack ─┐
  ├─ MultivariateWindow ─ denoised / score  ├─> SegmentationWindow
  │                                         ├─> PeakMapWindow
  ├─ FittingWindow ────── fit fractions ────┤
  └─ PhaseMapWindow <──── candidates ───────┘
```

ウインドウ間は Qt signal/slot で `LayerAdded`, `SessionChanged`, `TaskProgress`, `WarningRaised` を通知する。別ウインドウの内部 widget を参照する実装は禁止する。

### 操作モデル

| 操作 | UI | 処理層 command | 出力 |
| --- | --- | --- | --- |
| データを開く | Toolbar | `load_scan` | Scan table と raw image |
| I0 を定義 | I0 タブ、手描きROI または `i0.txt` | `set_i0` / `compute_od` | OD stack、警告 |
| ROI を描く | Image workspace | `add_roi` / `update_roi` | 名前付き mask、平均 XANES |
| 位置合わせ | Registration dialog | `estimate_shifts` / `apply_registration` | shifts、登録済み stack、品質指標 |
| しきい値/クラスタ | Cluster dialog | `segment` / `cluster_spectra` | label image、クラスタ表・平均曲線 |
| RGB/スペクトルマップ | Peak Map / Spectral Map dialog | `make_peak_map` / `make_spectral_map` | float map、表示 LUT、統計 |

## ダイアログ

- **RegistrationDialog**: 基準エネルギー、POC/手動シフト、補正前後表示、フレーム別シフト表、品質指標。
- **ClusterAnalysisDialog**: 表示範囲、ヒストグラム、しきい値、最小画素数、ラベル重ね描き、クラスタ選択と平均スペクトル。
- **PeakMapDialog**: R/G/B のエネルギー、ベースライン、正規化、閾値、結果統計、PNG/TIFF/CSV 出力。
- **ReferenceDialog**: `reference/*.txt` の選択、エネルギー補間、重ね描き。初期段階ではフィットは含めない。

## 非同期・エラー表示

ロード、OD、位置合わせ、画素マップは `QThreadPool` worker で実行する。worker は入力のスナップショットを受け、結果または構造化例外だけを返す。画面コンポーネントへの直接アクセスは禁止する。エラーは「対象ファイル/フレーム/期待形状/実際形状」を status dock に表示する。

## 初期実装の境界

第1段階は MainWindow に「開く・raw表示・メタデータ表・フレーム選択」を実装する。第2段階で I0/OD と ROI スペクトルを追加する。この時点で Xojo の核心用途を安全に置換でき、位置合わせ・クラスタ・RGB マップは後続ダイアログとして増設できる。

## 起動と開発環境

Python 実行はプロジェクト直下の Makefile を唯一の入口にする。`make setup` が `.venv` を作成し、GUI/解析/開発用の依存関係をその環境だけに導入する。`make run` と `make test` はシステム Python を使わず `.venv/bin/python` を明示的に使う。

```bash
make setup
make test
make run ARGS="--inspect '/Volumes/Extreme Pro/stxm/UVSOR/191022/UV_191022002/UV_191022002.hdr'"
```

GUI 実装後も起動コマンドは `make run` のままとし、`src/muaxis/__main__.py` の入口だけを GUI 起動へ切り替える。

現在はこの入口を実装済みで、引数なしの `make run` は `MainWindow` を起動する。`make run ARGS="--open /path/to/scan.hdr"` なら指定データを開いた状態で起動でき、`--inspect` は従来どおり GUI を使わない読込確認に利用できる。
