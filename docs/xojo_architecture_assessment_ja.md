# muNEXAFS / XANES Xojo プロジェクト調査

調査日: 2026-09-08  
対象: `xojo/XANES.xojo_binary_project`（Xojo 2021r3.1 のバイナリ形式）

## 結論

これは STXM のエネルギー走査画像スタックを読み込み、I0 補正後の光学密度 (OD) を基に、位置合わせ、ROI スペクトル、クラスタ解析、ピークの RGB マッピングを行うデスクトップ解析アプリケーションである。

処理の核は次である。

```text
.hdr + .xim スタック
  -> エネルギー / 光源電流 / 時刻を取得
  -> I0（手描き領域の平均、又は i0.txt）で正規化
  -> OD = ln(I0 / I)
  -> （任意）ドリフト補正・差分画像
  -> ROI 平均 XANES / クラスタ平均 XANES / RGB ピークマップ
  -> 画像・テキスト・設定を保存
```

古い構成では Window、Module、Thread が多数の共有グローバル状態（`hipics`、`pixdivsum`、選択ペン等）を直接読み書きする。そのため Python 版では「数値計算コア」と「GUI 状態」を分離することが最重要である。

## 入出力と主要データ

| 種類 | 実装上の扱い | Python 版の扱い |
| --- | --- | --- |
| `.hdr` | 走査ごとの `.xim` ファイル名、energy、source current、time を列挙 | 専用パーサで `ScanMetadata` とファイル一覧へ変換 |
| `.xim` | タブ区切りの 2D 強度行列 | `numpy.ndarray[height, width]` として読込 |
| `.txt` / `.raw` | 単一画像としても入力可 | 同一の 2D 読込経路へ統合 |
| `i0.txt` | 各エネルギーの背景/I0値 | `numpy.ndarray[n_energy]` として読込・検証 |
| `drift.txt` | 各画像の x/y シフト | `numpy.ndarray[n_energy, 2]` として読込 |
| `EC/diff_*.jpg` | 既存の差分画像を再読込 | 任意の互換読込機能（初期移植の必須対象外） |
| `reference/*.txt` | C、O、Ca 吸収端の参照スペクトル | `resources/reference/` として配布し、参照重ね描き・フィットに利用 |

## 実データで確認した仕様

参照先: `/Volumes/Extreme Pro/stxm/UVSOR`。この配下には macOS の `._*` 補助ファイルを除いて多数の実測 `.hdr` があり、少なくとも 2015–2023 年の UVSOR 測定キャンペーンを含む。読込時には隠しファイルと `._*` を必ず除外する。

代表例 `191022/UV_191022002` で次を確認した。

| 項目 | 実測値 | 実装への反映 |
| --- | --- | --- |
| 測定種別 | `NEXAFS Image Scan` / `Image Stack` | STXM-NEXAFS スタックとして扱う |
| 空間軸 | Sample X/Y、各 212 点、範囲 32 µm | 配列形状 `(Y=212, X=212)` と座標軸を保持する。各 `.xim` 行の末尾には空タブがあるため reader でその列だけ除外する。ヘッダの Points と実ファイルの形状が一致しない場合は警告する |
| エネルギー軸 | 114 点、280.0–299.8 eV | `.hdr` 内の `Image000_0`–`Image113_0` をファイルの `_a000`–`_a113` と同じ index で対応させる |
| 画像 | 各 `.xim` はタブ区切り、212 行 × 212 列（行末に空タブあり）、代表画像の値域 0–2766 | 行末の空列だけを除く専用 reader で、矩形・有限値を検証する |
| I0 | `i0.txt`: 114 行の `値,0.000000`、代表値域 507.1863–2695.863 | 第 1 列を I0 として使用し、フレーム数一致を必須検証する |
| ドリフト | `drift.txt`: 114 行の `x,y`、例では x=0.0088–1.0 px、y=0.1320–13.4794 px | 画像登録の初期値/再現用として読込み、適用前後を別データとして保持する |
| source current | `StorageRingCurrent = 0.00` | 0/欠損値は無効値である。旧式の `I * 300/current` を実行してはならず、補正有効化時にもこの条件はエラーとして扱う |

`UV_191022002.hdr` の `StackAxis` はエネルギー配列を含む一方、各 `ImageNNN_0` レコードが acquisition time、StorageRingCurrent、ZP の記録を持つ。Python パーサは両者を保存し、エネルギー値は Image レコードを優先、StackAxis を整合性チェックに使う。

`picSTXM` が実質的な 1 エネルギー画像で、強度、OD、表示画像、I0、ドリフト、メタデータを抱える。Python 版ではこれを分割し、画像スタック全体を `ScanStack` が持つ。

```text
ScanStack
  energy_eV:        (N,)
  source_current:   (N,)
  acquired_at:      (N,)
  transmission:     (N, Y, X)
  i0:               (N,) または (N, Y, X)
  optical_density:  (N, Y, X)
  shifts_xy:        (N, 2)
  metadata / provenance
```

## Xojo の構成と役割

### Window

| Xojo 要素 | 役割 | Python GUI での移植先 |
| --- | --- | --- |
| `Window1` | 主画面。ファイル/エネルギー一覧、原画像・I0・OD・スペクトル表示、ROI 塗り、I0 塗り、保存、補正/補助画面の起点 | `MainWindow`。中央の画像/スペクトル表示、左のスタック表、右の解析・ROI 操作パネル |
| `Window2` | 各エネルギー画像の相対シフトを閲覧・手修正し、POC/相互相関/重心ベースの位置合わせを適用 | `RegistrationDialog`。基準フレーム選択、シフト表、前後比較、再計算・適用 |
| `Window4` | 画素ごとのスペクトルを基にしたマッピングと合成色表示。`thrMap` の進捗を表示 | `SpectralMapDialog`。演算式・閾値・色チャネルを明示し、結果をレイヤとして表示 |
| `Window5` | OD/差分画像のヒストグラム、階調範囲、連結クラスタ、クラスタごとの平均 XANES | `ClusterAnalysisDialog`。ヒストグラム/閾値、ラベル画像、クラスタ表、スペクトル図 |
| `Window6` | 解析画像の選択、カラーマップ、閾値/LUT、画像表示・保存の補助画面 | `ImageDisplayDialog`（初期版では MainWindow の表示設定ドックに統合） |
| `Window7` | ピークエネルギーを R/G/B に割り当て、基準・正規化・しきい値を指定して RGB マップと統計を生成 | `PeakMapDialog`。R/G/B のエネルギー、ベースライン、正規化、統計、エクスポート |

### Class

| Xojo class | 主要メソッド | 役割と Python 方針 |
| --- | --- | --- |
| `picSTXM` | `GetIntensity`, `hipicFromTxt`, `gethw` | タブ区切り画像とメタデータ/ODを保持。Python ではデータ保持を `ScanStack` に移し、`io.read_matrix` として読込処理を独立させる。 |
| `hiPicture` | `fft2`, `ifft2`, `getFFTpic`, `poc`, `crosscorr`, `gravityCenter`, `shift` | Xojo `Picture` に数値処理を混在させた画像ユーティリティ。`numpy` / `scipy.ndimage` / `skimage.registration` に置換する。 |
| `hipicGraph` | `DrawPlot`, `DrawSymbol`, `GetRange`, `GetValue`, `DrawVerticalLine` | 描画と座標変換を担当。計算は `plotting` に限定し、描画は `pyqtgraph`（又は Matplotlib）に移す。 |
| `Complex` | 演算子、三角/双曲線関数、FFT用演算 | Python 組込みの `complex` と NumPy FFT で完全に置換可能。移植しない。 |
| `thrLoad` | `Run` | 読込、I0、OD、既存ドリフト/差分データ復元をバックグラウンド実行。Python では worker が不変に近い結果を返し、GUI スレッドで反映する。 |
| `thrMap` | `Run` | 画素ごとのスペクトルマップ計算。`numpy` ベクトル化を優先し、必要時のみキャンセル可能 worker にする。 |

### Module

| Xojo module | 主なメソッド | 役割 |
| --- | --- | --- |
| `modSTXM` | `readSTXM`, `readfile` | `.hdr` と画像スタックを読み込む。 |
|  | `drawAbsorbance`, `Datashift` | `ln(I0/I)` を算出。`Datashift` は位置補正済み強度から OD を再計算する。 |
|  | `getgraph`, `drawgraph`, `clearGraph` | カラーペンで塗った ROI のエネルギー対 OD を平均し、複数曲線を描画する。 |
|  | `getPosition`, `shiftCorr` | POC/相互相関/重心による相対位置合わせと画像移動。 |
| `muModuleStandard` | `getPallets` ほか | パレット、標準的なファイル/画像補助。内容の多くは GUI ライブラリまたは小さな utility に分割する。 |
| `FFT_Suite` | `FFT`, `FFT_int`, 各種 window 関数、`ZeroPad` | FFT 実装。`numpy.fft` と `scipy.signal.windows` に置換する。 |

## 処理詳細と移植上の判定

1. **I0 と OD**
   - 旧式は手描き I0 領域、または `i0.txt` の背景値を使用し、各画像で `OD = log(I0 / transmission)` を計算する。
   - ゼロ・負値・欠損値の扱いが明示されていない。Python 版では `NaN` に統一し、分母下限 `epsilon`、マスク、警告を必須にする。
   - source current 補正の切替があり、`I * 300 / source_current` を適用する。係数 300 は UI/設定に露出し、来歴に記録する。

2. **位置合わせ**
   - POC（phase-only correlation）、相互相関、重心を含む。最終的に `imageShift` で補正し、OD を再算出する。
   - Python 版は `skimage.registration.phase_cross_correlation` を標準とし、サブピクセル移動は `scipy.ndimage.shift` を使用する。元データを破壊せず `registered_transmission` を派生データとして保存する。

3. **ROI とスペクトル**
   - 色ごとの塗りマスクを画素ごとに集計し、エネルギーごとの平均 OD を描く。
   - Python 版では `bool[Y, X]` の ROI マスクを独立した名前付きレイヤとして保存する。任意数の ROI を許可し、CSV と画像をエクスポートする。

4. **クラスタとマップ**
   - `Window5` はしきい値化画像の連結成分を作り、クラスタ平均スペクトルを作成する。
   - `Window4` / `thrMap` は画素スペクトルから複数チャネルの成分量を計算する。旧コードは説明変数と式の一部が UI イベントへ埋込まれているため、移植時は式を `MappingRecipe` として明文化し、検証サンプルで旧結果と比較する。

5. **表示と保存**
   - Xojo の `Picture` は解析配列と表示用の 8-bit 階調画像の両方を兼ねる。
   - Python 版では解析値を float 配列のまま保持し、LUT/コントラスト/透明度を表示専用にする。PNG/TIFF、CSV、設定 JSON（将来は HDF5/OME-Zarr）を出力対象にする。

## 推奨する Python アーキテクチャ

```text
src/muaxis/
  domain/
    models.py             # ScanStack, Roi, Registration, AnalysisSession
    validation.py
  io/
    stxm.py               # .hdr/.xim/.txt/.raw 読込
    project.py            # セッション JSON + 配列保存
    export.py             # CSV/PNG/TIFF
  processing/
    absorbance.py         # I0、光源電流補正、OD
    registration.py       # POC、シフト、品質指標
    roi.py                # ROI 集計・スペクトル
    cluster.py            # threshold/connected components
    mapping.py            # peak/RGB/spectral map
  gui/
    main_window.py
    registration_dialog.py
    cluster_dialog.py
    peak_map_dialog.py
    workers.py
  plotting/
    image_layers.py
    spectra.py
tests/
  fixtures/               # 匿名化した小型 hdr/xim と期待結果
  test_io.py
  test_absorbance.py
  test_registration.py
  test_roi.py
```

GUI はクロスプラットフォームの **PySide6 + pyqtgraph** を推奨する。理由は、画像スタックの高速表示、ROI 描画、複数スペクトルの対話操作、`QThread`/signal による安全なバックグラウンド処理を一貫して実装できるためである。科学計算は NumPy/SciPy/scikit-image に集約する。

## 実装順序と受入条件

1. `.hdr/.xim` 読込と検証、スタック表・画像スライダ表示。
2. I0 ROI / `i0.txt`、source current 補正、OD の算出・表示。
3. 名前付き ROI、ROI 平均スペクトル、CSV/画像エクスポート。
4. POC 位置合わせと手修正、補正前後の比較・ドリフト保存。
5. ヒストグラム、クラスタ、ピーク RGB、スペクトルマップ。
6. 小型の既知データに対し、各段階で旧版との数値比較テストを追加。

初期受入条件は、同一の `.hdr/.xim` と I0 指定から、(a) エネルギー順、(b) 強度行列の形状、(c) ROI 平均 OD、(d) POC シフトが再現可能な許容差内にあることとする。表示画像のピクセル一致ではなく、元の浮動小数点データを比較対象とする。

## 要確認事項

- バイナリ Xojo プロジェクトからコードを抽出して調査した。最終的な互換実装に入る前に、Xojo IDE から XML/Text project として export したソースを確保し、イベント名・UI キャプション・コメントアウトされていない分岐を照合する。
- 代表的な実測データ一式（`.hdr` と参照される `.xim`、任意で `i0.txt` / `drift.txt`）と、旧版で保存した期待結果が必要である。これがないと科学的な数値互換性は検証できない。
- `thrMap` の物理的定義（係数 a/b/c/d、閾値、各 UI パラメータの意味）はコードだけでは十分に自明でない。測定者の運用手順または画面キャプチャで仕様を固定する。
