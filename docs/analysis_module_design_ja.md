# STXM-NEXAFS 解析モジュール基本設計

## 方針

各解析は `ScanStack`（入力）、設定 dataclass（パラメータ）、結果 dataclass（出力）の純粋な処理関数として実装する。GUI は設定を作り、結果を表示・保存するだけで、NumPy 配列や計算式を直接扱わない。これにより、同一機能を GUI、CLI、Notebook、バッチ処理で共有できる。

```text
io → validation → preprocess → registration → OD
                                           └→ segmentation / ROI
OD + reference library → fitting → phase identification → maps / report
                     └→ peak mapping ─────────────────────┘
```

## 「非線形ピークフィッティング」と Xojo 版の関係

非線形ピークフィッティングとは、スペクトルを Gaussian、Lorentzian、Voigt、step/edge などの関数の和で表し、ピーク位置・幅・高さ・面積・背景係数を未知パラメータとして、残差が最小になるよう反復最適化する方法である。ピーク位置や幅も動かせるため、単純な参照スペクトルの足し合わせとは異なる。

今回抽出した Xojo ソースには、**この意味での非線形ピークフィッティングは実装されていない**。

- `Window4`/`Window6` 周辺の参照処理は、参照 TXT を測定エネルギーへ線形補間し、各スペクトルを min-max 正規化して `R2 = 1 - Σ(y-y_ref)^2 / ...` を比較する処理である。複数参照を選ぶ場合も重み `dratio` を足し合わせているが、係数を最適化する NNLS ではない。
- `Window7` の Peak analysis は指定エネルギーにおける RGB の高さ・積分値・平均・分散を作る閾値型マッピングであり、ピーク関数の最適化ではない。
- `fitQuad` / `fitQuad1` は二次/一次多項式係数を解析的に求める補助関数で、ピーク位置・幅を推定する非線形フィットではない。

したがって Python 版では、既存互換の「参照スペクトル照合/R2」と、新機能の NNLS、さらに将来の非線形ピークフィットを同じ `fitting` モジュールの別 strategy として実装する。既存結果を再現するため、最初から非線形フィットへ置き換えない。

全画像配列の軸順は **`(energy, y, x)`**、座標は **`(x, y)`**、エネルギーの単位は **eV** に固定する。計算値は float64、無効画素は `NaN`、分類ラベルは `int32`（背景 0、相/クラスタは 1 以上）とする。

## モジュール一覧

| 層 | モジュール | 主な責務 | 主出力 |
| --- | --- | --- | --- |
| domain | `models` | `ScanStack`、ROI、Session、操作来歴の不変モデル | 解析状態 |
| io | `stxm` | `.hdr/.xim/i0.txt/drift.txt` 読込 | 強度スタック、メタデータ |
| io | `reference` | 参照スペクトル読込、メタデータ/元素/相の管理 | `ReferenceSpectrum` |
| validation | `dataset_qc` | 形状、欠損、エネルギー単調性、飽和、I0、ドリフトの検査 | `QualityReport` |
| preprocessing | `calibration` | エネルギー校正、補間、binning、不要フレーム除外 | 校正済み stack |
| preprocessing | `normalization` | I0、光源電流、暗電流、ODの算出 | transmission / OD |
| preprocessing | `denoise` | 任意のノイズ低減、外れ値/ホットピクセル補正 | 派生 stack |
| registration | `estimate` | POC、相互相関、重心、品質指標によるシフト推定 | `RegistrationResult` |
| registration | `transform` | 平行移動/補間/共通有効領域の適用 | 登録済み stack、valid mask |
| roi | `masks` | 手描き/多角形/ラベル ROI、論理和差、永続化 | `Roi` / mask |
| roi | `spectra` | ROI・画素・線プロファイルの平均、分散、誤差 | `Spectrum` |
| segmentation | `threshold` | 単一/二値/範囲/適応閾値、形態学処理 | boolean mask |
| segmentation | `labeling` | 連結成分、最小面積、測定量、クラスタ表 | label image / regions |
| clustering | `spectral` | 画素スペクトルの k-means/GMM/hierarchical/PCA | labels / embeddings |
| mapping | `peak` | ピーク位置・高さ・面積・事前相分布の予測 | float/RGB map、信頼度 |
| fitting | `preprocess` | フィット範囲、前縁/後縁、ベースライン、正規化 | 前処理済み spectrum |
| fitting | `reference_match` | Xojo互換の補間・正規化・R²類似度 | 候補順位 / residual |
| fitting | `linear` | 参照線形結合、NNLS、重み付き最小二乗 | fractions / residual / covariance |
| fitting | `nonlinear` | ピーク関数/エッジ関数の非線形フィット（将来） | parameters / uncertainty |
| decomposition | `svd` | スタックの低ランク分解、ノイズ/成分探索（将来） | components / singular values / scores |
| decomposition | `pca` | 平均中心化、主成分、score/loading、寄与率（将来） | scores / loadings / explained variance |
| identification | `phase` | フィット・ピーク特徴・ルールから相候補を順位付け | `PhaseAssignment` |
| visualization | `colormap` | スカラー、label、相、RGB、多相オーバーレイ | 表示レイヤ |
| export | `results` | CSV、TIFF/PNG、NPZ/HDF5、JSON session、解析報告 | 再読込可能な成果物 |
| application | `commands` | 処理の実行、Undo/Redo、キャッシュ無効化 | 新しい Session |
| application | `workers` | GUI非同期処理、キャンセル、進捗通知 | task event |

## 中核データモデル

```python
@dataclass(frozen=True)
class ScanStack:
    energy_eV: NDArray[float]             # (E,)
    transmission: NDArray[float]          # (E, Y, X)
    x_um: NDArray[float]                  # (X,)
    y_um: NDArray[float]                  # (Y,)
    i0: NDArray[float] | None             # (E,) または (E, Y, X)
    valid_mask: NDArray[bool]             # (Y, X)
    metadata: ScanMetadata

@dataclass(frozen=True)
class Spectrum:
    energy_eV: NDArray[float]
    value: NDArray[float]
    std: NDArray[float] | None
    n_pixels: int
    source: str                           # ROI名/cluster番号/pixel座標

@dataclass(frozen=True)
class FitResult:
    model_name: str
    coefficients: dict[str, float]
    reconstructed: NDArray[float]
    residual: NDArray[float]
    rmse: float
    r_squared: float
    uncertainty: dict[str, float] | None
```

データは `AnalysisSession` に、**元データ → 各派生データ → 生成設定 → ソフトウェア版**の有向非巡回グラフとして保持する。ユーザーの ROI 編集または補正パラメータ変更時には、その下流のキャッシュだけを失効させる。

## 各解析モジュール

### 1. 読込・品質管理・前処理

#### `io.stxm`（実装済み）

- `.hdr` の `ImageNNN_0` を基準に `.xim` を対応付け、ディレクトリ順に依存しない。
- 画像、エネルギー、I0、ドリフトを読み込む。
- 読込時は数値を変えない。OD・補正・位置合わせは後段だけが担う。

#### `validation.dataset_qc`

- エネルギーの重複/非単調、フレーム・I0・ドリフトの数の不一致、ゼロ/負/飽和画素、欠損、急激な全画面強度変化を検査する。
- StorageRingCurrent が 0/欠損なら source-current 補正を禁止する。`UV_191022002` はこのケースである。
- 結果を error（計算停止）、warning（継続可）、info に分け、GUI の status dock と保存レポートへ渡す。

#### `preprocessing.calibration` / `normalization` / `denoise`

- **校正**: 基準ピークとの相関又はユーザー指定オフセットにより、エネルギー軸を補正する。補間の方式と範囲外処理を記録する。
- **正規化**: `OD = ln(I0 / I)` を算出する。`I <= 0`、`I0 <= 0`、非有限値は `NaN` としてマスクし、epsilon で隠蔽しない。
- **ノイズ低減**: median、Gaussian、PCA/SVD、NLM を任意の派生処理として提供する。定量フィットでは元ODまたは前処理内容を明示したODだけを使う。

### 2. レジストレーション

#### `registration.estimate`

入力: 任意の 2D 表示/OD stack、`RegistrationConfig(method, reference_index, upsample_factor, roi_mask)`  
出力: `RegistrationResult(shifts_xy, error, phase_difference, quality, reference_index)`。

- 標準方式は `skimage.registration.phase_cross_correlation` の POC。サブピクセル推定に対応する。
- 代替として相互相関、重心、手入力シフト、隣接フレーム累積を持つ。
- 画素全体でなく ROI を使えるようにする。品質が低いフレームは警告し、隣接推定/手修正を促す。

#### `registration.transform`

- `scipy.ndimage.shift` で補間を適用する。補間次数、境界モード、NaN の扱いを設定に保存する。
- 補正後の共通有効領域を `valid_mask` として作成する。端部の補間値を平均・フィットへ混ぜない。
- 原画像は変更せず、`registered_transmission` と、そこから再計算した `registered_od` を別ノードにする。

### 3. ROI・閾値・クラスタラベリング

#### `roi.masks` / `roi.spectra`

- ROI は名前、色、作成方法、boolean mask、座標系、作成時の親データIDを持つ。
- 平均、中央値、標準偏差、標準誤差、画素数、無効画素数をエネルギーごとに返す。
- マスクは I0 ROI、解析ROI、除外ROIの型を区別する。ROI はログに残す。

#### `segmentation.threshold`

入力: 選択エネルギーのOD、ピーク差分、比、フィット係数、又は任意 feature map。  
出力: boolean mask。

- 固定閾値、範囲閾値、Otsu、適応閾値を提供する。
- opening/closing、穴埋め、border exclusion は設定として独立させる。
- しきい値の対象データ/単位/正規化状態を結果名に含め、ODと8-bit表示値を取り違えないようにする。

#### `segmentation.labeling`

- 4近傍/8近傍を選択可能な connected-component labeling。
- 最小/最大面積、除外境界、ROI 制限を適用する。
- 各領域について面積、重心、bbox、平均OD、平均スペクトル、形状指標を `RegionTable` として返す。

#### `clustering.spectral`

- ピクセルをエネルギー方向のベクトルとして扱い、標準化・PCA後に k-means、GMM、階層クラスタを実施する。
- 入力マスク、エネルギー範囲、標準化方式、ランダムseedを必ず保存する。
- クラスタ番号は任意であるため、色や相名は別レイヤに保持する。連結性が必要なら labeling と組み合わせる。

### 4. ピークマッピング（事前相分布予測）

#### `mapping.peak`

これは**相の確定ではなく候補分布を高速に可視化するスクリーニング処理**と位置付ける。入力は OD stack と `PeakMapRecipe`、出力は各 feature map、RGB map、valid mask、警告である。

`PeakMapRecipe` の例:

```python
PeakMapRecipe(
    channels={"R": PeakFeature(energy_eV=285.0, baseline=(284.0, 286.0)),
              "G": PeakFeature(energy_eV=286.7, baseline=(286.0, 287.3)),
              "B": PeakFeature(energy_eV=288.5, baseline=(288.0, 289.0))},
    feature="baseline_subtracted_height",  # height / ratio / area / first_derivative
    normalization="per_channel_percentile",
    percentile=(1, 99),
)
```

- 指定エネルギーが実測点にない場合は線形補間し、補間距離を記録する。
- ベースライン差分、高さ、ピーク面積、ピーク位置、比、一次微分を実装対象とする。
- RGBは表示方式であり、各 channel の生の float map と正規化パラメータを必ず保存する。
- 事前相分布には `candidate_phase_score`（0–1 又は未正規化の類似度）を出し、フィット残差/信頼度が低い画素を透明表示にできるようにする。

### 5. スペクトルフィッティング

#### `fitting.preprocess`

- フィット窓、前縁/後縁範囲、多項式ベースライン、正規化、重み、除外点を適用する。
- 入力と出力のエネルギー軸を保持し、参照スペクトルには同じ格子への補間を行う。
- ROI 平均にまず対応し、pixel-wise fitting は同じ関数を block/chunk 単位で呼ぶ。

#### `fitting.linear`

入力: `Spectrum`、複数の `ReferenceSpectrum`、`LinearFitConfig`。  
出力: `FitResult` と相ごとの係数 map（pixel-wise 時）。

- 初期標準は NNLS（係数 >= 0）とする。必要時は係数和=1 の制約を選べるようにする。
- 重み付き最小二乗、参照ごとの energy shift、交差検証による参照候補選択を後から追加可能な API にする。
- 出力は係数、再構成、残差、RMSE、R²、自由度、推定不確かさ。単に最大係数だけを返さない。

#### `fitting.reference_match`

- Xojo 版の互換経路。参照 TXT を測定エネルギー軸へ線形補間し、min-max 正規化、R²相当の残差を計算する。
- `ReferenceMatchResult` は最良参照だけでなく、全候補のスコア、補間範囲、正規化方式を返す。
- これは混合比の定量ではないため、GUI でも「similarity / candidate」と表記し、相同定の確定結果と区別する。

#### `fitting.nonlinear`

- Gaussian/Lorentzian/Voigt と step/edge 関数の和を `scipy.optimize.least_squares` でフィットする。
- 初期値・bounds・ロバスト損失関数を設定可能にし、収束状態を返す。
- ピーク位置/面積/幅の定量には使うが、相の混合比の主手段は参照線形結合とする。

### 6. SVD / PCA によるスペクトル分解（将来）

#### `decomposition.svd`

- 有効画素 × エネルギーの行列を平均中心化または baseline 補正し、`X = UΣVᵀ` を分解する。
- 特異値の scree plot、累積寄与率、低ランク再構成、残差 map を返す。
- SVD の成分は物理相とは限らない。成分の符号・順序・回転は不定なので、相名や相カラーを直接付けない。

#### `decomposition.pca`

- SVD を用いた PCA を標準実装とし、score（画素側）、loading（エネルギー側）、explained variance、再構成誤差を返す。
- ROI/マスク、energy range、標準化（center、z-score、none）、component 数、random seed を記録する。
- PCA score map は `mapping` や `clustering` の入力 feature に利用できる。相同定では、既知試料で主成分を校正した単独判定、または参照フィット/検証済み分類器との併用を選択できる。

SVD/PCA は独立した **多変量解析層** とする。単独で score/loading のパターン、クラスタ、既知データとの対応を使った相同定に利用できる一方、OD 前処理後のノイズ評価、低ランクノイズ除去、他のピーク/RGB/相マップの feature 作成にも利用する。従って、`fitting` や `identification` へ直接埋め込まず、結果を共通の `FeatureSet` として渡す。将来、NMF/ICAを追加する場合も同じ `fit(matrix, config) -> DecompositionResult` API に揃える。

SVD/PCA の用途は次の3種類を明示する。

1. **単独解析**: score/loading の形状、寄与率、既知試料の主成分パターンから相候補を作る。
2. **ノイズ除去**: 上位 `k` 成分による低ランク再構成を作り、元データとの差分・残差を品質指標として保存する。元データを上書きしない。
3. **他マップの補助**: score map、loading、再構成スペクトル、残差 map を閾値、クラスタ、ピークマップ、参照フィットの入力 feature にする。

### 7. 相同定・カラーマッピング

#### `io.reference` / `identification.phase`

- 参照ライブラリはスペクトル配列に加え、相名、組成、元素吸収端、出典、測定法、前処理、信頼度を持つ。
- 相同定は「分類名だけ」ではなく、候補、係数、類似度/残差、適用条件、除外理由を返す。
- 決定方式は段階的にする。

```text
候補抽出（元素/edge/energy window）
  → 参照との類似度・NNLS fit
  → 残差と係数閾値による信頼度
  → phase label / unknown / mixed
```

- `unknown` と `mixed` は必須カテゴリにし、どの参照にも無理に割り当てない。

#### `visualization.colormap`

- continuous map: perceptually uniform LUT（viridis/cividis等）と実値カラーバー。
- label/phase map: 固定した phase-to-color table。セッションをまたいでも同じ相は同じ色にする。
- RGB peak map: 各色の feature/normalization/透明度を凡例とともに保存する。
- overlay: raw/OD 背景 + 半透明の ROI/cluster/phase 境界。表示専用で解析配列を変更しない。

## 必須の横断モジュール

| モジュール | 必要な理由 |
| --- | --- |
| `domain.provenance` | どの I0、ROI、シフト、参照、閾値、ソフトウェア版で結果を出したかを保存し、再現可能にする。 |
| `application.commands` | Undo/Redo と下流キャッシュ無効化を統一する。GUIイベントの複製を防ぐ。 |
| `application.workers` | 大きな stack の読込、pixel-wise fitting、cluster を GUI から安全に分離し、キャンセル・進捗を提供する。 |
| `export.results` | CSV（スペクトル/領域表）、OME-TIFF又はTIFF（map）、NPZ/HDF5（解析配列）、JSON（session/recipe）を一組として出力する。 |
| `testing.fixtures` | 小型実データと期待値で reader、OD、shift、ROI、fit を数値回帰テストする。 |
| `logging.audit` | 操作、警告、例外、所要時間を記録して、後から結果を説明できるようにする。 |

## 実装優先順位

1. `domain.models`、`validation.dataset_qc`、`preprocessing.normalization`、`roi.masks/spectra`。
2. `registration.estimate/transform` と共通有効領域の扱い。
3. `segmentation.threshold/labeling`、クラスタ平均スペクトル。
4. `mapping.peak` と RGB 表示。ここまでは「事前相分布の予測」。
5. `io.reference`、`fitting.preprocess/linear`、`identification.phase`。ここから根拠付きの相同定。
6. `decomposition.svd/pca` を加え、クラスタ/マップの feature として検証する。
7. `fitting.nonlinear`、pixel-wise fitting、レポート/高次エクスポート。

各段階は GUI に組み込む前に、CLI とテストで独立検証する。特に相同定は、既知試料または既存 Xojo 結果との比較セットを作成してから導入する。
