# 作業報告書: Windows テスト CI の追加（PR #650 対応）

- 日付: 2026-08-20 〜 2026-08-21
- 作業ブランチ: `feature/windows-test`（`feature/windows` から分岐）
- 検証用 PR: https://github.com/lzpel/simsopt/pull/2 （fork 内 `feature/windows-test` → `feature/windows`）
- 発端: upstream PR [hiddenSymmetries/simsopt#650](https://github.com/hiddenSymmetries/simsopt/pull/650) にてメンテナ mbkumar より
  「テストワークフローも Windows で実行するよう更新してほしい」との要求

## 目的

`.github/workflows/tests.yml` を拡張し、ubuntu と Windows で**テストコマンドを共有**しながら
ユニットテストを Windows でも実行する。Windows 側は次の2変種を用意して比較する:

| 変種 | Python | ビルド | 位置づけ |
|---|---|---|---|
| `default`（Cygwin） | Cygwin python39 (3.9) | gcc 14 (Cygwin) | Unix 互換路線の検証 |
| `native-python` | setup-python 3.11 (MSVC ビルド) | MSVC | 出荷 wheel と同一のビルド方式 |

## 最終結果（run 32447216715）

| ジョブ | 結果 |
|---|---|
| ubuntu-24.04 × unit / integrated | ✅ green（既存動作にリグレッションなし） |
| **windows-2025 × native-python** | ✅ **green — 全8ディレクトリのユニットテストスイート通過**（3ラン連続） |
| windows-2025 × Cygwin | ❌ red — ビルド・インストールは成功、テストが構造的要因で失敗 |

### Cygwin 変種の構造的限界（実測で確定）

- **jax**: Cygwin 用の配布物（wheel/ビルド可能な sdist）が存在しない。`simsopt.geo` 等が無条件 import するため **32 テストモジュールが import 不能**。回避不能
- **scipy**: Cygwin パッケージが存在せず（全 Python 版）、**30 モジュールが import 不能**
- Cygwin 公式 matplotlib パッケージが、同じく公式の numpy 2.0.1 とバイナリ非互換（`numpy.core.multiarray failed to import`）

## CI 反復の記録（13 ラン）

| # | 失敗箇所 | 原因 | 対処 |
|---|---|---|---|
| 1 | symlink ステップ | ランナーは Windows で run スクリプトを CRLF で書き出し、Cygwin sh は `\r` を除去しない | `SHELLOPTS=igncr` を Cygwin が PATH に入る前に `GITHUB_ENV` へ設定 |
| 2 | pip 依存導入 | matplotlib sdist のビルド分離環境が numpy をソースビルド、native ccache × Cygwin cc が衝突 | matplotlib/h5py を Cygwin バイナリパッケージ化、Cygwin ccache/ninja 追加、cmake/ninja/scikit-build の pip 導入を Linux 限定へ |
| 3 | `pip install .` | **jaxlib が Cygwin に存在しない**（`from versions: none`） | ユーザー判断で「--no-deps 続行」と「native 変種新設」の両方を実施 |
| 4 | CMake configure | ビルド分離環境で FindPython が libpython/numpy を発見できず（毎回 numpy を約14分再ビルド） | `--no-build-isolation` + `CMAKE_ARGS` で interpreter/include/library を明示 |
| 5 | （native 初回）テスト12件失敗 | ①テストの `/tmp/` ハードコード ②cp1252 コンソールでの Unicode print ③数値不一致 | ①5ファイル13箇所を `tempfile.gettempdir()` 化 ②`PYTHONUTF8: 1` ③下記「発見したバグ」参照 |
| 6 | Cygwin numpy import | `liblapack0` の alternatives postinstall が動かず `cyglapack-0.dll` リンク欠落 | まず symlink → **Windows ローダは Cygwin symlink を辿れず** Permission denied → **コピー**で解決 |
| 7-8 | （native）fluxobjective ×3 | **OpenMP データ競合**（下記） | `integral_BdotN.cpp` 修正 → **native green 達成** |
| 10 | Cygwin C++ コンパイル | Cygwin gcc は旧 libstdc++ ABI がデフォルト、xtensor が gcc14 で削除済みの `std::has_trivial_default_constructor` を選択 | CMakeLists に Cygwin 分岐 `-D_GLIBCXX_USE_CXX11_ABI=1` |
| 11 | 同上 | xsimd が `_GNU_SOURCE` 定義下で newlib に無い `::sincosf` を要求 | サポート済み構成 `NO_XSIMD=1` を使用 |
| 12 | リンク | Cygwin ld が LTO オブジェクトのシンボルを auto-export できない（`symbol wrong type (4 vs 3)`） | `-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF`（pybind11 の -flto 注入も止まる） |
| 13 | テスト実行 | pip の console script が PATH 外の `/usr/local/bin` に入る | `GITHUB_PATH` に追加 → **Cygwin もテスト実行に到達**（上記の構造的限界が確定） |

## 発見・修正した本体のバグ

### 1. `src/simsoptpp/integral_BdotN.cpp` の OpenMP データ競合（修正済み）

`mod_B_squared` が `#pragma omp parallel for` ループの**外側**で宣言されており、スレッド間で共有されていた。
`SquaredFlux` の `definition="normalized"` / `"local"` が**全プラットフォームで非決定的な値**を返す
（CI と手元で 0.1〜0.5% の変動を観測。`OMP_NUM_THREADS=1` で完全に決定的になることを確認）。
Linux では偶然テストを通過し続けていた潜在バグで、Windows CI 追加が表面化させた。
ループ内宣言（スレッドローカル化）で修正。

### 2. `src/simsoptpp/boozerradialinterpolant.cpp` の同種の競合（未修正・要 upstream 報告）

`fourier_transform_odd` / `fourier_transform_even` にて `kmns(im) +=` と `norm +=` が
`#pragma omp parallel for` 内で **reduction 指定なし**に行われている。より重度の競合であり、
別途修正を upstream に提案する価値がある。

## テスト・ワークフローへのその他の変更

- テスト中の `/tmp/…` ハードコード 13 箇所（`tests/field/`, `tests/geo/` の5ファイル）を
  `os.path.join(tempfile.gettempdir(), …)` に置換（Linux では従来どおり /tmp に解決）
- `tests/field/test_selffieldforces.py` の per-coil 和 vs 一括計算の比較 atol を 1e-30 → 1e-28
  （約 1e-23 同士の比較で float64 のノイズ床未満だったため。観測絶対差 1.3e-30）
- ユニットテストステップを「最初の失敗で停止」から「全ディレクトリ実行→最後に失敗判定」の
  ループへ変更（1ランで全失敗ディレクトリが判明する）
- `windows × integrated` は matrix exclude（examples が MPI/VMEC 前提のため）
- MPI テスト・VMEC2000・booz_xform・virtual_casing・desc-opt・coverage アップロードは
  `runner.os == 'Linux'` ガード

## 推奨事項

1. **upstream（PR #650）には native-python 変種のみを載せる。**
   出荷される wheel と同じ MSVC ビルドを検証でき、フルスイートが green。
   Cygwin 変種は「実測の結果、jax/scipy が存在せず実用にならない」ことが確定したため、
   tests.yml から関連ステップを除去してシンプル化するのが良い。
2. データ競合修正（integral_BdotN）は Windows テスト追加の成果として PR に含める。
   boozerradialinterpolant の競合は別 issue/PR として報告する。
3. 留意点: native ジョブは実行に約 2.5 時間かかる（field/geo 系のテストが遅い）。
   upstream 導入時に頻度・対象の調整（例: nightly 化や対象ディレクトリの絞り込み）を
   メンテナと相談する余地がある。
