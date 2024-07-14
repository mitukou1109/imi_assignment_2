#set page(margin: 20mm, numbering: "1 / 1")
#set text(font: "Noto Serif CJK JP", size: 11pt)

#set heading(numbering: "1.1")
#show heading: curr_head => locate(
  loc => {
    let heads = query(selector(heading).before(loc))
    if heads.len() > 1 {
      let prev_head = heads.at(-2)
      if (
        (
          curr_head.location().page() != prev_head.location().page() and curr_head.location().position().y > 60pt
        ) or curr_head.location().position().y - prev_head.location().position().y > 40pt
      ) {
        return [
          #v(1em)
          #curr_head
        ]
      }
    }
    return [#curr_head]
  },
)
#show heading: curr_head => locate(
  loc => {
    let heads = query(selector(heading).after(loc))
    if heads.len() > 0 {
      let next_head = heads.at(0)
      if (
        next_head.location().page() == curr_head.location().page() and next_head.location().position().y - curr_head.location().position().y < 40pt
      ) {
        return [#curr_head]
      }
    }
    return [
      #curr_head
      #par(text(size: 0.5em, ""))
    ]
  },
)

#set par(leading: 1em, justify: true, first-line-indent: 1em)

#show figure.where(kind: image): set figure(supplement: "Fig.")
#show figure.where(kind: table): set figure.caption(position: top)

#set math.equation(numbering: "(1)")
#show math.equation: it => h(0.25em, weak: true) + it + h(0.25em, weak: true)
#let trp = $sans(upright(T))$

#align(center, text(18pt)[*知能機械情報学 課題2*])
#align(center, text[03223008 坂本光皓])
#align(
  center,
  text[#datetime.today().display("[year]年[month padding:none]月[day padding:none]日")],
)
#v(1em)

= AdaBoost

== 概要

AdaBoostとは適応的ブースティング（Adaptive
Boosting）の略であり，弱学習器を増強することで精度の高い強学習器を構成するブースティングの一種である．
この手法ではそれぞれの識別器を，前段の識別器の性能に応じて重み付けしたパターンを用いて学習させる．
すなわち，ある識別器で誤識別されたパターンには大きな重みが付与され，後段の識別器の学習において重視される．すべての識別器が学習された後，各識別器の信頼度に応じた重み付け多数決で最終的な識別を行う．

== データセット

実装したアルゴリズムの評価にはIrisデータセットを用いる．これは3種類のアヤメの花びら（petal），がく片（sepal）それぞれの長さと幅という特徴量から構成されている．
このデータセットはサンプル数が150件と少ないが，異なる種類の植物を正確に分類するための初歩的な手法の理解に役立てることができる．

== 性能評価

Irisデータセットのうちversicolor，virginicaの2クラスについて，花びらの長さおよび幅を特徴量としてAdaBoostで学習を行う．なお，弱学習器には決定株を用いた．

@adaboost_result に識別器の数を10としたとき得られた決定境界を示す．正解率は98
%，学習時間は22msであった．ハイパーパラメータである識別器の数は10より大きくしても性能の向上が見られなかったため，このデータセットに対しての適切な値であるといえる．

また，データの分布から明らかなように2クラスは線形分離不可能であるが，AdaBoostはこのような非線形分類問題にも適用可能である．

#figure(image("adaboost_result.png", height: 62.5mm), caption: [学習で得られた決定境界]) <adaboost_result>

= Kernel K-means

== 概要

Kernel K-meansは教師なしクラスタリング手法の一つであるK-meansをカーネル法を用いて拡張したものである．パターン$bold(x)_n$の特徴空間における距離をカーネル関数$k(bold(x)_i, bold(x)_j)$を用いて
$ d(phi.alt(bold(x)_n), bold(nu)_k) = k(bold(x)_n, bold(x)_n) - 2 / (|cal(C)_k|) sum_(j in cal(C)_k) k(bold(x)_n, bold(x)_j) + 1 / (|cal(C)_k|^2) sum_(i in cal(C)_k) sum_(j in cal(C)_k) k(bold(x)_i, bold(x)_j) $
と定めることで，陽に特徴量$phi.alt(bold(x)_n)$や特徴空間における代表点を計算せずクラスタを分離できる．

== データセット

実装したアルゴリズムの評価には，円状に生成したノイズを含むデータを用いる．内側と外側の円の2つの領域にデータが分布しているため線形分離不可能であり，Kernel
K-meansやKernel SVM，ニューラルネットワーク等の非線形分類器のテストに適する．

== 性能評価

@kernel_k_means_result aにサンプル数が1000，ガウシアンノイズの標準偏差が0.1，外径に対する内径の比が0.2となるように生成したデータを示す．これに対し，線形カーネル$k(bold(x)_i, bold(x)_j) = bold(x)_i^trp bold(x)_j$およびガウスカーネル$k(bold(x)_i, bold(x)_j) = exp(-gamma norm(bold(x)_i - bold(x)_j)^2)$を用いたクラスタリングを行ったところ，@kernel_k_means_result
b，cのような結果が得られた．分類に要した時間は線形カーネルが68ms，ガウスカーネルが115msであった．なお，各パターンに割り当てるクラスタは完全に収束しなかったため，1000個のパターンのうち割り当て先が変化したものが50個未満の場合に収束したとみなして処理を終了している．

線形カーネルにはハイパーパラメータはなく，線形分類器となるため@kernel_k_means_result
bのように内側と外側の区別ができていない．一方，ガウスカーネルはハイパーパラメータとして決定境界の分散（の逆数）$gamma$があり，値を試行錯誤的に調整することでクラスタリング結果が@kernel_k_means_result
c のように入力データと一致した．ただし，$gamma$に対する結果の変化が激しいため，膨大なデータに対するパラメータの調整は相当の時間が必要であると考えられる．

また，適切なカーネル関数を用いることで通常のK-meansでは分離できないデータに対してもクラスタリングを行うことができることがわかる．


#figure(grid(
  columns: 3,
  row-gutter: 0.5em,
  image("kernel_k_means_dataset.png"),
  image("kernel_k_means_linear_result.png"),
  image("kernel_k_means_gauss_result.png"),
  "a) 入力データ",
  "b) 線形カーネル",
  "c) ガウスカーネル",
), gap: 1em, caption: [Kernel K-meansによるクラスタリング結果]) <kernel_k_means_result>