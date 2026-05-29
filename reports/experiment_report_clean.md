# 实验报告：局部质量、MI 与变分近似诊断

## 0. 概览

本报告分析三组实验结果：

| 实验 | 数据来源 | 核心问题 | 主要结论 |
|---|---|---|---|
| EXP1 | `results/exp1_small_bayes_regression` | Bayesian updating 是否保持非活跃坐标在 0 附近的局部质量阶 | 支持保持局部幂阶；posterior 主要改变常数因子 |
| EXP2 | `results/exp2_actual_vi_training` | global KL / ELBO 训练是否足以保证局部小质量区域被覆盖 | 不足够；diagonal Gaussian 明显漏掉局部 spike mass |
| EXP3 | `results/exp3_directional_normalisation` | directional normalisation 是否能识别局部 under-coverage | 能够准确识别；理论分类与数值分类完全一致 |

三个实验形成一条统一叙事：

1. Bayesian updating 在非活跃稀疏方向上基本保持 local mass order。
2. 普通 global VI 目标不能保证局部小质量区域被正确覆盖。
3. 基于局部质量比和方向性归一化的诊断量可以更直接地识别 local under-coverage。

---

## 1. EXP1：Small Bayesian Regression

### 1.1 实验目的

EXP1 检验在稀疏线性回归模型中，Bayesian updating 后 posterior 是否保持非活跃坐标在 0 附近的局部质量阶。

这里关注的是局部质量：

\[
P(|X|<r)
\]

当 \(r\to 0\) 时的幂阶行为。若

\[
p((-r,r)) \asymp r,
\qquad
q((-r,r)) \asymp r,
\]

则 prior 与 posterior 在 0 附近具有相同的 power-type local order。

### 1.2 实验设置

| 项目 | 设置 |
|---|---|
| 样本量 | \(n=100\) |
| 维度 | \(d=20\) |
| active coordinates | 4 |
| inactive coordinates | 16 |
| seeds | 0 到 4 |
| prior families | Gaussian, Laplace, Student-t |
| posterior approximation | MAP 周围的 Laplace approximation |
| 半径范围 | \(10^{-6}\) 到 \(10^{-0.5}\) |

需要注意：EXP1 使用的是 MAP 周围的 Laplace approximation，而不是 exact posterior。因此该实验应表述为近似 posterior 下的局部质量诊断。

### 1.3 Inactive coordinates 的主要结果

inactive coordinates 是 EXP1 的主证据，因为这些坐标对应真实稀疏方向。

| prior | mean log mass ratio | median mass ratio | mean abs MAP | mean posterior sd |
|---|---:|---:|---:|---:|
| Gaussian | 1.707 | 7.364 | 0.0437 | 0.0555 |
| Laplace | 1.771 | 6.171 | 0.0385 | 0.0510 |
| Student-t | 1.795 | 7.911 | 0.0436 | 0.0554 |

其中 mass ratio 指

\[
R_r=\frac{q((-r,r))}{p((-r,r))}.
\]

结果显示，posterior 在 inactive coordinates 的 0 邻域中拥有比 prior 更高的局部质量；典型比例约为 6 到 8 倍。但这主要是常数因子变化，而不是局部幂阶变化。

局部斜率的中位数如下：

| prior | prior median slope | posterior median slope |
|---|---:|---:|
| Gaussian | 1.000 | 1.000 |
| Laplace | 0.999 | 1.000 |
| Student-t | 1.000 | 1.000 |

因此 inactive coordinates 上的结果支持：

\[
p((-r,r)) \asymp r
\quad\Longrightarrow\quad
q((-r,r)) \asymp r.
\]

换言之，Bayesian updating 改变了 0 附近的局部质量规模，但没有改变其主要幂阶。

### 1.4 Active coordinates 的对照作用

active coordinates 的结果如下：

| prior | mean log mass ratio | median mass ratio | mean abs MAP | mean posterior sd |
|---|---:|---:|---:|---:|
| Gaussian | -319.189 | 0.000 | 1.396 | 0.0567 |
| Laplace | -326.052 | 0.000 | 1.408 | 0.0563 |
| Student-t | -320.412 | 0.000 | 1.408 | 0.0571 |

active coordinates 的 MAP 约为 1.4，明显远离 0。因此 posterior 在 0 邻域中的质量极小，很多半径下出现数值接近 0 的现象。

这部分不适合作为 zero-local-sparsity preservation 的主证据。它更适合作为对照：

- active coordinates 被数据推离 0；
- inactive coordinates 仍保持 0 附近的局部幂阶；
- 因此局部质量保持现象主要发生在非活跃稀疏方向上。

### 1.5 数值稳定性

Hessian diagnostics 如下：

| prior | median condition number | max condition number | status |
|---|---:|---:|---|
| Gaussian | 5.92 | 6.52 | positive definite |
| Student-t | 5.90 | 6.55 | positive definite |
| Laplace | 490.41 | 1330.34 | positive definite |

Laplace prior 的 Hessian condition number 明显更大。这很可能来自 Laplace prior 在 0 附近的尖点结构；即使代码中使用了平滑近似，数值条件仍比 Gaussian 和 Student-t 更差。

不过所有 Hessian 都是 positive definite，optimisation 全部 converged，linear algebra fallback 为 none。因此该问题不影响 EXP1 的主要结论，只需在论文中说明 Laplace case 的数值条件更差。

### 1.6 EXP1 结论

EXP1 支持如下说法：

> 在非活跃稀疏坐标上，Bayesian updating 后 posterior 的 0 邻域局部质量相对于 prior 发生常数级变化，但局部幂阶基本保持不变。

更适合写进论文的表述是：

> The inactive-coordinate diagnostics suggest that Bayesian updating preserves the local mass order near zero, while changing only the leading-scale factor. Active coordinates behave differently because the likelihood moves their posterior mass away from zero.

---

## 2. EXP2：Actual VI Training

### 2.1 实验目的

EXP2 检验 global KL / ELBO 训练是否足以保证变分分布覆盖目标分布中的局部 spike mass。

目标分布是 spike-and-main mixture。局部区域对应 spike component 附近的小球。实验比较两类变分族：

1. diagonal Gaussian；
2. two-component diagonal Gaussian mixture。

同时比较两个训练方向：

1. exclusive KL；
2. inclusive KL。

### 2.2 实验设置

| 项目 | 设置 |
|---|---|
| 维度 | \(d=2,10\) |
| spike weight | \(\epsilon=0.01,0.05,0.1\) |
| spike scale | \(\tau=0.05,0.1\) |
| variational families | diagonal Gaussian, two-component diagonal Gaussian mixture |
| objectives | exclusive KL, inclusive KL |
| evaluation samples | 262144 |
| local mass estimation | Sobol / QMC |

核心诊断量是局部质量比：

\[
R_r=\frac{q(B_r)}{p(B_r)}.
\]

若 \(R_r\approx 1\)，说明变分分布较好覆盖目标分布的局部 spike mass；若 \(R_r\ll 1\)，说明发生 local under-coverage。

### 2.3 Global KL 与 local mass ratio 的整体结果

| dimension | family | objective | mean \(D_{\mathrm{KL}}(q\|p)\) | mean \(D_{\mathrm{KL}}(p\|q)\) | mean local mass ratio | median local mass ratio | reliability |
|---:|---|---|---:|---:|---:|---:|---:|
| 2 | diagonal Gaussian | exclusive KL | 0.049 | 0.164 | 0.0416 | 0.0254 | 1.0 |
| 2 | diagonal Gaussian | inclusive KL | 0.058 | 0.154 | 0.0466 | 0.0337 | 1.0 |
| 2 | mixture-2 | exclusive KL | 0.000108 | 0.000107 | 0.994 | 0.995 | 1.0 |
| 2 | mixture-2 | inclusive KL | 0.000294 | 0.000267 | 0.988 | 0.983 | 1.0 |
| 10 | diagonal Gaussian | exclusive KL | 0.056 | 1.556 | 0.000 | 0.000 | 0.0 |
| 10 | diagonal Gaussian | inclusive KL | 0.112 | 1.497 | 0.000 | 0.000 | 0.0 |
| 10 | mixture-2 | exclusive KL | 0.000533 | 0.000547 | 1.002 | 1.004 | 1.0 |
| 10 | mixture-2 | inclusive KL | 0.001377 | 0.001386 | 0.976 | 0.975 | 1.0 |

### 2.4 Diagonal Gaussian 的局部覆盖失败

diagonal Gaussian 的 local mass ratio 非常低：

- 在 \(d=2\) 中，mean local mass ratio 只有约 0.04 到 0.05；
- 在 \(d=10\) 中，QMC 估计下 \(q(B_r)\) 出现 zero-count，local mass ratio 记为 0，reliability 也为 0。

这说明 diagonal Gaussian 虽然可以在 global KL 上取得看似不差的数值，但它不能有效覆盖目标分布中的 spike component。

尤其是 \(d=10\) 的结果最有说明力：

\[
D_{\mathrm{KL}}(q\|p)\approx 0.056,
\qquad
R_r\approx 0.
\]

这说明 global KL 小并不推出局部小质量区域被覆盖。

### 2.5 Mixture family 的恢复效果

two-component diagonal Gaussian mixture 的结果非常接近理想状态：

- \(d=2\) 中 local mass ratio 约为 0.99；
- \(d=10\) 中 local mass ratio 也约为 0.98 到 1.00；
- 所有 mixture case 的 reliability 都为 1。

learned spike weight 也基本等于目标 \(\epsilon\)：

| target \(\epsilon\) | dimension | objective | mean learned spike weight |
|---:|---:|---|---:|
| 0.01 | 2 | exclusive KL | 0.0101 |
| 0.01 | 2 | inclusive KL | 0.0099 |
| 0.01 | 10 | exclusive KL | 0.0100 |
| 0.01 | 10 | inclusive KL | 0.0099 |
| 0.05 | 2 | exclusive KL | 0.0501 |
| 0.05 | 2 | inclusive KL | 0.0502 |
| 0.05 | 10 | exclusive KL | 0.0499 |
| 0.05 | 10 | inclusive KL | 0.0498 |
| 0.10 | 2 | exclusive KL | 0.1003 |
| 0.10 | 2 | inclusive KL | 0.1008 |
| 0.10 | 10 | exclusive KL | 0.0996 |
| 0.10 | 10 | inclusive KL | 0.0982 |

这说明 mixture family 基本成功学习了目标分布的 spike component。

### 2.6 EXP2 的核心解释

EXP2 的主要对比不是 exclusive KL 与 inclusive KL 的细微差异，而是变分族容量的差异。

diagonal Gaussian 结构太弱，无法表达 spike-and-main mixture 的局部结构。因此它会将概率质量集中在主成分附近，并漏掉低权重 spike 区域。

mixture family 具有单独表示 spike component 的能力，因此可以恢复正确的局部质量比例。

### 2.7 EXP2 结论

EXP2 支持如下说法：

> Global KL or ELBO optimisation does not by itself guarantee local mass coverage. A variational family may achieve small global divergence while assigning almost no probability to a small but structurally important local region.

论文中可以把 EXP2 作为反例型实验：

\[
\text{small global KL}
\quad\not\Longrightarrow\quad
\text{accurate local mass coverage}.
\]

---

## 3. EXP3：Directional Normalisation

### 3.1 实验目的

EXP3 使用可控的一维密度族检验 directional normalisation 是否能识别 local under-coverage。

密度族为：

\[
\rho_a(x)\propto |x|^{a-1}\exp(-x^2/2).
\]

在 0 附近，有

\[
P_a((-r,r))\asymp r^a.
\]

因此 theoretical MI 为：

\[
\mathrm{MI}_{\mathrm{pow}}(P_a,0)=\frac{1}{a}.
\]

### 3.2 实验设置

| 项目 | 设置 |
|---|---|
| \(a_p\) | \(0.5,1,2\) |
| \(a_q\) | \(0.25,0.5,1,2,4\) |
| \(\alpha\) | \(0.3,0.5,0.8\) |
| 半径范围 | \(10^{-4}\) 到 \(10^{-0.5}\) |
| raw rows | 1800 |
| summary rows | 45 |
| failures | 0 |
| warning count | 0 |

理论判别规则是：

\[
a_q\le a_p
\quad\Longleftrightarrow\quad
\mathrm{MI}_q\ge \mathrm{MI}_p.
\]

当 \(a_q\le a_p\) 时，\(q\) 在 0 附近的局部质量阶不低于 \(p\)。当 \(a_q>a_p\) 时，\(q\) 在 0 附近发生 local under-coverage。

### 3.3 MI 估计精度

EXP3 的 MI 估计与理论值高度一致：

| \(a\) | theoretical MI | estimated MI |
|---:|---:|---:|
| 0.25 | 4.000 | 4.004 |
| 0.5 | 2.000 | 2.002 |
| 1.0 | 1.000 | 1.001 |
| 2.0 | 0.500 | 0.500 |
| 4.0 | 0.250 | 0.250 |

这说明该实验中的 slope-based MI estimation 是可靠的。

### 3.4 理论分类与数值分类

| \(\alpha\) | cases | preserved cases | non-preserved cases | classification match rate | max warning count |
|---:|---:|---:|---:|---:|---:|
| 0.3 | 15 | 9 | 6 | 1.0 | 0 |
| 0.5 | 15 | 9 | 6 | 1.0 | 0 |
| 0.8 | 15 | 9 | 6 | 1.0 | 0 |

全部 45 个 case 中，数值分类与理论分类完全一致。

### 3.5 Local mass ratio 的理论行为

由于

\[
p((-r,r))\asymp r^{a_p},
\qquad
q((-r,r))\asymp r^{a_q},
\]

所以

\[
\frac{q((-r,r))}{p((-r,r))}
\asymp r^{a_q-a_p}.
\]

因此：

| 情况 | local mass ratio 行为 | 解释 |
|---|---|---|
| \(a_q<a_p\) | \(R_r\to\infty\) | \(q\) 在 0 附近更重 |
| \(a_q=a_p\) | \(R_r\to 1\) | 二者局部阶相同 |
| \(a_q>a_p\) | \(R_r\to 0\) | \(q\) 在 0 附近 under-covers \(p\) |

数值结果与此完全一致。以 \(\alpha=0.5\) 为例：

| \(a_p\) | \(a_q\) | preserved? | min local mass ratio |
|---:|---:|---|---:|
| 0.5 | 0.25 | True | 1.406 |
| 0.5 | 0.5 | True | 1.000 |
| 0.5 | 1.0 | False | \(8.60\times 10^{-3}\) |
| 0.5 | 2.0 | False | \(5.39\times 10^{-7}\) |
| 0.5 | 4.0 | False | \(1.35\times 10^{-15}\) |
| 2.0 | 0.25 | True | 14.890 |
| 2.0 | 1.0 | True | 5.089 |
| 2.0 | 2.0 | True | 1.000 |
| 2.0 | 4.0 | False | \(2.50\times 10^{-9}\) |

### 3.6 Directional normalisation 的诊断作用

EXP3 计算了两个方向的归一化局部 RE-KL 诊断量：

\[
\frac{\mathrm{REKL}_{q,p}(B_r)}{p(B_r)},
\qquad
\frac{\mathrm{REKL}_{p,q}(B_r)}{q(B_r)}.
\]

结果显示，两个方向对 under-coverage 的敏感性不同。

当 \(a_q\le a_p\) 时，\(q\) 在 0 附近不比 \(p\) 少，最小 local mass ratio 至少为 1。同时，\(\mathrm{REKL}_{p,q}(B_r)/q(B_r)\) 在 preserved cases 中保持小于 1 左右。

当 \(a_q>a_p\) 时，\(q\) 在 0 附近低估 \(p\) 的局部质量。此时 \(\mathrm{REKL}_{p,q}(B_r)/q(B_r)\) 会显著增大，而 \(\mathrm{REKL}_{q,p}(B_r)/p(B_r)\) 可能仍然不大。

整体结果如下：

| case type | min local mass ratio | max \(\mathrm{REKL}_{q,p}(B_r)/p(B_r)\) | max \(\mathrm{REKL}_{p,q}(B_r)/q(B_r)\) |
|---|---:|---:|---:|
| preserved | 1.000 | \(7.58\times 10^7\) | 1.000 |
| non-preserved | \(1.35\times 10^{-15}\) | 1.000 | \(2.97\times 10^{15}\) |

这说明要诊断 \(q\) 是否在局部低估 \(p\)，方向性是关键。错误方向的比较量可能看起来温和，但实际 local mass ratio 已经趋近于 0。

### 3.7 EXP3 结论

EXP3 是三组实验中最干净的理论验证型实验。它支持如下结论：

> In the controlled one-dimensional family, MI preservation is exactly characterised by \(a_q\le a_p\), and the directional normalisation diagnostics correctly identify local under-coverage in all tested cases.

---

## 4. 三个实验的统一解释

### 4.1 从 Bayesian updating 到 VI failure

EXP1 与 EXP2 的关系是：

- EXP1 展示 Bayesian updating 本身可以在非活跃稀疏方向上保持局部质量阶；
- EXP2 展示 VI approximation 可能破坏局部质量覆盖，尤其当变分族无法表达目标局部结构时。

因此问题不在于 Bayesian updating 一定破坏 sparsity structure，而在于变分近似可能无法传递目标分布的局部形状。

### 4.2 从 global divergence 到 local diagnostics

EXP2 表明：

\[
\text{small global KL}
\quad\not\Longrightarrow\quad
\text{accurate local mass coverage}.
\]

EXP3 进一步说明，局部质量比和方向性归一化诊断量可以直接刻画这种失败。

### 4.3 适合论文中的主线表述

可以将三个实验组织成如下结构：

1. **EXP1: Bayesian updating preserves local mass order.**  
   In inactive sparse coordinates, posterior local mass near zero changes by a constant factor but keeps the same power order.

2. **EXP2: Global VI objectives may miss local mass.**  
   A diagonal Gaussian approximation can have small global KL but almost zero local mass in the spike region. A mixture variational family repairs this failure.

3. **EXP3: Directional local diagnostics identify under-coverage.**  
   In a controlled one-dimensional family, the criterion \(a_q\le a_p\) exactly matches MI preservation, and the directional normalisation diagnostics detect the failure cases.

---

## 5. 写作建议

### 5.1 EXP1 的写法

建议强调 inactive coordinates，不要把 active coordinates 当作主证据。

可写作：

> In the inactive coordinates, the posterior-to-prior local mass ratio remains positive and stable across small radii, while the estimated local slopes remain close to one. This indicates that Bayesian updating changes the local scale but preserves the local power order near zero.

### 5.2 EXP2 的写法

建议突出 diagonal Gaussian 和 mixture family 的对比。

可写作：

> The diagonal Gaussian approximation severely under-covers the spike region despite achieving moderate global KL values. In contrast, the two-component mixture family recovers both the local mass ratio and the spike weight, indicating that variational family expressiveness is essential for preserving local structure.

### 5.3 EXP3 的写法

建议把 EXP3 作为理论验证实验。

可写作：

> In the controlled density family \(\rho_a(x)\propto |x|^{a-1}\exp(-x^2/2)\), the estimated MI values agree with their theoretical values \(1/a\). The numerical classification of preservation versus under-coverage matches the theoretical criterion \(a_q\le a_p\) in all tested cases.

---

## 6. 最终结论

这三组实验共同支持以下结论：

1. Bayesian updating 在非活跃稀疏方向上可以保持 0 附近的局部质量阶。
2. Global KL / ELBO 数值较小并不保证局部小质量区域被正确覆盖。
3. 变分族的表达能力对局部质量覆盖至关重要。
4. 局部质量比与方向性归一化诊断量比单纯 global divergence 更适合发现 local under-coverage。
5. EXP3 的理论-数值一致性最强，适合作为核心验证实验；EXP2 适合作为反例型实验；EXP1 适合作为 Bayesian updating 下 local order preservation 的支持性实验。
