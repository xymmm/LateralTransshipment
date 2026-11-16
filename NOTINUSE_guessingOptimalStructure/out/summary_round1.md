# Structure of the optimal policy of SDP-2: first round

11 Nov 2026. Scripts in `structure/` (`common.py`, `structure_tests.py`, `policy_maps.py`), outputs in `structure/out/`.
Every number below is read from `out/tests_<instance>.json`, `out/tests_summary.md` or a direct evaluation of the stored solution.
Statements are observations on the six instances unless marked "provable".

## 1. What was computed

Six instances of test bed II with $T=4$, Poisson demand, $z=1$, $v=0.5$, $h=1$, $b=5$:

| Patterns | Expected demands of locations 1 and 2 | $(K,R)$ | Window of inventory levels | $\widetilde C_1(\vec i_0)$ |
|---|---|---|---|---|
| STAT–STAT | (7,7,7,7), (7,7,7,7) | (20,20) | $[-19,38]$ | 194.5782 |
| STAT–STAT | same | (50,2) | $[-30,60]$ | 258.5492 |
| LCY1–LCY2 | (2,5,10,11), (6,13,7,2) | (20,20) | $[-28,56]$ | 195.2299 |
| LCY1–LCY2 | same | (50,2) | $[-28,56]$ | 241.4939 |
| STAT–STAT/3 | (7,7,7,7), (2,2,2,2) | (20,20) | $[-19,38]$ | 145.4252 |
| STAT–STAT/3 | same | (50,2) | $[-24,48]$ | 173.9287 |

The window is the adequate range of the paper. SDP-2 is solved on $[lo - T D,\, hi]$ so that no demand outcome is dropped on the window. On LCY1–LCY2 with $(K,R)=(50,2)$ the value function on the window is identical to the solve of the paper on the adequate range, and extending the upper end by 50% changes it by at most $1.1\times10^{-13}$.

Notation: $\vec i$ opening state, $\tilde{\vec i}=\langle i^1-W_t, i^2+W_t\rangle$ post-transshipment state, $\vec y=\tilde{\vec i}+\vec q_t$ levels after reordering, and
$H_t(\vec y)=z(y^1+y^2)+\mathbb E[\text{holding}+\text{backorder}+\widetilde C_{t+1}(\vec y-\vec d_t)]$, so that
$\dbtilde C_t(\tilde{\vec i})=-z(\tilde i^1+\tilde i^2)+\min_{\vec y\ge\tilde{\vec i}}\{K\,\mathbb 1(y^1>\tilde i^1)+K\,\mathbb 1(y^2>\tilde i^2)+H_t(\vec y)\}$.

All tests are tie-aware with tolerance $10^{-8}$. The routine for $K$-convexity returns exactly $K$ on the single-location value functions of every instance (20.0000 and 50.0000).

## 2. Second stage $\dbtilde C_t$

**2.1 Each location reorders below a threshold, given the inventory of the other location. Holds everywhere.**
For every fixed $\tilde i^2$ in the window, location 1 reorders if and only if $\tilde i^1\le s^1_t(\tilde i^2)$, and the same holds for location 2. No violation in any instance, period or value of the other inventory (6 instances, 4 periods, 2 locations).

**2.2 The thresholds are not monotone in the inventory of the other location when $K/R=25$.**
With $(K,R)=(20,20)$ the threshold is constant or non-increasing in the other inventory, with one exception (LCY1–LCY2, $t=1$, location 2: one increase). With $(K,R)=(50,2)$ it rises and falls. STAT–STAT, $t=1$: $s^1_1(\tilde i^2)=1,5,4,2,1,2$ at $\tilde i^2=2,6,10,14,18,22$, against the single-location threshold 2. There is no monotone switching curve between "reorder" and "do not reorder".

**2.3 Where both locations reorder there is one pair of reorder-up-to levels.** Provable: it is the minimiser of $H_t$. It is close to the single-location levels: STAT–STAT $(50,2)$, $t=1$: (25,26) against 25 and 25; LCY1–LCY2 $(50,2)$, $t=1$: (22,29) against 24 and 28.

**2.4 Where one location reorders, its reorder-up-to level depends on the inventory of the other location.**
- $(K,R)=(20,20)$, STAT–STAT, $t=1$: $S^1_1(\tilde i^2)=21$ for $\tilde i^2=5,9,\dots,29$, the single-location level, and it falls to 17 and 12 at $\tilde i^2=33$ and 37.
- $(K,R)=(50,2)$, STAT–STAT, $t=1$: $S^1_1(\tilde i^2)=48,44,40,36,32,28$ at $\tilde i^2=2,6,10,14,18,22$, so $S^1_1(\tilde i^2)+\tilde i^2=50$. The reordering location raises the total stock of the two locations to twice the single-location level of 25. It reorders for both locations and transships later. The same slope of $-1$ appears in LCY1–LCY2 (47, 43, 39, 35, 31, 27 at $\tilde i^2=4,\dots,24$) and STAT–STAT/3 (36, 32, 28 at $\tilde i^2=-4,0,4$).

In the maps this is the diagonal line of dots in row 2.

## 3. First stage $\widetilde C_t$

**3.1 A transshipment is never followed by reorders at both locations.** 0 opening states in every instance and period. Provable by an exchange argument: keep the levels $\vec y$ after reordering and reduce the transshipment. If $Q^1_t>W_t>0$ and $Q^2_t>0$, the action $W'=0$, $Q'^1=Q^1_t-W_t$, $Q'^2=Q^2_t+W_t$ reaches the same $\vec y$ and saves $R+vW_t$. If $Q^1_t\le W_t$, the action $W'=W_t-Q^1_t$, $Q'^1=0$, $Q'^2=Q^2_t+Q^1_t$ reaches the same $\vec y$ and saves $K+vQ^1_t$.

**3.2 The sender reorders after shipping only when $K$ is large relative to $R$.** With $(K,R)=(20,20)$ this never occurs. With $(K,R)=(50,2)$ it occurs in 58, 70, 47, 0 opening states of periods 1 to 4 in STAT–STAT, 25, 161, 100, 0 in LCY1–LCY2 and 61, 71, 37, 0 in STAT–STAT/3. Provable by the same exchange: it requires $Q^1_t>W_t$ and $K\ge R+vW_t$.
Example, STAT–STAT $(50,2)$, $t=1$, state (2,8): the optimal action ships both units to location 2 and reorders 40 at location 1, $\vec y=(40,10)$, with $\widetilde C_1=211.395$ against $\dbtilde C_1=212.037$ without the transshipment. The reorder of location 1 serves both locations in the same period, limited by the stock on hand at location 1.

**3.3 The states without transshipment do not form one interval along a line $i^1+i^2=n$, and the two directions are not ordered. False in general.**
Lines with a violation, periods 1 to 4:

| Instance | $(20,20)$ | $(50,2)$ | Lines tested |
|---|---|---|---|
| STAT–STAT | 0, 0, 0, 0 | 16, 17, 11, 0 | 111 and 177 |
| LCY1–LCY2 | 1, 0, 0, 0 | 13, 25, 16, 1 | 165 |
| STAT–STAT/3 | 0, 0, 0, 0 | 15, 15, 9, 1 | 111 and 141 |

Counterexamples:
- STAT–STAT, $(50,2)$, $t=1$, line $n=10$: no transshipment at (1,9); $W=2,3,4$ from location 1 to 2 at (2,8), (3,7), (4,6); $W=-5,-4,-3,-2$ at (5,5), (6,4), (7,3), (8,2); no transshipment at (9,1). The transshipment goes from the location that reorders to the location that does not.
- LCY1–LCY2, $(20,20)$, $t=1$, line $n=15$: no transshipment at (13,2); $W=6,7,8$ at (14,1), (15,0), (16,−1), all to $\tilde{\vec i}=(8,7)$; no transshipment and a reorder at location 2 at (17,−2) to (22,−7); $W=15$ at (23,−8), again to (8,7).
- Last period, LCY1–LCY2, $(50,2)$, $t=4$, line $n=-1$: no transshipment at (9,−10); $W=3,4$ at (10,−11), (11,−12); no transshipment and a reorder at location 2 at (12,−13) to (15,−16); $W=9$ at (16,−17). The target is $\tilde{\vec i}=(7,-8)$ throughout. This is a one-period problem, so the interval property fails even without a future.

**3.4 One target per line and direction when $K/R=1$, up to two when $K/R=25$.** With $(20,20)$ every line has a single post-transshipment state per direction in every instance and period. With $(50,2)$ some lines have two: STAT–STAT, $t=1$, $n=11$, direction 1 to 2: $\tilde i^1=0$ (ship everything, 3.2) and $\tilde i^1=28$.

**3.5 The second-stage value along a line has several local minima**: up to 4 strict local minima on one line, for example in STAT–STAT at $t=2$ under both cost settings.

## 4. $K$-convexity

Minimal constant $\kappa$ such that $f(x+a)+\kappa\ge f(x)+\frac ab\,(f(x)-f(x-b))$ along a lattice direction $d$, compared with $K(d)=K\times$ number of positive components of $d$. Directions (1,0), (0,1), (1,1), (1,2), (2,1), (1,3), (3,1), (2,3), (3,2), (1,4), (4,1). Number of failures:

| Instance | $H_t$: coordinates, all directions | $\dbtilde C_t$: coordinates, all directions | $\widetilde C_t$: coordinates, all directions |
|---|---|---|---|
| STAT–STAT (20,20) | 2/8, 2/44 | 2/8, 2/44 | 6/8, 18/44 |
| STAT–STAT (50,2) | 6/8, 6/44 | 0/8, 0/44 | 8/8, 8/44 |
| LCY1–LCY2 (20,20) | 6/8, 27/44 | 3/8, 11/44 | 8/8, 39/44 |
| LCY1–LCY2 (50,2) | 2/8, 2/44 | 0/8, 0/44 | 8/8, 8/44 |
| STAT–STAT/3 (20,20) | 5/8, 19/44 | 2/8, 9/44 | 8/8, 31/44 |
| STAT–STAT/3 (50,2) | 5/8, 5/44 | 0/8, 0/44 | 8/8, 8/44 |

- **Last period.** $H_T$ is convex ($\kappa=0$), and $\dbtilde C_T$ is $K$-convex in each coordinate and $2K$-convex along (1,1) in every instance. Provable: $H_T$ is separable.
- **$\widetilde C_t$ is not $K$-convex in either coordinate. False in every instance.** STAT–STAT $(50,2)$, $t=1$, direction (1,0): $\widetilde C_1(-1,0)=259.549$, $\widetilde C_1(0,0)=258.549$, $\widetilde C_1(50,0)=122.585$, so $\kappa=85.96>50$. A larger $i^1$ saves the reorders of both locations through transshipment. The largest value is $\kappa=101.46$ with $K=20$ (LCY1–LCY2 $(20,20)$, $t=3$, direction (0,1), states (−25,17), (−25,18), (−25,56)), which exceeds $2K+R$.
- **$\dbtilde C_t$ is $K$-convex in every tested direction when $K/R=25$, and not when $K/R=1$.** STAT–STAT/3 $(20,20)$, $t=1$, direction (0,1): $\dbtilde C_1(4,11)=116.090$, $\dbtilde C_1(4,12)=119.079$, $\dbtilde C_1(4,38)=136.713$, $\kappa=60.09>20$.
- **$H_t$ for $t<T$ is not $K$-convex in the coordinates** in any instance, because $\widetilde C_{t+1}$ is not. The usual induction therefore breaks at the transshipment step $\widetilde C_t=\min_W\{u(|W|)+\dbtilde C_t\}$.

## 5. Summary

| | Status |
|---|---|
| Threshold in the own inventory given the other location (2.1) | holds in all tests; no proof; a proof by $K$-convexity of $H_t$ is not available |
| One pair of levels where both reorder (2.3) | provable |
| Reorder-up-to level of one location independent of the other location (2.4) | false; slope $-1$ in the other inventory when $K/R=25$ |
| Monotone thresholds $s^j_t$ in the other inventory (2.2) | false when $K/R=25$ |
| No transshipment followed by two reorders (3.1) | holds in all tests; provable |
| Sender reorders only if $K\ge R+v|W_t|$ (3.2) | consistent with all tests; provable |
| No-transshipment interval along $i^1+i^2=n$, ordered directions (3.3) | false, even in the last period |
| One target per line and direction (3.4) | holds when $K/R=1$, false when $K/R=25$ |
| $K$-convexity of $\widetilde C_t$ in each coordinate (4) | false in every instance |
| $K$-convexity of $\dbtilde C_t$ (4) | holds in all tested directions when $K/R=25$; false when $K/R=1$ |

## 6. Limits of this round

- Six instances, $T=4$, Poisson demand, two cost settings. The observations that hold are not established beyond them.
- The two-dimensional test checks eleven lattice directions, which is necessary and not sufficient for $K$-convexity in $\mathbb R^2$.
- The definition of $K$-convexity in $\mathbb R^n$ attributed to Gallego and Sethi and the switching-curve structure of Abouee-Mehrizi et al. (2015) are used from memory here and must be checked against the papers before anything is written up.
- Ties are broken as in `lt_sdp.SDP2`: smallest reorder quantities, lowest $W$. The pair (25,26) in 2.3 is one of two tied pairs.
