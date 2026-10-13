## 研究問題的回答（v0.2 作者分析，依據上方自動產生的數字）

所有數字來自 `outputs/v02/summary.json`（seed 42、3000 句，除非另註明）與 `outputs/natural/natural_summary.json`。每個 run 同時報告 v0.1 stage（同一語料、同一 v0.1 pipeline）與 v0.2 posterior 的指標，所以表中 `v0.1` 行 = 該 run 的 v0.1 數字（legacy split 開啟的 `*_v01` run 才等於 v0.1 report 的主結果），`v0.2` 行 = 同一 gold 上的 v0.2 posterior。

### 與規格的一處已記錄偏離

生成參數是每個 class 的 **top-down expansion 分佈** `θ_C(e)`（Kurihara–Sato VB-PCFG），不是規格 §4/§5 的 bottom-up `θ_AB = P(C|A,B)`、`π_x = P(C|x)`。原因（`docs/v02_model.md` §1）：句子已觀測時，bottom-up labelling model 沒有 string likelihood——完整 rule inventory 下每棵樹的 labelling 總和恆為 1，不完整的 inventory 反而因「讓更多 labelling 合法」而得分，單一 class 的 `A+A→A` 文法最大化它；structure learning 就沒有訊號。v0.1 的 MDL data cost 本來就是 top-down（`N(C,e)·−log P(e|C)`），v0.2 是它的 Bayesian 版本。`P(C|A,B)` 與 `π_x` 由 expected counts 推導並照規格報告（`rule_conditional`、`type_mixture`，`Σ_C P(C|A,B)=1` 有測試）。其餘規格項目（finite K_max、sparse Inside–Outside、Dirichlet CAVI、ELBO + structure prior、greedy merge/split、ProbLog oracle、punctuation prior、adaptive constituent context 只作 proposal）均照做。

### RQ7（先答）：production inference 與 ProbLog / brute force 是否一致？

**一致。** `tests/test_chart_exact.py`：1 class / 2 classes / ambiguous lexical / multiple derivations / unknown token，含與不含 λ、ψ potentials，Python reference chart = brute-force enumerator（|Δ| < 1e-9：log Z、`P(Z_ij=C|s)`、`P(Q_ij=1|s)`、expected rule/lexical/root counts、Viterbi），numba backend = reference（1e-9），chunk layer = 分段枚舉。`tests/problog_reference/`（15 cases，含 shared-intermediate multi-hop）：ProbLog = chart = brute force（|ΔP| < 1e-8）。找到並修掉的真實 bug 全在 oracle 編碼側（root 必須是一個 annotated disjunction，否則不同 root 的 derivation 不互斥而少算；UNK disjunct 每 class 只能一個），production chart 沒有改過一行——這正是 oracle 的價值。ProbLog 不在 pipeline 任何路徑上（`problog_reference.enabled: false`，未安裝時測試 skip）。

### RQ1：hard/heuristic decisions → posterior marginal，class/constituency 是否更穩定？

**是，而且不靠任何 structure move。** Synthetic-1 `clean_v02_grammar_only`（只把 v0.1 文法當 warm start 跑 CAVI）：class F1 0.983 → **1.000**（8/8 class）、constituent F1 0.998 → 1.000、composition P 0.778 → **1.000**（R 1.0）。v0.1 的兩條 crutch relation `Det+Nn→Nn`（support 66）與 `Adj+Adj→Adj` 在 posterior expected count 下歸零（E[N] < 0.5，之後被 prune proposal 以 ΔF=+10.5 移除）；v0.1 的 Viterbi hard-EM 把它們鎖住，sum-product 沒有。seed 穩定性：v0.1 clean seed1/seed2 class F1 0.967/0.927、comp P 0.545/0.545 → v0.2 0.985/0.996、0.875/0.778；amb seed1/seed2 0.958/0.936 → 0.996/0.966。10k 句：v0.1 0.889（12 class，bare-S 類）→ v0.2 0.972（但 comp 0.625/0.714，見下）。

### RQ2：Inside–Outside 是否自然處理 structural ambiguity？

**是。** 不再有 heuristic confidence propagation：`structural_ambiguity.json` 的 hyperpaths 現在由 sum over derivations 吸收，constituency 是 chart marginal。證據：(i) constituent F1 在 clean/amb 達 1.000/0.995、noisy 0.902 → 0.951（+5 點，v0.1 的 Viterbi 分析在 filler 周圍常選錯括號）；(ii) 以 posterior > 0.5 解碼與以 MAP derivation 解碼的 constituent F1 在 clean 上相同（1.0），表示 posterior 很尖；(iii) 10k 句的 bare-S 類（`dog waits` 作 root）在 v0.2 一開始就沒有 extra root class（`extra_root_classes = []`）。

### RQ3：Dirichlet VB 是否能自然表示 lexical/category ambiguity？

**能表示，也能發現，但「發現」依賴 lexicon-add proposal，而 objective 也會接受 gold 不認的重新分析。**

- 表示：Synthetic-2 legacy split 關閉時，四個 planted 詞由 `lex_add` proposal 加入第二個 lexical entry（ΔF：walks +1623、light +1253、duck +881、fast +636 nats），posterior type mixture 變成 `walks = 0.55 Vt + 0.45 VP`、`fast = 0.62 Adj + 0.38 Adv`、`light = 0.65 Nn + 0.35 Adj`、`duck = 0.62 Nn + 0.21 VP + 0.17 NP`（gold 0.6/0.4、0.5/0.5、0.5/0.5、0.6/0.4）；occurrence 指派正確率 **0.948**（majority baseline 0.63；v0.1 legacy split 0.976）。這就是規格要的「ambiguity 由 q(Z_o=C) 自然表示」。
- 代價：同一個 objective 也以 ΔF ≈ +200 nats/詞把 bare noun（man、woman、horse…）加進 NP class——`Nn+VP→S` 與 `NP+VP→S` 在字串上 weakly equivalent，把 bare noun 當 NP 讓 S 的 expansion 分佈更尖。gold 把 bare noun 標為 Nn，所以 ambiguity type F1 只有 0.47–0.62（false: man/teacher/woman/horse/bird），clean 上也出現同樣的 6 個 false ambiguity。這是 model selection 的結果，不是 bug，我們**沒有**用加大 `lambda_lex` 把它壓掉（規格：不要盲目調參）；在自然語言裡 bare plural 本來就是 NP。
- `amb_v02_ms`（無 adaptive context）更保守：ambiguity F1 0.857、occ acc 0.933、0 false split，但 duck 沒切開（mixture Nn 1.0）。`amb_v02_grammar_only` / `ctx`（無 proposals）完全切不開（0.0 / 0.639）：**posterior 只能在 lexicon 已有兩個 entry 時表示 ambiguity，entry 本身必須由 proposal 提出**。legacy split 作初始化（`amb_v02_legacy_split`）結果與不用相近（occ acc 0.948），所以舊機制可以退場作 ablation。

### RQ4：class-level ELBO split/merge 是否減少 fragmentation / over-refinement？

**減少，且每一步都可審計。** 人工案例（`tests/test_cavi_structure.py`）：NP 被拆成 NP/NP2 → merge 接受（ΔF +38）；Adj+Adv 被合成一類 → split 接受（ΔF +468，兩半正好是 6 個形容詞 / 5 個副詞）；所有錯誤提案（split S 成 NP+VP / Nn+VP 兩半：ΔF −28；merge VP+Adv：−740；13 條 `rule_add`：各 −10）被拒；`monotonicity_violations = 0`。真實 run：Synthetic-2 的 duplicate S 類 `C4275+C4276` merge（+7.2）消掉 v0.1 的 bare-S 過細類；noisy 上 class F1 0.843 → **0.906**（`noisy_v02_ms`），fragmentation 清零；10k 句 0.889 → 0.972。沒做到的：noisy 的 composition precision 仍 0.18（filler 子串造成的 `S+Adv→S`、`Det+Adj→Det` 等 false relation，見失敗模式），而 10k 句少了 `Nn+VP→S`、`Vt+Nn→VP`（bare noun 被併入 NP 後這兩條被 prune）。

### RQ5：abstract constituent-context 作 proposal 是否緩解 sparsity 而不 self-reinforce？

**沒有可觀測的增益，也沒有 collapse。** `*_v02_ms`（無 adaptive）vs `*_v02_adaptive`：clean 完全相同（10 class、comp 1.0/1.0）；amb 0.995/0.5833 → 0.993/0.7778（comp P 升，但 ambiguity F1 0.857 → 0.615，因 adaptive 版接受了更多 bare-noun lex_add）；noisy 0.906 → 0.824（adaptive 版多接受一次 split，把 Nn 拆成兩個 fragment）。abstract feature 在 merge candidate 排名中確實出現（例如 VP vs Adv 的 raw similarity 0.96 但 abstract 0.58，所以該 merge 排到後面並被 ΔF 拒絕 −563），但決定權都在 ΔF；因此它既沒有 self-reinforcing（ELBO 不含它），也還沒證明有用。RQ5 的答案是「安全但目前多餘」。

### RQ6：punctuation structural prior 是否提高自然語料 stability / coverage？

見下方 TinyStories 段落（synthetic 文法沒有標點，`clean_v02_punct` 與 `clean_v02_adaptive` 結果相同，符合預期）。

### v0.1 已知 failure modes 逐項檢查（improved / unchanged / worse）

| failure mode | 結果 | 證據 |
|---|---|---|
| NP/Nn fragmentation | **improved** | clean 3k：v0.1 NP→2 fragments、Nn→2 → v0.2 每個 gold class 1 個 fragment（`diagnostics_v02.fragmentation`）；noisy：Nn 4 fragments → 1；10k：12 class → 8 主類 |
| bare-S over-refinement | **improved** | v0.1 10k 的 `Nn+VP` root class（1073 次）→ v0.2 `extra_root_classes = []`；Synthetic-2 的 duplicate S merge 接受 |
| Adj→Det contamination | **improved** | amb seed2：v0.1 Adj 併入 Det（fast 正確率 0.397）→ v0.2 所有 class purity ≥ 0.95 除一個 VP/Vt 混合類（0.863）；`fast = 0.60 Adj + 0.40 Adv` |
| private crutch rules | **improved on synthetic-1/2, visible elsewhere** | clean 3k 的 `Det+Nn→Nn` 歸零；private penalty 在 noisy/seed2 仍標出 self-paired rules（dominant share 0.63–1.0），這些 rule 都在 false relation 列表裡。已知瑕疵：報告的 run 對 `X+X→Y` 兩側各計一次（share 可到 2），已修正為每個 realisation 計一次，但 suite 未重跑 |
| filler-induced rules | **unchanged** | noisy：`uh/um/well` 仍自成一類（v0.1 併入 Adv），`filler_induced_rules` 4 條（filler share 0.51–0.71），comp P 0.27 → 0.18。posterior 不能解決「filler 是高頻規則子串」這個資料性質 |
| long-context sparsity | **unchanged（繞過）** | lexical context 只作 leaf potential；`grammar_only` 與 `ctx` 在三個語料上幾乎相同（clean 同、amb 同、noisy 0.866 vs 0.866），即 context 證據沒有超出 v0.1 已給的初始化；adaptive context 也沒補上（RQ5） |
| ambiguity pollution | **improved** | seed2 的鏈式失敗消失（purity 表）；但新的「bare noun → NP」false ambiguity 取而代之 |

### 新增的 failure modes（v0.2 特有）

1. **string-equivalent reanalysis**：objective 偏好把 bare noun 加進 NP（ΔF +200/詞），使 ambiguity precision 下降、`Nn+VP→S` 在 10k 被 prune。只有字串證據時這是 unidentifiable 的（v0.1 report 已指出）；v0.2 把它從「偶然」變成「系統性選擇」。
2. **random initialisation 不夠**：`random_k8/k12` class F1 0.91/0.82（64/139 class，大量 rule_prune 與 merge 之後仍碎），v0.1 warm start 是必要的（RQ1 的「posterior 更穩」是以 v0.1 初始化為前提）。
3. **adaptive split 在 noisy 上過度**：一次被接受的 Nn split（ΔF > 5）把 Nn 拆成兩個 fragment，class F1 0.906 → 0.824；`split_threshold` 5 nats 對 3000 句太低，但我們沒有為單一語料調它。
4. **效率**：structure learning 佔 v0.2 時間 80–90%（3k 句 160–300 s；10k 句 548 s；每個 proposal 要 5 次 E-step），`proposal_eval: affected` 在自然語料是必要的。

### 關鍵失敗案例（可在各 run 的 report.md「Structure moves」與「failure modes」段落逐條查到）

1. `amb_v02_adaptive`：`lex_add(man → NP)` ΔF +205.8（ELBO +207.8、prior −2.0），之後 `rule_prune` 把 `Nn+VP→S` 的 expected count 掃空。
2. `noisy_v02_ms`：filler 類 `C3138+C4171.2` 參與 4 條 filler-induced rule（`X + filler → X`），posterior 無法拒絕，因為它們確實壓縮語料。
3. `amb_v02_seed2`：Vt 碎成 2 個 class（0.80/0.20），merge candidate 被 ΔF 拒絕——posterior 認為它們的 composition-out 不同（其中一個是 `walks` 的 Vt sense 所在）。
4. `clean_v02_random_k12`：Det 碎成 2 類、`S+Adv→S` 等 false relation；merge/split 無法從隨機起點收斂到 8 類（40 個 accepted moves 後仍 139 類，多為 singleton 多詞 span）。

### 自然語言（TinyStories，10k 句，raw text only；RQ6 在此回答）

數字在上方「Natural-language study」段落與 `outputs/natural/`。v0.1 stage 完全相同（1270 accepted class、132 relation、135.6 bits/句），v0.2 由其 top-300 class（以 derivation usage 排序）warm start，chunk fallback 開啟、beam chart（`max_classes_per_cell 32`，mean retained mass 0.9997，max dropped 0.057）。總時間 1601 s（v0.2 部分 1480 s，其中 structure learning 1271 s）。

- **coverage**：只有 0.05% 的句子由單一 root 推導（MAP），posterior 平均 6.3 chunk/句（v0.1 Viterbi 5.6），unknown-token mass 19.8k/103k tokens（token 不在任何 class 的 lexicon 內，包含 v0.1 時 freq<3 的詞與 300 cap 之外的 class）。壓縮 110.9 bits/句（-ELBO/N）優於 v0.1 的 135.6 bits/句（MDL），但 fallback 的 chunk 序列本身便宜，兩者不可直接比。
- **complexity**：292 class、367 rule、756 lexical entry；21/67 個 structure move 被接受（2 merge、1 split、prune、`lex_add(Lily, play)`、`rule_add`）。
- **stability（兩個不相交 5k 子集）**：class pairwise F1 **0.327 / ARI 0.258**（v0.1 同一比較 0.224 / 0.222，1666 vs 2038 個共有分類 type），constituent-set agreement 0.655（v0.1 0.659），**rule agreement（class 對齊後 Jaccard）0.039**（378 vs 365 rule），ambiguity agreement 0.11。亦即 class 結構的可重現性略升，relation 幾乎完全不可重現——10k 句的自然語料下 (E,R) 中的 R 仍是 data-sparsity 的受害者（v0.1 RQ5 結論不變）。
- **punctuation prior（RQ6）**：三種模式——as token：parsed 0.05%、367 rule、110.9 bits；removed：parsed 1.0%、346 rule、108.6 bits；structural prior：parsed 0.0%、**254 rule**、149.1 bits。structural prior 把 rule 數壓低三成並提高子集 stability（class F1 0.357 / ARI 0.238、constituent agreement **0.732**、rule agreement 0.078，皆高於 as-token 的 0.327 / 0.258 / 0.655 / 0.039），但 bits/句變差 38 bits：ψ 是 virtual evidence，所有被懲罰的 span（跨逗號）付出 log-mass，所以目標值不可與其他模式比較；coverage 沒有提高。**結論：punctuation structural prior 提高 stability（尤其 constituent agreement），不提高 coverage。**
- **ambiguity**：291 個 type 的第二 class mass ≥ 0.2；最高 entropy 的 type（juice、walk、promise、work、kids…）全部是「主要 mass 在 `<unk>`-dominated class C20901」的低頻詞，這是 fallback class 的 artefact，不是語言學 ambiguity（規格要求不宣稱）。真正像 ambiguity 的例子在較高頻詞：`It`（v0.1 legacy split 也切過）在 v0.2 以 lexical entry 出現在主語代名詞類與 expletive 用法。
- **class 可讀性**：top class 與 v0.1 一致（句末標點、`and`、主語代名詞+人名、`the`、`,`、copula、`to`、`a`、人名、所有格、小寫代名詞），posterior 另外顯示 type mixture（`Lily` 0.73 在人名類、0.27 在 `He/She/They` 類；`was very` 0.55 在 copula 類）。
- **瓶頸**：(i) `<unk>` 類 C20901（mass 7.3k）吸收了所有 OOV 與被 cap 掉的詞並以 `C20901+C20901→C20901` 自我組合（807 次）——這是 v0.2 版的 filler 問題；(ii) 每個 proposal 的 affected-sentence E-step 在 10k 句仍需 10–30 s，4 輪 structure learning 已佔 21 分鐘；(iii) k_max=300 的 cap 是人為的（v0.1 有 1270 class），被 cap 的 class 的詞直接成為 `<unk>`。

### 下一步建議

1. **bare-noun / NP 的 identifiability**：加入「同一 token 在兩個 class 的 entry 必須有不同 composition 行為」的 prior（或直接把 `lambda_lex` 與 lexical reuse prior 綁定），否則 objective 會把每個名詞都做成 NP/Nn 雙 entry。
2. **`<unk>` 的分解**：以 character/morphology-free 的 frequency-tier 類（rare-word class per existing class，`C → <unk_C>`）取代單一 UNK expansion，避免 UNK 類自我組合成 filler。
3. **k_max 與 class cap**：改為把被 cap 的 v0.1 class 以 context similarity 併入最近的保留 class，而非丟棄。
4. **效率**：proposal 的 affected-sentence E-step 可以只在包含 moved class 的 *cell* 上重算（增量 inside-outside），預期 5–10×；expectation semiring 可省去 outside 的第二次掃描。
5. **adaptive context**：目前無益；若要保留，應只在 split proposal 缺 composition 證據時啟用，並以 ΔF 回饋選 feature。
6. **朝 CCG 走**：v0.2 的 posterior `P(C|A,B)` 與 rule expected counts 已是 CCG slash-category 因式分解需要的軟統計（modifier rule = 結果類等於一側；functor = 另一側），下一版可直接從 (E,R) 的 posterior 推 slash categories。
