## 研究問題的回答（作者分析，依據上方自動產生的數字）

所有數字來自 `outputs/summary.json`（seed 42，3000 句合成語料，除非另註明）。縮寫：class F1 = gold-constituent span types 的 pairwise F1；const F1 = span-level constituent F1（derivation level）；comp = composition relation R(A,B,C) 經 Hungarian alignment 後的 P/R/F1。

### 驗收狀態

| 門檻 | 結果 | 通過 |
|---|---|---|
| Synthetic-1 class pairwise F1 ≥ 0.95 | 0.983 (ARI 0.976, purity 0.984, 8/8 classes) | ✓ |
| Synthetic-1 constituent F1 ≥ 0.95 | 0.998 (len≥2: 0.999, precision 1.000) | ✓ |
| Synthetic-1 composition F1 ≥ 0.95 | 0.875 (recall 1.000 = 7/7 gold relations；precision 0.778：多出 `Det+Nn->Nn` (support 66, 來自 NP/Nn fragment 混淆) 與 `Adj+Adj->Adj` (support 2)) | ✗（差距來自 2 條低支持度 spurious relation，未調參壓掉） |
| Synthetic-2 ambiguous type F1 ≥ 0.90 | 1.000 (4/4, 0 false splits) | ✓ |
| Synthetic-2 occurrence assignment accuracy ≥ 0.90 | 0.978 (majority-sense baseline 0.628) | ✓ |

依規格「達不到不要盲目調參」：composition precision 沒有被門檻化掉，而是在下面交代原因。

### RQ1：只用 raw-text context P(context|span) 能否恢復 latent equivalence classes？

**能，但有三個前提，而且每個前提都是實驗逼出來的。**

1. **多尺度 context 必須經過 sparsity 校正。** 3000 句時 k=3,4 的 joint context 幾乎都是 unique（scale-1 可共享的 context mass 0.998，scale-4 只有 0.60）。直接把各尺度混成一個分佈做 Bhattacharyya，Vt / Adj / Adv 三類永遠合不起來（similarity 被稀疏尺度稀釋到 0.3 以下），而 `clean_sim_bc_uncorrected` 只到 class F1 0.886、238 個 class。用 split-half self-similarity 做 attenuation correction（sim = Σ_k sim_k / Σ_k √(self_k(x)·self_k(y))）之後是 0.983、8 個 class、0 個 big-class false merge。
2. **context range 要夠。** `clean_k1` 0.882 → `clean_k12` 0.937 → `clean_k1234` 0.983；Synthetic-2 上 ambiguity F1 0.857 → 1.000 → 1.000。k=1 時 VP 的 lexical 成員（sleeps）與 Adv（quickly）共享 `(noun _ EOS)` context（gold 不同類之間的最高 BC = 0.64，同類最低 0.71），要到 k≥3 才能分開（RQ5）。噪音語料例外：`noisy_k1` 0.869 > `noisy_k1234` 0.843——filler 使長 context 更稀疏。
3. **只有 clustering 不夠，稀有 type 要靠 grammar 自己貼標籤。** 不做 MDL（ablation A/B/C）時 class F1 只有 0.740、70 個 class：高頻 lexical 類完全正確，但稀有的多詞 span（`old man`、`young horse`）各自留成 singleton。開啟 MDL + predictive chart 後，這些 span 由 `Adj+Nn->Nn` 推導而得到類別，才到 0.983。

Baseline 比較（clean）：weighted Jaccard 0.590（對頻率差敏感）、Jensen-Shannon 0.844、Bhattacharyya+校正 0.983；JS 與 Jaccard 走 Python pair loop，慢 4 倍。

殘餘錯誤：(a) NP/Nn 小碎片的 false merge（30 件，都是 4 個以下 type 的小類，如 `the small dog`(NP) 與 `small dog`(Nn) 都只在主語位置出現）；(b) 噪音語料 filler `uh / well` 被併入 Adv、`big` 一度併入 Det（class purity 0.91）。

### RQ2：加入 A+B→C composition evidence 後，能否分開真正 constituent 與偶然 substring？

**作為加法式分數 (αS_ctx+βS_comp+γS_prod)：不能；作為 derivation（MDL elimination + predictive chart）：能。**

- Ablation A（只 context）、B（只 composition+productivity）、C（兩者相加）結果**完全相同**：class F1 0.740、70 個 class、len≥2 constituent precision 0.43、composition precision 0.02。原因：`NP+Vt`、`Vt+Det`、`Nn+Vt` 這類高頻非 constituent 的 context 一致性、composition coverage 與 productivity 都和真 constituent 一樣高（它們本來就是「規則的」字串），加法式分數分不開。
- Ablation D（+ complexity penalty = MDL）：len≥2 precision 0.43 → **1.000**、const F1 0.769 → 0.998、composition precision 0.02 → 0.78、class 數 70 → 8，所有 7 條 gold relation 全部恢復。機制不是「composition 分數高」，而是「有了真 class 與真 relation 之後，非 constituent class 對編碼語料沒有貢獻」：`NP+Vt->X` 一移除，所有句子仍由 `NP+VP->S`、`Vt+NP->VP` 推導，grammar bits 減少、data bits 不增。關鍵工程細節（都在失敗案例裡逼出來）：chart 必須允許 grammar 推導「不是 candidate span 的稀有 span」（否則 `sees the small cat` 不能當 VP 節點，`the dog sees` 反而靠覆蓋率勝出）；data cost 必須是由 derivation counts 計算的 closed form（否則 hard-EM 把先用到的分析鎖死）；rule elimination 只能在 predictive 階段做（否則 `VP+Adv->VP` 在稀有 span 無法使用的階段被誤刪）。
- 噪音語料：D 把 class F1 從 0.520 提到 0.843、const F1 0.744 → 0.903，但 composition precision 只有 0.27：`Det+Det->Det`(679)、`Adv+NP->NP`(77) 等 spurious relation 來自 filler（`uh the`、`the um`）與 fragment 混淆。
- **原則性限制：**(i) 只有字串證據時，`Adj+Nn->Nn` 與 `Det+Adj->Det` 在沒有 bare noun 的 grammar 裡是 weakly equivalent（生成同一字串集），MDL 任選其一（第一版 grammar 得到 `Det+Adj->Det`，composition F1 0.80）；加入 `Nn+VP->S`、`Vt+Nn->VP`（bare noun，對應自然語言的 bare plural）後才可辨識，但代價是 NP/Nn context 重疊造成上述碎片混淆。(ii) `dog waits` 這種 span 在 17% 出現是 bare-S constituent、83% 是 `the dog waits` 內的非 constituent：系統有時為它保留一個獨立 root class（10k 句時 1073 次使用），它是「過細但不錯」的分析（S_bare），卻算成 false relation；class prior 掃描顯示 c_class 12→100 對 3k 句反而變差（comp F1 0.875→0.80，13 class）、對 10k 句變好（0.63→0.80），**不是單調的**，所以不能靠調 prior 解決——這是 class identifiability 問題。

### RQ3：同一 surface expression 的 occurrence context 是否足以發現 category ambiguity？

**足以，前提是兩個 sense 所屬的 class 已經被其他詞建立起來。** Synthetic-2 的 4 個 planted 詞（duck Nn/VP、walks VP/Vt、fast Adj/Adv、light Adj/Nn）全部被切開且無 false split（type F1 1.000），occurrence 指派正確率 0.978（walks 0.987、duck 0.976、light 0.981、fast 0.965），majority baseline 0.628。
- 提案方式很關鍵：用「各 class 的 context profile（排除該詞自身）對每個 occurrence 算 likelihood」的 guided assignment 可行；無監督 2-means（`amb_E_kmeans_proposal`）**一個 split 都沒接受**（兩群對齊到同一 class）。
- context range：k=1 只有 0.857 / 0.819，k≤2 以上即 1.000 / 0.975。
- 失敗案例（seed 2）：`light`、`fast` 兩個 ambiguous 詞佔 Adj 8 個成員中的 2 個，Adj 的 context 分佈被 Nn/Adv context 污染，clustering 階段 Adj 被併入 Det；之後 `fast` 的 Adj sense 沒有 class 可對齊（accuracy 0.397）。這是「ambiguity 污染 class identifiability」的鏈式失敗，不是 split 機制本身的錯。

### RQ4：ambiguity split 能否被 MDL / grammar compression 客觀支持，而非 clustering artefact？

**能，而且 MDL 是唯一擋住 artefact 的東西。**
- `amb_E_no_mdl_criterion`（只用 context 準則）：接受 20 個 split，其中 16 個是 false（`small`、`quickly`… 因 topic-like context 變化被拆），type F1 0.333；加上 Δ_split = DL(H1) − DL(H2) − λ_split > 0 後：4/4、0 false。
- split 確實改善 grammar consistency：`amb_D_no_splitting` → `amb_E_full` 的 occurrence-level class F1 0.912 → 0.961，composition F1 0.632 → 0.667，bits/sentence 23.82 → 23.44（壓縮變好）。
- 必要的修正：若把 ambiguous 詞單一指派為 Nn，它的 VP-sense 句子會催生 `NP+Nn->S` 這種「只被該詞自己支撐」的 crutch relation，使 split 看起來沒有 DL 收益（修正前 seed 1/2 漏掉 duck/fast/light）。把「該詞佔用量 ≥ 50% 的 rule」從所有假設中排除後，seed 1 也達 1.000 / 0.979。這對應規格中「只被該 expression 使用的 branch 應受強 penalty」，只是作用在 relation 而非 class 上。

### RQ5：失敗時主要瓶頸是哪一個？

依證據排序：
1. **class identifiability（最主要）。** 所有 run 的殘餘錯誤都集中在 NP/Nn 碎片、bare-S class、以及 seed 2 的 Adj→Det 污染；composition precision 的損失幾乎全部可追溯到這裡。seed 間的變異（clean 0.927–0.983；amb occurrence accuracy 0.788–0.979）也來自同一來源。
2. **data sparsity。** n=1000：0.901（28 class）；n=3000：0.983；k≥3 context 在 3k 句時大多 unique；稀有多詞 span 只能靠 grammar 推導取得類別。但 n=10000 並未更好（0.889，12 class）：資料變多讓 bare-S class 的 data-term 優勢變大——所以 sparsity 不是唯一瓶頸。
3. **constituency criterion。** 字串 weak equivalence（attachment 方向）與「過細 class」問題無法由 raw text 解決；MDL 常數（c_class、c_rule）是明確的 prior，但實驗顯示它不是可靠槓桿。
4. **context representation。** VP/Adv 需要 k≥3（固定尺度）；自然語言中這類「只在更長距離才可分」的對立會更多——adaptive / unbounded context refinement 是下一步。
5. **ambiguity overfitting**：在 MDL 準則下已受控（0 false splits），但對上游 class 污染脆弱。

噪音語料（Synthetic-3）綜合了以上各點：class F1 0.843、const F1 0.903、relation recall 0.857 但 precision 0.273；filler 被視為 Adv、rare words 留成 singleton、topic 相關性本身沒有造成 false split。

### 自然語言（TinyStories，10k 句，raw text only）

本節的數字是 intrinsic 指標（stability、compression、held-out context prediction、productivity），**不是** syntax recovery accuracy；完整數字與 top classes / relations / splits 在上方自動產生的 "Natural-language study" 段落與 `outputs/natural/`。

- **規模**：10,000 句、103k tokens、vocab 4,747；14,754 個候選 span type（freq ≥ 3）、203k occurrences、602k unique contexts、702k span–context edges；7,266 個 provisional class → MDL 後 1,270 個 accepted class、132 條 relation；總時間 109 s（增量式 MDL closed form 之後；優化前 426 s，見 `docs/profiling.md`），peak memory 1.7 GB。剩下的瓶頸是 pure-Python CKY（約一半時間），50k/100k 句在 v0.1 仍未跑，留待 parser 加速後做 sample complexity。
- **lexical class 有明顯 signal**（人工檢查 top classes）：主語代名詞與人名 {He, They, She, Sammy, Lisa, John, …}、受詞代名詞 {them, him, it, everything}、小寫主語代名詞 {he, they, she}、所有格 {their, her, its, his}、冠詞 {the} / {a}、copula {was, is, felt, wasn't}、句末標點 {., !}、引號、連接詞 {and} 與 complementizer 片段 {that he, because they}、地點名詞 {kitchen, bowl, window, tree, sky, pond, park, …}。這些與 RQ1 的結論一致：k≤4 的 context 加上 sparsity 校正足以把高頻 lexical 類分開。
- **constituent 結構只恢復到局部**：MDL 樣本中 0 句被完整推導、每句 5.5 個 chunk、1,751 個 unk token；132 條 relation 多是兩三個 token 的局部組合（`was + so`、`tree + .`、`The + bird`、`She + felt`、`One + day ,`），而非完整的 NP/VP/S 層級。多詞成員混入了以相同邊界結尾的非 constituent（`He smiled and` 進入主語類、`to go outside and` 進入 `to` 類）：這正是 Synthetic-3 中 filler/substring 現象的自然語言版本，MDL 在 1,270 個 class 的規模下無法像合成語料那樣把它們全部清掉（RQ2 的機制依賴「真 class 與真 relation 先存在」）。
- **ambiguity**：722 個 type 被測試、2 個 split 被接受：`It`（`It was a …` 的 expletive 用法 vs `It had …` 的主語用法，split_gain +57 bits）與 `grass`（`on the grass .` vs `in the grass .`，+54 bits——後者看起來是地點類被切得太細的 artefact，而非 category ambiguity）。被拒絕的 top 候選（`.`、`They`、`he`、`she`、`so`）全部因「第二群對不上任何其他詞使用的 class」被擋下，符合設計。
- **stability**：同一語料換 seed → class pairwise F1 / ARI 1.0（pipeline 幾乎是 deterministic 的，seed 只影響 split-half 與 MDL 抽樣，所以這個數字沒有資訊量）；**兩個不相交的 5k 半份語料 → pairwise F1 0.22 / ARI 0.22**（2,038 個共有已分類 type），constituent-set agreement 0.66，split agreement 0。也就是說在 5k 句規模下，高頻 lexical 類以外的 class 結構不可重現——這是 RQ5 的 data sparsity 瓶頸在自然語言上的直接證據。
- **held-out context prediction（負結果）**：27,118 個 held-out occurrence 的 k=1 joint context，mean log2 P：class model −14.87、type model −14.62、background −14.42——背景分佈最好；rare types（freq<10）也是 type (−12.73) 優於 class (−13.80)。只有 6.0% 的 held-out context 是「type 沒見過、class 見過」。結論：以 (left, right) joint bigram 為單位的 context 在 10k 句下稀疏到任何條件化都比不上 unigram 背景；這個指標需要改成 one-sided / backoff context 才有鑑別力，v0.1 不據此宣稱 class 的泛化能力。
- **punctuation ablation**：去掉標點後 1,217 個 class、118 條 relation、123 bits/句（vs 136），結構相近；標點本身形成最高頻的類（`.` `!`）與最高支持度的 relation（`X + .`），是自然語料中最強的 boundary signal。

### 關鍵失敗案例（可在各 run 的 report.md 中逐條查到「why merged / why rejected / why split」）

1. **false merge（clean/amb/noisy 都有）**：`the small dog`(NP) ⟷ `small dog`(Nn)：兩者都只以 2–4 次出現在主語位置，k=1..4 context 完全相同 `(BOS _ sleeps)`；校正後 similarity 1.0。只有更多資料或更長 context 才能分。
2. **過細 class**：`dog waits / man runs`（Nn+VP）在 10k 句形成獨立 root class（1073 次使用），MDL 移除會多花 ~300 bits（data term），grammar 只省 22 bits。
3. **string-equivalent 分析**：無 bare noun 的 grammar 下 `Det+Adj->Det` 取代 `Adj+Nn->Nn`（兩者 DL 幾乎相同）。
4. **crutch relation 阻擋 split**：`NP+Nn->S`（只由 duck/light 的錯誤指派支撐）使 `duck` 的 split_gain = −102.8 bits；排除 private rules 後 +194.7 bits。
5. **class 污染**：seed 2 中 Adj 因 `light`+`fast` 的混合分佈被併入 Det，連帶 `Adj+Nn->Nn` 消失、`fast` 無法切對。
6. **噪音**：`uh the`、`the um` 等 filler 子串形成 `Det+Det->Det`(679)；`uh / well` 併入 Adv。

### 下一步建議

1. **class-level MDL merge/split**：目前 MDL 只刪 class/relation，不合併碎片；加入「合併兩個 class 是否縮短 DL」（含 parameter cost 的 Stolcke 式 prior）可直接處理 NP/Nn 碎片與 bare-S 過細 class。
2. **adaptive context refinement**：對 similarity 高但 composition 行為不一致的 class pair（VP vs Adv、NP vs Nn）自動擴展 context 直到可分，而非固定 k=1..4。
3. **多詞 span 的 occurrence-level ambiguity**，包括「constituent vs non-constituent」兩態（`dog waits` 案例），而不只 class A vs class B。
4. **把 ambiguous 詞的影響從 class profile 中隔離**（split 之後重算 class context profile 並回饋到 clustering），避免 seed 2 的污染鏈。
5. **效率**：CKY/MDL 用 numba 或 C++ 實作；predictive chart 加 beam/剪枝；這是 50k–500k 句自然語料的前提。
6. **朝 CCG 走**：(E,R) 已經給出 functor/argument 的候選（每條 A+B→C 中「結果類等於某一側」的 rule 是 modifier，否則其中一側是 functor），下一版可從 R 的結構因式分解 slash categories，而不需要任何 POS/treebank。
