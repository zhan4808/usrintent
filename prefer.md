## PREFER: Prompt Ensemble Learning via Feedback-Reflect-Refine

## Chenrui Zhang^1 B, Lin Liu^2 *, Jinpeng Wang^1 ,

## Chuyuan Wang^1 , Xiao Sun^1 , Hongyu Wang^1 , Mingchen Cai^1

(^1) Meituan Inc., Beijing, China (^2) Beijing Jiaotong University, Beijing, China
Bchenrui.zhang@pku.edu.cn, linliu@bjtu.edu.cn,{wangjinpeng04,wangchuyuan,
sunxiao10,wanghongyu15,caimingchen}@meituan.com
Abstract
As an effective tool for eliciting the power of Large Lan-
guage Models (LLMs), prompting has recently demonstrated
unprecedented abilities across a variety of complex tasks. To
further improve the performance, prompt ensemble has at-
tracted substantial interest for tackling the hallucination and
instability of LLMs. However, existing methods usually adopt
a two-stage paradigm, which requires a pre-prepared set of
prompts with substantial manual effort, and is unable to per-
form directed optimization for different weak learners. In this
paper, we propose a simple, universal, and automatic method
named PREFER(PRomptEnsemble learning viaFeedback-
REflect-Refine) to address the stated limitations. Specifically,
given the fact that weak learners are supposed to focus on
hard examples during boosting, PREFERbuilds a feedback
mechanism for reflecting on the inadequacies of existing
weak learners. Based on this, the LLM is required to automat-
ically synthesize new prompts for iterative refinement. More-
over, to enhance stability of the prompt effect evaluation, we
propose a novel prompt bagging method involving forward
and backward thinking, which is superior to majority voting
and is beneficial for both feedback and weight calculation in
boosting. Extensive experiments demonstrate that our PRE-
FERachieves state-of-the-art performance in multiple types
of tasks by a significant margin. We have made our code pub-
licly available^1.

## Introduction

```
Large Language Models (LLMs) have recently flourished
across a variety of fields, demonstrating unprecedented abil-
ities in myriad of complex tasks (Zhao et al. 2023b; Ouyang
et al. 2022). Trained with large-scale web data on massive
parameters, LLMs show emergent abilities beyond the orig-
inal linguistic competence (Wei et al. 2022a), which perform
tremendous versatility in both academia and industry. To
elicit the power of pretrained LLMs directly or adapt LLMs
to specific domains, various paradigms are proposed, includ-
ing prompt engineering (Qiao et al. 2022), p-tuning (Liu
et al. 2021), and LoRA finetuning (Hu et al. 2021), etc. Due
to the immense scale of the model parameters, finetuning on
all or even part of LLMs is costly and time-consuming. To
this end, as a simple and effective paradigm, prompt engi-
neering explores a fundamentally new way of invoking in-
*This work was done during the internship at Meituan.
```
(^1) https://github.com/zcrwind/PREFER

## { , }

## { , }

## { , }

```
LLM
```
```
Feedback
How to solve the rest?
```
```
1 2
```
```
4
```
```
5
```
```
6
```
```
How to solve?
```
```
How to solve issues
according to the situation?
```
```
3
```
```
Input Ground Truth
```
```
Hard Examples
```
```
Different strokes for
different folks.
Answer
```
```
7
```
```
Sick
```
```
Bug
```
```
Rainstorm
```
```
Reflect
```
```
Refine
```
```
Figure 1: High-level overview of feedback-reflect-refine
paradigm.ptdenotes the prompt at thet-th iteration.
```
```
trinsic knowledge and reasoning ability of LLMs based on a
pretrain-prompt-predict manner (Liu et al. 2023).
Though promising, the na ̈ıve prompting approaches are
afflicted by several limitations. As generative language mod-
els, LLMs’ output commonly has a large variance. For in-
stance, the reasoning logic and predicted results could be
contradictory in multiple runs, although the input prompts
are fixed. In addition, LLMs suffer from the notoriously hal-
lucination issue (Ji et al. 2023), leading to results that are
plausible-sounding but factually incorrect or irrelevant to the
inputs. Furthermore, the quality of LLMs’ output is suscep-
tible to the given prompts, which entails substantial manual
effort and domain expertise to find out the reliable prompts.
As a promising solution to these issues, prompt ensem-
ble learning has attracted substantial interest in the commu-
nity very recently, demonstrating significant improvements
in both effectiveness and stability across various tasks. As
a representative work, PromptBoosting (Hou et al. 2023)
applies the traditional ADABOOST(Freund and Schapire
1997) algorithm over a set of pre-defined prompts for text
classification. BPE (Pitis et al. 2023) focuses on Chain-of-
Thought (CoT) (Wei et al. 2022b) boosting and builds few-
shot CoT prompts based on self-consistency (Wang et al.
2022). These efforts empirically demonstrate the strength
of prompt ensembles for LLM-based tasks, yielding excep-
```
# arXiv:2308.12033v1 [cs.CL] 23 Aug 2023


tional performance gains over single-prompt baselines.
However, despite their success, existing prompt ensem-
ble approaches, which typically adopt a two-stage process,
have several limitations. First, they require a pre-prepared
set of prompts in advance, which are either manually de-
fined or generated by another language model with heavy
parameters. This preliminary work is costly and laborious,
often involving a trial-and-error or pre-evaluation process to
ensure the quality of pre-defined prompts. Second, the two-
stage paradigm fixes the prompts to be used in the ensemble
process, limiting the adaptability and scalability of prompt
boosting, as the prompts cannot be optimized jointly. Since
the relationships between prompts are ignored during the
iterative boosting process, the pre-defined prompts tend to
be sub-optimal and susceptible. Moreover, existing methods
conduct ensembles either in boosting or in bagging individ-
ually, neglecting the potential benefits of combining the two
worlds to enhance performance.
To alleviate the above issues, we advocate that a smarter
paradigm for prompt ensemble in the era of LLMs is ex-
pected to be automatic, self-adaptive and joint-optimizable.
Such paradigm reduces the need for manual effort and do-
main expertise, as well as takes prompt relations into consid-
eration for directed optimization. Accordingly, we propose
a simple, automatic and universal approach called PREFER
(PRomptEnsemble learning viaFeedback-REflect-Refine),
towards a more effective prompt ensemble via utilizing
the generative and reflective capabilities that LLMs excel
at (Madaan et al. 2023). As shown in Figure 1, our PREFER
adopts afeedback-reflect-refinecircle for prompt boosting.
Concretely speaking, inspired by the fact that weak learn-
ers pay more attention to hard examples via weight redis-
tribution during boosting, we propose to transfer this hard-
sample-oriented weighting into nature language feedback,
which returns error information to the LLM for reflection.
Hence, considering the reflection information, the LLM per-
ceives the inadequacies of existing prompts and is able to
generate new prompts to refine them purposefully. Attribute
to the feedback-reflect-refine path, the LLM jointly opti-
mizes the downstream tasks solving and prompt generation
in an automatic manner. Iterating along this path, potential
conflict and redundancy among prompts are reduced, which
is vital for building a more stable and faster learner.
Furthermore, to adequately unleash the ability of each
prompt and further enhance the stability during boosting,
we propose a bilateral bagging approach, which incor-
porates forward and backward thinking for multi-source
verification. Specifically, drawing inspiration from human
decision-making, wherein uncertain answers are often re-
solved through a process of elimination, we instruct the
LLM to compute a confidence score for each response and
subsequently filter out the most uncertain answers. Given
the observed tendency of LLMs to overestimate confidence
in their predictions (Zhao et al. 2021), our bilateral bag-
ging approach assesses the responses from both forward and
backward directions, in which the overconfidence bias can
be counteracted subtly. The empirical results demonstrate
the superiority of our bilateral bagging approach compared
to other regular methods such as majority voting in both ef-

```
fectiveness and efficiency.
We conduct extensive experiments and in-depth case stud-
ies on a number of tasks, including reasoning, topic classifi-
cation, hate speech discrimination, etc. The empirical results
testify the effectiveness of our PREFERapproach. Moreover,
PREFERshows superiority in both stability and efficiency
compared to existing approaches. We will provide the source
code for reproducibility in the supplementary material.
```
## Related Work

```
Our work is conceptually related to several subareas of arti-
ficial intelligent, including Large Language Models (LLMs),
prompt engineering, and prompt ensemble learning. In this
section, we briefly review the works in each subarea.
```
### Large Language Models

```
Nowadays, Large Language Models (LLMs) have made rev-
olutionary progress and posed significant impact on various
artificial intelligent community (Zhao et al. 2023b; Ouyang
et al. 2022). According to the scale law, LLMs demonstrate
unprecedent power (called emergent abilities) with the rapid
growth of model parameters and data volume (Wei et al.
2022a). For instance, the most prominent applications in-
cluding ChatGPT and GPT-4 (OpenAI 2023) have shown
surprising reasoning ability, human-like conversation skills,
as well as a rich reserve of factual commonsense. Based on
the surprising emergent abilities, a series of classical algo-
rithms can evolve to a more intelligent version. In this paper,
we provide a pilot work on ensemble algorithm as a prelim-
inary study. We believe that our proposed approach could
not only simply serve as a strong baseline to foster future
research on prompt ensemble, but also shed light on the po-
tential research direction towards improving classical algo-
rithms with the power of LLMs.
```
### Prompt Engineering

```
In order to invoke the power of LLMs, a series of ap-
proaches have been proposed in the community, including
parameter-efficient fine-tuning (Hu et al. 2021; Liu et al.
2021) and prompt engineering (Qiao et al. 2022; Liu et al.
2023), etc. Due to the heavy weight of LLMs, fully or even
partly fine-tuning them is expensive and inefficient. Accord-
ingly, as an out-of-the-box paradigm, prompt engineering
(aka prompting) has emerged as a new approach for adapting
pretrain-prompt-predict path for downstream tasks. Tremen-
dous cutting-edge effort has been made towards this area to
improve the performance of prompting. Concretely, prompt-
ing adopts natural language as additional inputs, acting as
instructions or hints to LLMs. For example, GPT2 (Rad-
ford et al. 2019) allows for unsupervised learning of LLM
on multiple tasks through handcrafted task-specific prompts.
However, building prompts manually can be expensive, bi-
ased and sub-optimal (Liu et al. 2023). Another line of
works are devoted to conducting prompting in an automatic
way. STaR (Zelikman et al. 2022) utilizes a simple loop to
bootstrap LLMs with a self-taught manner, in which Chain-
of-Thought (CoT) (Wei et al. 2022b) rationale is iteratively
generated to hint the question answering process. Closer to
```

```
Feedback
```
```
weight update
```
```
Refine
```
```
LLM
```
```
Bilateral
Bagging
```
```
Boosting
{ , } { , }
```
```
For , succeed, but failed. Iteration
How to solve the rest? Prompt Weight
Boosting Error
Instance Weight
```
```
Bilateral
Bagging
```
```
weight update
```
```
Bilateral Prompt Bagging
```
```
contains confusing words.Too coarse description. Reflect
No guidance for evidence...
```
Figure 2: The pipeline of PREFER. Given the initial promptp 0 , LLM partially solves the problem via incorporating backward
thinking. Then the error information will be used for prompt optimization through the feedback-reflect-refine process. Iterating
this process and finally ensembling prompts based on evolved weights.

our work, APO (Pryzant et al. 2023) iteratively optimizes the
single prompt in a feedback manner, which treats the textual
reflection information as gradient in classical deep learning.

### Prompt Ensemble Learning

Prior studies have proven that LLMs have multiple reason-
ing paths for a single problem, which could lead to dis-
tinct outputs from identical inputs (Wang et al. 2022). To
this end, prompt ensemble learning has been presented as a
solution, which combines several individual prompts to ob-
tain better stability and generalization performance. Boost-
ing and bagging are two typical ensemble methods widely
adopted in numerous classical tasks, while their adaptation
on LLMs is still in its infancy. Current works for prompt
boosting typically utilize a two-stage paradigm. Prompt-
Boosting (Hou et al. 2023) has done a preliminary trial on
this way, which conducts the traditional ADABOOST(Fre-
und and Schapire 1997) algorithm over a pre-defined prompt
set for text classification. On the other hand, existing prompt
bagging approaches mainly rely on regular majority voting,
which can be computationally intensive. Notably, BPE (Pitis
et al. 2023) focuses on constructing few-shot CoT prompts
based on self-consistency (Wang et al. 2022), which offers
better performance than a single prompt in the case of in-
troducing exponentially additional computation. In this pa-
per, we propose a computation-efficiency prompt bagging
approach inspired by the human ethology, which incorpo-
rates prompt boosting for further performance improvement.

## Our PREFERApproach

### Preliminaries

In this section, we introduce preliminaries of our PREFER
approach, including the problem formulation and the dis-
mantling of key components.
Considering a reasoning or classification task driven by
LLMs, given the training dataDtr =

#### S

i{(xi,yi)}, the
goal of the proposed PREFERis to automatically construct a
prompt setP=

#### S

```
t{pt}along with prompt weights
```
#### S

t{λt}
via LLM-augmented ensemble learning, which can then be
utilized cooperatively for the subsequent inference. Here

```
xi ∈ X denotes the input texts andyi ∈ Ydenotes the
output label. It is noted that an initial promptp 0 is provided
as the seed for the subsequent iteration. Instead of requiring
any supervised fine-tuning (SFT) or reinforcement learning,
our proposed PREFERutilizes out-of-box LLM API (e.g.,
ChatGPT or GPT-4) as the foundation modelMfor uni-
versality and flexibility. As illustrated in Figure 2, our PRE-
FERmainly contains two components, i.e. feedback-driven
prompt boosting and bilateral prompt bagging, which will
be elaborated in sections below.
```
### Prompt Boosting via Feedback-Reflect-Refine

```
Before delving into the technical details of the proposed
prompt boosting approach, we first provide our design
principle, based on the thinking about what characteristics
should an intelligent prompt boosting have in the era of
LLMs. Review that boosting algorithms combine several in-
dividual weak learners to obtain better generalization per-
formance. Considering the fact that weaker learners are sup-
posed to pay more attention to hard samples during boost-
ing, we advocate that an intelligent boosting algorithm is
expected to understand what problems the previous weak
learners cannot solve. That is, instead of building prompts
individually, the relation among prompts should be consid-
ered for better performance and faster convergence. In an-
other vein, to reduce the manual effort, the prompt boost-
ing process should be automatic, where each prompt can be
constructed without manual intervention. Furthermore, the
prompt boosting should be universal and adaptive, for em-
powering any prompting-based task with the superiority of
ensemble learning seamlessly.
Our proposed PREFERembraces all the above design
principles, towards a simple, automatic and adaptive prompt
ensemble paradigm. Inspired by the classical boosting al-
gorithm such as ADABOOST(Freund and Schapire 1997)
and iterative prompting algorithms (Pryzant et al. 2023), we
adopt an iterative manner to build the prompt set where each
prompt is treated as a weak learner. As illustrated in Fig-
ure 2, acting as a weak learner, each prompt can only han-
dle part of the instance space, where new prompts will be
added to expand the solving space by introducing more in-
```

Listing 1:solving prompt

# Task
Given two sentences, determine whether
sentence 2 provides an answer to the
question posed by sentence 1.

# Output format
Explain your reasoning process in one
sentence and Answer "Yes" or "No" as the
label.

# Prediction
Sentence 1: {text1}
Sentence 2: {text2}
Label:[]

Listing 2:feedback prompt

I’m trying to write a Textual Entailment
task prompt. My current prompt is: {prompt}
But this prompt gets the following examples
wrong: {error_info}

Give {num_feedbacks} reasons why the prompt
could have gotten these examples wrong. Wrap
each reason with <START> and <END>.

formation. Based on the error-ambiguity decomposition of
ensemble learning (Opitz and Shavlik 1995), the ensemble
error mathematically contains two parts:

```
Eensemble=E ̄−A ̄ (1)
```
whereE ̄andA ̄respectively denote the average error and the
average ambiguity (also called diversity) of individual weak
learners. Based on Eq.(1), the ensemble performance is pos-
itively correlated with both the accuracy and diversity of
weak learners. Considering this requirement, the prompt in
each iteration is supposed to focus on the hard examples that
the prompts in previous iterations cannot handle. Inspired by
the way human reflect and refine for improving performance
when tackling difficult tasks, we propose a feedback-reflect-
refine pipeline, asking the LLM to consider the relation of
prompts in the iteration, generate new informative prompts,
and optimize them jointly.
Concretely speaking, we define two types of prompt tem-
plates, namely thesolving promptand thefeedback
prompt, which are respectively responsible for solving
downstream tasks and conducting the feedback process. Fol-
lowing In-Context Learning (ICL) (Dai et al. 2022), we
format both types of prompts with the component of the
instruction, demonstration and output format. Exemplary
cases of these two templates are illustrated in Listing 1
and Listing 2, respectively. Given the initial seed promptp 0
and the corresponding performance, we build the feedback
prompt based on the feedback template and the wrong exam-
ples. This is reminiscent of the gradient in deep learning op-
timization, which indicates the direction of model optimiza-
tion, the key difference lies that the feedback form changes
from numerical into textual. The feedback prompt will then
be fed to the LLMMfor self-reflecting, andMprovides a

```
series of reasons why the current promptptcan solve some
examples well but not others. Based on the reflection, the
LLM is asked to generate new prompts in connection with
hard examples specified in the previous iteration. In detail,
the sampled wrong examples and corresponding textual la-
bels are combined toerrorinfoin Listing 2. Mathemat-
ically, this feedback-reflect-refine process can be formulated
via the Bayesian theory:
```
```
P(pt|X,Y,pt− 1 ) =P(Rt|X,Y,pt− 1 )·P(pt|Rt) (2)
hereRtdenotes the reflection of the LLMMat thet-th iter-
ation. It is noted that our PREFERonly modifies the instruc-
tion of thesolving prompt, while other parts remain
unchanged.
Close to our work, APO (Pryzant et al. 2023) also con-
ducts a feedback-based mechanism for prompt optimization.
Nevertheless, there are several intrinsic differences between
such iterative prompting approach and our PREFER. First,
APO aims to search for a single prompt covering the largest
possible solution space, while our PREFERorganizes a set
of prompts via ensemble learning, which works in tandem
to cover multiple sub-spaces. Second, our PREFERproposes
an effective bagging approach to reduce the variance of the
LLM, which is superior to the regular techniques such as
beam search or Monte Carlo search in APO. Experimental
results demonstrate that our PREFERoutperforms APO by a
quite large margin with less computational cost and higher
stability.
```
### Bilateral Prompt Bagging

```
As shown in Eq.(1), the quality and stability of weak learn-
ers is essential to the ensemble performance. Due to the
generative property of language model, LLMs’ outputs are
highly sensitive to the input prompts, which affects the sta-
bility of both the feedback and weight calculation process.
To alleviate this issue, direct solutions include majority vot-
ing or beam search, which is commonly used in the commu-
nity (Wang et al. 2022; Li et al. 2023). However, these meth-
ods are computationally intensive, especially for LLMs with
massive parameters. Accordingly, to enhance the ability and
stability of each prompt with limited calculation burden, we
further propose a bagging approach calledbilateral prompt
bagging, which draws inspiration from human behavior of
utilizing forward and backward thinking for tackling diffi-
cult tasks.
Concretely speaking, humans commonly adopt the pro-
cess of elimination when they are not sure about the decision
making. Inspired by this, we advocate that similar spirits
can be utilized in the prompt bagging. In each iteration, the
LLMMis required to evaluate its answer’s confidence by
utilizing the generated promptptfollowed by a confidence
evaluation clause. When the evaluation result is not confi-
dent enough, the reverse thinking takes effect via conduct-
ing elimination process. In detail, we consider the quantita-
tive confidence score evaluation in both forward and back-
ward thinking. Take the classification task as an example, in
the forward evaluation,Mis required to measure the confi-
dence that each candidate answer is the correct one. As for
the backward evaluation,Mis required reversely to measure
```

Algorithm 1: Our PREFERAlgorithm

Input: Training dataDtr=

#### S

i{(xi,yi)}, the LLMM, the
seed promptp 0 , the prompt templatesTsolvingandTfeedback
Output: the result prompt setP=

#### S

S t{pt}and their weights
t{λt}, the reflection set

#### S

t{Rt}
1:Set the initial data weight toω(0)i = 1/|Dtr|,∀i ∈
{ 0 ,···,|Dtr|},P={p 0 }.
2:fort= 0toNdo
3: ift > 0 then
4: Generate newptwith{M, reflectionRt− 1 }
5: end if
6: Solve target tasks with{pt,Tsolving,ωi}
7: Conduct bilateral bagging
8: Build feedback promptwith {errorinfo,
Tfeedback}
9: Perform feedback and get the reflectionRt
10: Compute weighted error as Eq.(4)
11: Update the weight onptby Eq.(5)
12: Update the instance weights inDtrby Eq.(6) fol-
lowed by re-normalization
13: P=P ∪pt,R=R∪Rt
14: end for
15: return

#### S

```
t{pt},
```
#### S

```
t{λt},
```
#### S

```
t{Rt}
```
the confidence that each candidate answer is excluded. For
notational simplicity, we name the confidence scores corre-
sponding to the forward and backward evaluations withS+
andS−respectively. After these, the final probability can
be calculated via combiningS+andS−with a subtractive
fashion:

```
yˆ= arg maxj
```
```
eS
```
```
+j−S−j
PK
ce
```
```
Sc+−S−c
```
#### (3)

hereˆydenotes the predicted answer,candjdenote the
indexes of candidate answers. It is noted that LLMs tend
to evaluate confidence score overconfidently (Zhao et al.
2021), while our proposal ingeniously circumvents this in-
adequacy via positive and negative offsets. We believe that
such paradigm can also shed light on the community of
LLMs’ calibration (Zhao et al. 2023a).
Attributed to the introduction of reverse thinking mecha-
nism, the accuracy-versus-efficiency dilemma can be largely
alleviated for prompt bagging. Experimental results explic-
itly manifest that such bilateral bagging outperforms regular
methods (e.g., majority voting) in both effectiveness and ef-
ficiency.

Overall Algorithm To sum up, we conclude the proposed
PREFERin Algorithm 1. Basically, our PREFERfollows the
pipeline of the classical ADABOOST(Freund and Schapire
1997) algorithm, while enhancing it with thefeedback-
reflect-refine boostingand thebilateral prompt bagging.
Both branches can co-adapt and cooperate for automatic
prompt set optimization. In detail, the weighted ensemble
error in thet-th iteration is calculated as:

```
error(t)=
```
```
|DXtr|
```
```
i=
```
```
ω(it)·I
```
#### 

```
yi̸=M(pt,xi)
```
#### 

```
P|Dtr|
i ωi
```
#### (4)

```
hereIis the identify function. Moreover, the weight in each
iteration is updated based on the above error information as:
```
```
λ(t)= log
```
```
1 −error(t)
error(t)
```
```
+ log
```
#### 

#### |Y|− 1

#### 

#### (5)

```
Finally, the instance weights in training datasetDtrcan be
updated by:
```
```
ωi(t)=ω(it−1)·exp
```
#### 

```
λ(t)·I
```
#### 

```
yi̸=M(pt,xi)
```
#### 

#### (6)

```
here∀i ∈ { 0 ,···,|Dtr|}is the index of training exam-
ples. Once the process of Algorithm 1 is complete, opti-
mized prompts
```
#### S

```
t{pt}along with their weights
```
#### S

```
t{λt}can
be obtained, which can then be utilized for application via
weighted decision making. Moreover, the intermediate re-
flection
```
#### S

```
t{Rt}naturally provides abundant interpretability
for prompt boosting.
```
## Experiments

### Experimental Settings

```
Datasets We conduct experiments on a wide range of tasks
including natural language inference and classification:
```
- Natural Language Inference
    SNLI(Bowman et al. 2015),MNLI(Williams, Nangia,
    and Bowman 2017), andRTE(Dagan, Glickman, and
    Magnini 2005): textual entailment inference;
    QNLI(Rajpurkar et al. 2016): question-answering infer-
    ence.
- Natural Language Classification
    Ethos (Mollas et al. 2020): hate speech detection;
    Liar(Wang 2017): fake news classification;
    ArSarcasm(Farha and Magdy 2020): Arabic sarcasm de-
    tection.

```
Compared Baselines To manifest the superiority of our
PREFERapproach, we compare it with several state-of-
the-art baselines. As the closest work to our proposal,
PromptBoosting (Hou et al. 2023) conducts the traditional
ADABOOSTalgorithm over a pre-defined prompt set for text
classification. As a remarkable work of iterative prompting
methods, APO (Pryzant et al. 2023) utilizes an iterative man-
ner for optimizing a single prompt, where the performance
of the previous prompt will be used to form a natural lan-
guage “gradient” that guides the prompt optimization. More-
over, we also conduct single-prompt and Chain-of-Thought
(CoT) enhanced single-prompt experiments, to figure out the
superiority of our PREFERcompared with vanilla and opti-
mized non-iterative prompting works. Lastly, we compare a
variant of our PREFER, which rewrites synonymous prompts
for boosting instead of feedback-reflect-refine paradigm, for
ascertaining the utility of LLMs’ reflective ability.
Running settings To make a fair comparison, we closely
follow the experimental protocols that were set up in APO
with our own data split. In detail, we mainly conduct devel-
oping and evaluation of our PREFERin few-shot settings.
For each task, we randomly samplekexamples from the
original training dataset, to buildk-shot training setDtr. By
default, thekin this paper is set to 50. We use F1-score for
performance evaluation.
```

```
Datasets SNLI MNLI QNLI RTE Ethos Liar ArSarcasm
Single Prompt 0.587 0.660 0.660 0.720 0.833 0.535 0.
Single Prompt (CoT) 0.575 0.685 0.660 0.731 0.804 0.549 0.
Synonym Ensemble 0.580 0.746 0.720 0.659 0.812 0.572 0.
PromptBoosting 0.619 0.574 0.631 0.673 - - -
APO - - - - 0.964 0.663 0.
APO* - - - - 0.947 0.658 0.
Ours 0.647 0.767 0.793 0.753 0.963 0.744 0.
```
Table 1: Main experimental results of our PREFERand the compared approaches. APO and APO* respectively denote the
reported and our reproduced results of the Automatic Prompt Optimization (Pryzant et al. 2023).Bold: best; underline: runner-
up (results are based on our reproduction).

```
Method −Feedback −Bagging Voting Ours
SNLI 0.580↓ 0.640 0.626 0.
MNLI 0.746 0.713 0.733 0.
QNLI 0.720 0.747 0.767 0.
RTE 0.659↓ 0.740 0.760 0.
Ethos 0.812↓ 0.947 0.938 0.
Liar 0.572↓ 0.718 0.701 0.
Sarcasm 0.572↓ 0.653↓ 0.649↓ 0.
```
Table 2: Experimental results of the ablation study.↓indi-
cates a severe performance drop (more than 10%).

### Experimental Results

In view of the key proposals in our PREFERapproach, we are
naturally motivated to ask the following interesting research
questions.

- RQ1. Is the prompt ensemble learning really useful for
    improving LLMs’ performance?
- RQ2. Are the feedback-driven boosting and bilateral
    bagging mechanism both useful for prompt synthesis in
    ensemble learning?
- RQ3. Is the reason why our proposal is superior to the
    iterative approaches due to the expansion of the sample
    space?

To figure out the answers to these questions, we conduct
sufficient experiments and the experimental results can be
found in Table 1. For the first question, we compare the
ensemble-based approaches (including PromptBoosting and
our PREFER) with the single-prompt-based approaches. As
shown in the experimental results, when compared to the
vanilla (Line 1) and CoT-enhanced single prompt approach
(Line 2), both PromptBoosting and our PREFERoutperform
them by a significant margin. For example, our PREFERout-
performs the second best approach by up to 6.3% for the
QNLIdataset, and 13.1% for theLiardataset. The general
trend that becomes apparent from the results in Table 1 is
that the more difficult the task is, the better ensemble learn-
ing performs. We conjecture that it is due to the feedback-
reflect-refine paradigm can achieve greater improvement for
the harder tasks, while the marginal gain of this mechanism
would be diminishing for easier tasks. It is noted that the
experimental results change marginally by adding Chain-of-
Thought (CoT) for single-prompt approach.

```
0 1 2 3 4 5
Optimization Step
```
```
0.
```
```
0.
```
```
0.
```
```
0.
```
```
0.
```
```
0.
```
```
0.
```
```
F
```
```
Ours
APO
```
```
Figure 3: Training process comparison for APO and ours.
```
```
To explore the second research question, we compare
our PREFERwith both the two-stage ensemble approach
PromptBoosting (Line 4) and the synonym rewriting ensem-
ble approach (Line 3). For PromptBoosting, we use the pub-
licly available code of (Hou et al. 2023) and conduct ex-
periments following its hyperparameter setting. For the syn-
onym rewriting ensemble, we conduct prompt rewriting op-
eration with same semantics, followed by regular ensemble
learning similar to our PREFER. As demonstrated in Table 1,
our approach consistently outperforms the two ensemble ap-
proaches by a significant margin, reaching around 5% to
35% relative improvement in most datasets. We attribute the
superiority of PREFERto its feedback-reflect-refine mecha-
nism as well as the design of the joint optimization paradigm
that naturally captures relations among weak learners.
As for the third question, APO (Pryzant et al. 2023) is
introduced as the remarkable approach of iterative prompt-
ing for comparison. It is noted that we reproduce the APO
approach (APO* at Line 6) for a strictly fair comparison,
which eliminates the interference from data sampling. Sim-
ilar performance trends are observed in this comparison,
that is, our PREFERoutperforms APO with the power of
feedback-reflect-refine boosting and bilateral prompt bag-
ging. It manifests that through expanding the sample space
in a nonlinear way, prompting performance can be enhanced
significantly than single-prompt methods with similar iter-
ation rounds. In fact, attributed to our bagging design, our
PREFERis superior to APO not only in effectiveness, but
also in stability and efficiency.
```
### Ablation Study

```
To figure out the effectiveness of each component in our pro-
posal, we perform ablations on both feedback-reflect-refine
```

```
APO Ours
```
```
Frequency b(N+ 2) +T|Dsample| 2 N+ 2
Tstep 1 579.0 s 132.4 s
Tstep 2 2100.4 s 336.1 s
```
Table 3: Comparison of training efficiency. Frequency de-
notes the number of API accesses required by the method
within each optimization step, whereN is training size
andb,T,|Dsample|are hyperparameters required by APO.
Tstep 1 and Tstep 2 represent the time required for the corre-
sponding optimization steps from the beginning, where we
setN= 50,b= 4,T= 20,|Dsample|= 16.

boosting and bilateral bagging, and the experimental results
are provided in Table 2. First, we remove the feedback mech-
anism in prompt boosting (“−Feedback”), in which the ini-
tial seed prompt is just modified by the LLM without di-
rected optimization, then the similar boosting and bagging
strategy is performed to align the settings of our PREFER.
As shown in Table 2, it is observed that the prompt ensemble
without feedback-reflect-refine path is sub-optimal, signify-
ing that such feedback mechanism plays an important role
for directed prompt boosting. Second, to figure out the ef-
fectiveness of our bilateral bagging component, we also turn
off the whole component (“−Bagging”) or replace it with
majority voting (“Voting”), as shown in the column 3 and
4 in Table 2, respectively. The experimental results convey
that our bilateral bagging is beneficial for PREFER, and dis-
tinctly outperform the regular bagging approach of majority
voting. Notably, the performance of majority voting is basi-
cally satisfactory, manifesting that the prompt bagging can
benefit the boosting prompt process consistently. An inter-
esting phenomenon is that removing the feedback-reflect-
refine module leads to more serious performance decline
than removing the bagging module. This is expected, since
the bagging mainly benefits the stability for each prompt,
while the boosting is more important for prompt ensemble.

### Training Efficiency

To further demonstrate the superiority of our method, we
conduct detailed experiments on theEthosdataset for train-
ing efficiency, including training time and convergence
speed. As shown in Figure 3, both APO and our PREFER
reach the peak at optimization step 2 to 3, which indi-
cates that neither approaches require extensive iterations to
achieve impressive results. Clearly, our PREFERhas a more
stable performance retention compared to APO during sub-
sequent iterations. On the other hand, considering the lim-
itations on the speed and frequency of LLM API accesses,
we compare the number of API accesses during training and
the time consumption for the first two prompt optimization
steps, which is displayed in Table 3. It can be observed that
the access number of APO increases rapidly during beam
search and bandit selection, which brings serious efficiency
problems. On the contrary, our PREFERdoes not enforce op-
timal optimization at each time step, but rather maintains a
stable and efficient improvement via ensemble learning.

```
Given two sentences, determine whether sentence 2 provides
an answer to the question posed by sentence 1.
```
```
Assess whether sentence 2 provides supporting evidence or
contradictory information to the argument made in sentence
1, both implicitly and explicitly.
```
```
The prompt does not provide any guidance on how to handle
cases where the question posed by sentence 1 is vague
or open-ended.
The prompt does not provide any guidance on how to handle
cases where sentence 1 and sentence 2 have different
levels of specificity or granularity.
The prompt does not take into account the possibility of
implicit answers, where sentence 2 provides a
plausible inference or implication rather than an explicit
statement.
```
```
Decide whether sentence 2 answers the question asked by
sentence 1 when given two sentences.
```
```
Initial prompt
```
```
Reflection
```
```
Refine
```
```
Synonymous Rewriting
```
```
Figure 4: Comparison of the generation obtained from our
feedback-reflect-refine paradigm and synonymous rewrite.
```
### Case Study

```
To visualize our feedback-reflect-refine paradigm, we pro-
vided a case study as an illustration. As shown in Figure
4, taking the nature language inference task on theQNLI
dataset as an example, we provide the intermediate output of
the LLM in the feedback-reflect-refine process, to show its
effectiveness and interpretability. Compared to the prompt
generated by synonymous rewriting (gray box), the one gen-
erated by our method is more informative and logically com-
pensates for the deficiencies of the previous prompt, thus
achieving directed prompt optimization.
```
## Conclusion

```
In this paper, we propose a simple, automatic, and uni-
versal prompt ensemble approach called PREFER(PRompt
Ensemble learning viaFeedback-REflect-Refine), empiri-
cally showing consistent and significant improvement over
previous baselines. PREFERcontains two main components,
including feedback-reflect-refine prompt boosting and bilat-
eral prompt bagging. Prompt boosting branch directly and
collectively optimizes prompt in an automatic fashion based
on the evolving self-reflection. Prompt bagging proposes a
bagging paradigm containing forward and backward coop-
eration inspired by human behavior, which adequately un-
earths the real quality of each generated prompt and thus en-
sures the stability of both the feedback-reflect-refine process
and weight calculation in boosting. In a parallel note, our
PREFERbrings the prompt ensemble approach with more
interpretability by harnessing the LLMs’ language ability.
For future work, two interesting questions worth studying,
namely 1) how to further reduce the calculation of prompt
ensemble to approach single-prompt colleagues, and 2) how
to make more classical algorithm more intelligent based on
the power of LLMs.
```

## References

Bowman, S. R.; Angeli, G.; Potts, C.; and Manning, C. D.

2015. A large annotated corpus for learning natural language
inference.arXiv preprint arXiv:1508.05326.

Dagan, I.; Glickman, O.; and Magnini, B. 2005. The pascal
recognising textual entailment challenge. InMachine learn-
ing challenges workshop, 177–190. Springer.

Dai, D.; Sun, Y.; Dong, L.; Hao, Y.; Sui, Z.; and Wei, F.

2022. Why can gpt learn in-context? language models se-
cretly perform gradient descent as meta optimizers. arXiv
preprint arXiv:2212.10559.

Farha, I. A.; and Magdy, W. 2020. From arabic sentiment
analysis to sarcasm detection: The arsarcasm dataset. InPro-
ceedings of the 4th Workshop on Open-Source Arabic Cor-
pora and Processing Tools, with a Shared Task on Offensive
Language Detection, 32–39.

Freund, Y.; and Schapire, R. E. 1997. A decision-theoretic
generalization of on-line learning and an application to
boosting.Journal of computer and system sciences, 55(1):
119–139.

Hou, B.; O’Connor, J.; Andreas, J.; Chang, S.; and Zhang,
Y. 2023. Promptboosting: Black-box text classification with
ten forward passes. InInternational Conference on Machine
Learning, 13309–13324. PMLR.

Hu, E. J.; Shen, Y.; Wallis, P.; Allen-Zhu, Z.; Li, Y.; Wang,
S.; Wang, L.; and Chen, W. 2021. Lora: Low-rank adaptation
of large language models.arXiv preprint arXiv:2106.09685.

Ji, Z.; Lee, N.; Frieske, R.; Yu, T.; Su, D.; Xu, Y.; Ishii, E.;
Bang, Y. J.; Madotto, A.; and Fung, P. 2023. Survey of hal-
lucination in natural language generation.ACM Computing
Surveys, 55(12): 1–38.

Li, Y.; Lin, Z.; Zhang, S.; Fu, Q.; Chen, B.; Lou, J.-G.; and
Chen, W. 2023. Making Language Models Better Reasoners
with Step-Aware Verifier. InProceedings of the 61st An-
nual Meeting of the Association for Computational Linguis-
tics (Volume 1: Long Papers), 5315–5333.

Liu, P.; Yuan, W.; Fu, J.; Jiang, Z.; Hayashi, H.; and Neubig,
G. 2023. Pre-train, prompt, and predict: A systematic survey
of prompting methods in natural language processing.ACM
Computing Surveys, 55(9): 1–35.

Liu, X.; Zheng, Y.; Du, Z.; Ding, M.; Qian, Y.; Yang, Z.;
and Tang, J. 2021. GPT understands, too. arXiv preprint
arXiv:2103.10385.

Madaan, A.; Tandon, N.; Gupta, P.; Hallinan, S.; Gao, L.;
Wiegreffe, S.; Alon, U.; Dziri, N.; Prabhumoye, S.; Yang,
Y.; et al. 2023. Self-refine: Iterative refinement with self-
feedback.arXiv preprint arXiv:2303.17651.

Mollas, I.; Chrysopoulou, Z.; Karlos, S.; and Tsoumakas, G.

2020. Ethos: an online hate speech detection dataset.arXiv
preprint arXiv:2006.08328.

OpenAI. 2023. GPT-4 Technical Report. Technical Report
arXiv:2303.08774, OpenAI.

Opitz, D.; and Shavlik, J. 1995. Generating accurate and
diverse members of a neural-network ensemble. Advances
in neural information processing systems, 8.

```
Ouyang, L.; Wu, J.; Jiang, X.; Almeida, D.; Wainwright, C.;
Mishkin, P.; Zhang, C.; Agarwal, S.; Slama, K.; Ray, A.;
et al. 2022. Training language models to follow instructions
with human feedback.Advances in Neural Information Pro-
cessing Systems, 35: 27730–27744.
Pitis, S.; Zhang, M. R.; Wang, A.; and Ba, J. 2023. Boosted
Prompt Ensembles for Large Language Models. arXiv
preprint arXiv:2304.05970.
Pryzant, R.; Iter, D.; Li, J.; Lee, Y. T.; Zhu, C.; and Zeng, M.
```
2023. Automatic prompt optimization with” gradient de-
scent” and beam search.arXiv preprint arXiv:2305.03495.
Qiao, S.; Ou, Y.; Zhang, N.; Chen, X.; Yao, Y.; Deng,
S.; Tan, C.; Huang, F.; and Chen, H. 2022. Reasoning
with language model prompting: A survey.arXiv preprint
arXiv:2212.09597.
Radford, A.; Wu, J.; Child, R.; Luan, D.; Amodei, D.;
Sutskever, I.; et al. 2019. Language models are unsupervised
multitask learners.OpenAI blog, 1(8): 9.
Rajpurkar, P.; Zhang, J.; Lopyrev, K.; and Liang, P. 2016.
Squad: 100,000+ questions for machine comprehension of
text.arXiv preprint arXiv:1606.05250.
Wang, W. Y. 2017. ”liar, liar pants on fire”: A new
benchmark dataset for fake news detection.arXiv preprint
arXiv:1705.00648.
Wang, X.; Wei, J.; Schuurmans, D.; Le, Q.; Chi, E.; Narang,
S.; Chowdhery, A.; and Zhou, D. 2022. Self-consistency
improves chain of thought reasoning in language models.
arXiv preprint arXiv:2203.11171.
Wei, J.; Tay, Y.; Bommasani, R.; Raffel, C.; Zoph, B.;
Borgeaud, S.; Yogatama, D.; Bosma, M.; Zhou, D.; Metzler,
D.; et al. 2022a. Emergent abilities of large language mod-
els.arXiv preprint arXiv:2206.07682.
Wei, J.; Wang, X.; Schuurmans, D.; Bosma, M.; Xia, F.;
Chi, E.; Le, Q. V.; Zhou, D.; et al. 2022b. Chain-of-
thought prompting elicits reasoning in large language mod-
els. Advances in Neural Information Processing Systems,
35: 24824–24837.
Williams, A.; Nangia, N.; and Bowman, S. R. 2017. A
broad-coverage challenge corpus for sentence understand-
ing through inference.arXiv preprint arXiv:1704.05426.
Zelikman, E.; Mu, J.; Goodman, N. D.; and Wu, Y. T. 2022.
Star: Self-taught reasoner bootstrapping reasoning with rea-
soning.arXiv preprint arXiv:2203.14465.
Zhao, T.; Wei, M.; Preston, J. S.; and Poon, H. 2023a. Auto-
matic Calibration and Error Correction for Large Language
Models via Pareto Optimal Self-Supervision.arXiv preprint
arXiv:2306.16564.
Zhao, W. X.; Zhou, K.; Li, J.; Tang, T.; Wang, X.; Hou,
Y.; Min, Y.; Zhang, B.; Zhang, J.; Dong, Z.; et al. 2023b.
A survey of large language models. arXiv preprint
arXiv:2303.18223.
Zhao, Z.; Wallace, E.; Feng, S.; Klein, D.; and Singh, S.
2021. Calibrate before use: Improving few-shot perfor-
mance of language models. InInternational Conference on
Machine Learning, 12697–12706. PMLR.


