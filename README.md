# Awesome Incremental Learning / Lifelong learning

## ContinualAI wiki

#### [An Open Community of Researchers and Enthusiasts on Continual/Lifelong Learning for AI](https://www.continualai.org/)

## Workshops

#### [Continual learning workshop NeurIPS 2018](https://sites.google.com/view/continual2018/home?authuser=0)

## single-head v.s. multi-head; w exemplars or w/o exemplars

Being aware of different training and test settings.

## Papers
### 2019
- <a name="todo"></a> Large Scale Incremental Learning (**CVPR2019**) [[paper](https://arxiv.org/abs/1905.13260)] 
>现代机器学习在逐步学习新类别时遭受灾难性遗忘。由于缺少旧类的数据，性能急剧下降。已经提出了增量学习方法来保留从旧类中获得的知识，通过使用知识提取并保留旧类中的一些示例。但是，这些方法难以扩展到大量类。我们认为这是因为两个因素的结合：（a）新旧类别之间的数据不平衡，以及（b）视觉上相似类别的数量不断增加。当训练数据不平衡时，区分越来越多的视觉上相似的类别是特别具有挑战性的。我们提出了一种简单有效的方法来解决这一数据不平衡问题。我们发现最后一个完全连接的层对新类具有强烈的偏差，并且这种偏差可以通过线性模型来校正。通过两个偏差参数，我们的方法在两个大型数据集上表现非常出色：ImageNet（1000类）和MS-Celeb-1M（10000类），分别优于最新算法
- <a name="todo"></a> Learning to Remember: A Synaptic Plasticity Driven Framework for Continual Learning (**CVPR2019**) [[paper](https://arxiv.org/abs/1904.03137)] 
- <a name="todo"></a> Task-Free Continual Learning (**CVPR2019**) [[paper](https://arxiv.org/pdf/1812.03596.pdf)]
- <a name="todo"></a> Learn to Grow: A Continual Structure Learning Framework for Overcoming Catastrophic Forgetting (**ICML2019**) [[paper](https://arxiv.org/abs/1904.00310)]
- <a name="todo"></a> Efficient Lifelong Learning with A-GEM (**ICLR2019**) [[paper](https://openreview.net/forum?id=Hkf2_sC5FX)] [[code](https://github.com/facebookresearch/agem)]
- <a name="todo"></a> Learning to Learn without Forgetting By Maximizing Transfer and Minimizing Interference (**ICLR2019**) [[paper](https://openreview.net/forum?id=B1gTShAct7)] 
- <a name="todo"></a> Overcoming Catastrophic Forgetting via Model Adaptation (**ICLR2019**) [[paper](https://openreview.net/forum?id=ryGvcoA5YX)] 
- <a name="todo"></a> A comprehensive, application-oriented study of catastrophic forgetting in DNNs (**ICLR2019**) [[paper](https://openreview.net/forum?id=BkloRs0qK7)] 

### 2018
- <a name="todo"></a> Memory Replay GANs: learning to generate images from new categories without forgetting
 (**NIPS2018**) [[paper](https://arxiv.org/abs/1809.02058)] [[code](https://github.com/WuChenshen/MeRGAN)]
 - <a name="todo"></a> Reinforced Continual Learning (**NIPS2018**) [[paper](http://papers.nips.cc/paper/7369-reinforced-continual-learning.pdf)] [[code](https://github.com/xujinfan/Reinforced-Continual-Learning)]
 - <a name="todo"></a> Online Structured Laplace Approximations for Overcoming Catastrophic Forgetting (**NIPS2018**) [[paper](http://papers.nips.cc/paper/7631-online-structured-laplace-approximations-for-overcoming-catastrophic-forgetting.pdf)]
- <a name="todo"></a> Rotate your Networks: Better Weight Consolidation and Less Catastrophic Forgetting (R-EWC) (**ICPR2018**) [[paper](https://arxiv.org/abs/1802.02950)] [[code](https://github.com/xialeiliu/RotateNetworks)]
- <a name="todo"></a> Exemplar-Supported Generative Reproduction for Class Incremental Learning  (**BMVC2018**) [[paper](http://bmvc2018.org/contents/papers/0325.pdf)] [[code](https://github.com/TonyPod/ESGR)]
- <a name="todo"></a> DeeSIL: Deep-Shallow Incremental Learning (**ECCV2018**) [[paper](https://arxiv.org/pdf/1808.06396.pdf)] 
- <a name="todo"></a> **Important!!** End-to-End Incremental Learning (**ECCV2018**) [[paper](https://arxiv.org/abs/1807.09536)][[code](https://github.com/fmcp/EndToEndIncrementalLearning)]
>虽然深度学习方法近年来由于其最先进的结果而脱颖而出，但它们仍然遭受灾难性的遗忘，在逐步增加新课程的培训时，整体表现急剧下降。这是由于当前的神经网络架构需要整个数据集，包括来自旧类和新类的所有样本，以更新模型 - 随着类数量的增加，这一要求变得容易不可持续。我们通过逐步学习深度神经网络的方法来解决这个问题，使用新数据并且只使用与旧类中的样本相对应的小样本集。这是基于由蒸馏措施组成的损失，以保留从旧类中获得的知识，以及用于学习新类的交叉熵损失。我们的增量训练是在保持整个框架端到端的同时实现的，即共同学习数据表示和分类器，而不像最近没有这种保证的方法。我们在CIFAR-100和ImageNet（ILSVRC 2012）图像分类数据集上广泛评估我们的方法，并展示最先进的性能。
- <a name="todo"></a> Riemannian Walk for Incremental Learning: Understanding Forgetting and Intransigence (**ECCV2018**)[[paper](http://arxiv-export-lb.library.cornell.edu/abs/1801.10112)] 
- <a name="todo"></a> Piggyback: Adapting a Single Network to Multiple Tasks by Learning to Mask Weights (**ECCV2018**) [[paper](https://arxiv.org/abs/1801.06519)] [[code](https://github.com/arunmallya/piggyback)]
 - <a name="todo"></a> Memory Aware Synapses: Learning what (not) to forget (**ECCV2018**) [[paper](https://arxiv.org/abs/1711.09601)] [[code](https://github.com/rahafaljundi/MAS-Memory-Aware-Synapses)]
  - <a name="todo"></a> Lifelong Learning via Progressive Distillation and Retrospection (**ECCV2018**) [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Saihui_Hou_Progressive_Lifelong_Learning_ECCV_2018_paper.pdf)] 
- <a name="todo"></a> PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning (**CVPR2018**) [[paper](https://arxiv.org/abs/1711.05769)] [[code](https://github.com/arunmallya/packnet)]
- <a name="todo"></a> Overcoming Catastrophic Forgetting with Hard Attention to the Task (**ICML2018**) [[paper](http://proceedings.mlr.press/v80/serra18a.html)] [[code](https://github.com/joansj/hat)]
- <a name="todo"></a> Lifelong Learning with Dynamically Expandable Networks (**ICLR2018**) [[paper](https://openreview.net/forum?id=Sk7KsfW0-)] 
- <a name="todo"></a> FearNet: Brain-Inspired Model for Incremental Learning (**ICLR2018**) [[paper](https://openreview.net/forum?id=SJ1Xmf-Rb)] 

### 2017
- <a name="todo"></a> Overcoming catastrophic forgetting in neural networks (EWC) (**PNAS2017**) [[paper](https://arxiv.org/abs/1612.00796)] [[code](https://github.com/ariseff/overcoming-catastrophic)] [[code](https://github.com/stokesj/EWC)]
- <a name="todo"></a> Continual Learning Through Synaptic Intelligence (**ICML2017**) [[paper](http://proceedings.mlr.press/v70/zenke17a.html)] [[code](https://github.com/ganguli-lab/pathint)]
- <a name="todo"></a> Gradient Episodic Memory for Continual Learning (**NIPS2017**) [[paper](https://arxiv.org/abs/1706.08840)] [[code](https://github.com/facebookresearch/GradientEpisodicMemory)]
- <a name="todo"></a> iCaRL: Incremental Classifier and Representation Learning (**CVPR2017**) [[paper](https://arxiv.org/abs/1611.07725)] [[code](https://github.com/srebuffi/iCaRL)]
- <a name="todo"></a> Continual Learning with Deep Generative Replay (**NIPS2017**) [[paper](https://arxiv.org/abs/1705.08690)] [[code](https://github.com/kuc2477/pytorch-deep-generative-replay)]
- <a name="todo"></a> Overcoming Catastrophic Forgetting by Incremental Moment Matching (**NIPS2017**) [[paper](https://arxiv.org/abs/1703.08475)]
- <a name="todo"></a> Expert Gate: Lifelong Learning with a Network of Experts (**CVPR2017**) [[paper](https://arxiv.org/abs/1611.06194)] 
- <a name="todo"></a> Encoder Based Lifelong Learning (**ICCV2017**) [[paper](https://arxiv.org/abs/1704.01920)] 

### 2016
- <a name="todo"></a> Learning without forgetting (**ECCV2016**) [[paper](https://link.springer.com/chapter/10.1007/978-3-319-46493-0_37)] [[code](https://github.com/lizhitwo/LearningWithoutForgetting)]



