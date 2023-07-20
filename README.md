# 具体代码在master分支
# CAAI-BDSC2023-TASK1
### CAAI-BDSC2023社交图谱链接预测 任务一：社交图谱小样本场景链接预测
### 比赛链接:https://tianchi.aliyun.com/competition/entrance/532073/introduction?spm=a2c22.12281925.0.0.7aa47137syzS2r
### 最终排名: 36/755  MRR（平均倒数排名）:0.3262
#### 一些想法写在前面：
  本次比赛是我第一次参加社交图谱的比赛，虽然之前对GNN有一定的了解，但从来没有实现过相关算法，虽然最终排名不高，但从高排名选手的分享中学到了很多，在这里做一个简单的学习记录，方便自己后续回顾。个人认为DeepFM+GNN+LGB等模型最后走融合才是该题目的正确解法。

#### 自已的思路:
  从一开始看到这个比赛题目的时候，感觉可以用传统的机器学习方法来解决，即：多路召回+混合排序，类似于天池学习赛的“新闻推荐”比赛。但本人又想学习一下图神经网络，因为KG在现在的推荐和搜索领域非常有研究的价值，而主办方baseline用的是CompGCN中的mult，CompGCN本身已经是近年的sota之作，所以就在baseline的基础上对GNN进行了学习。
  本人主要从以下几个方面对模型进行了优化：
| 改动 | 作用 | 
| :-----| ----: |
| 在边和节点的embedding中加入统计特征 | $\downarrow$ |
| mult替换为conve方式计算 | $\downarrow$ |
| 扩大layersize和dim |$\uparrow$ |
| 扩大负采样 |$\uparrow$ |
| 将随机负采样改为基于流行度的负采样 |$\uparrow$ |
| 进行数据增强 |$\uparrow$ |

#### 融入特征效果却变差了:
尝试了一系列的深度学习方法之后，通过实验结果（可视化）和理论分析发现，大多数的深度学习方法通过在数据中学习节点和边的特征嵌入的来获得结果，然而很难直接捕捉到一些统计性质的全局特征，比如说交互的时间以及交互的次数，即使将这些因素纳入深度模型的建模之中，也很难学习到有效的表示。所以，与其费尽心思在节点和边的特征嵌入中做文章，不如直接使用随机的embedding，构建更合理的网路结构和损失函数，以便模型能够更好的学习到知识图谱的关系结构，进而更好的做出预测。

#### 修改负采样效果提升:
原本负样本数量和正样本相当，我们将负采样扩大两倍，相当于训练数据的负样本是正样本的三倍，训练集数量的增加对模型性能的提升很直接。<br>
原本负采样的策略是随机的负采样，这样就很有可能采样到原本需要补全的正样本，这种数据集上的噪音会对模型的性能造成极大的影响。为了减少采样错误的几率，我们使用了基于流行度的负采样，使曝光度很高但不在正样本中的节点有更大的概率被选为负样本，这样负样本错误的几率就会被大大降低，提高了训练集的质量，使模型的精度提升了0.02。

#### 数据增强效果提升:
正确的增强：利用三元组中的if_voter_participate属性，如果三元组中存在这个属性且值为True，则增加一条起始为inviter，终点为voter的名字为“participate”的边。这种增强会将结果提升0.03。

错误的数据增强：为每条边增加一条对应的反向边。<br>
错误的数据增强：如果inviter和voter之间存在两条或以上的边，则增加一条inviter到voter的名字为“iage2”的边。这种增强会将valid数据集的MRR变得很高，但实际上结果很差，应该是发生了非常严重的过拟合了。

#### 后期了解的知识：
针对KG中的链路预测问题，目前有两条分支：平移-距离模型（trans系列、LineaRE）和语义匹配模型（ConvE、RESCAL）。<br>
平移距离模型利用了基于距离的评分函数，通过两个实体之间的距离对事实的合理性进行度量；语义匹配模型通过匹配实体的潜在语义及其向量空间表示中体现的关系来衡量事实的合理性。<br>
排名第一选手用的是LineaRE模型，属于平移-距离模型，在形式上类似于TransE，属于平移距离模型。然而，LineaRE本质上不同于TransE的其他变体，如TransH、TransR和TransD，它们将实体投影到平面或特定向量空间中，而LineaRE将知识图嵌入视为一项线性回归任务。它对知识图谱中的 connectivity patterns（包括 Symmetry、Antisymmetry、Inversion、 Composition）以及 mapping properties（1 对 1、1 对多、多对 1、多对多）都能有效建模。用第一名的话来说就是，“LineaRE的建模能力在目前我所知道的平移距离模型中是最优秀的，实际上的效果也证明了这点。”

#### 杂谈：
除了第一名和第二名之外，剩余的前六名选手使用的都是特征工程+GBDT，不得不说Xgboost和LGBM在目前的各种预测任务的比赛中的效果真的非常厉害。它可以在不到GNN 10分之一的运行时间内达到和它相近甚至远超GNN的精度，并且往往GBDT的稳定性要比GNN更好。<br>
出乎本人意料的是，比赛前排选手没有用GNN和机器学习模型进行融合的，按照我个人的理解，GNN主要学习知识图谱中节点之间的关系属性和连接属性，而机器学习模型主要学习用户和场景的统计特征，两者在理论上是非常互补的关系，所以经过合理的模型融合后应该会达到很不错的效果。具体的实践，我会在后续有空尝试一下。

后续我会将自己的代码和值得学习的前排选手的代码发出来，正在整理当中。。。。。<br>
很高兴分享这个比赛经历给大家，和大家一起讨论推荐算法的学习，有疑惑的可以发邮件到：546408143@qq.com
