---
layout: post
title:  "炼丹大法 A Recipe for Training Neural Networks"
---

Andrej Karparthy 2019年的一篇博客 [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/), 以下为全文翻译。

---
[TOC]

## 导言

### 1) 训练神经网络是一个["漏水"的概念](https://www.joelonsoftware.com/2002/11/11/the-law-of-leaky-abstractions/)

很多博客/教学视频很自豪的宣称“只需要30行代码即可解决数据训练问题”，给大家一种错误的印象即训练一个神经网络非常简单，即插即用

```python
>>> your_data = # 导入你的数据集
>>> model = SuperCrossValidator(SuperDuper.fit, your_data, ResNet50, SGDOptimizer)
# 征服世界吧
```

这样的代码和示例让人回想起标准的软件——拥有干净的API接口和抽象——比如[Requests网络库](http://docs.python-requests.org/en/master/)：
```python
>>> r = requests.get('https://api.github.com/user', auth=('user', 'pass'))
>>> r.status_code
200
```

看上去很酷，但其实背后需要很多工作将诸多复杂的细节隐藏在标准库中，只展现几行代码。  
然而神经网络与传统软件天差地别：它们不是标准化(off the shell)，即插即用(plug and play)的技术，
在ImageNet上训练一个分类器很简单，但只要稍稍修改就需要更深刻的理解
（参见原作者的另一篇博客["Yes you should understand backprop"](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b))。  
BackProp+SGD不意味着网络一定收敛；BatchNorm也不代表收敛一定更快；RNNs不代表你可以随意处理任何文本；可以使用RL(Reinforcement Learning)来抽象不见得应该使用RL。 

**如果你坚持使用并不理解的技术，那么你大概率会失败**

### 2) 训练神经网络会静静的失败

当你代码写错的时候，你会遇到各种各样的异常(Exception)：
 - 参数类型错误
 - 参数树立那个错误
 - import失败
 - 不存在的Key
 - ...

对于神经网络的训练，这些**软件的错误**只是一个开始。

当你搞定了所有的编译错误和语法问题，网络可能还是不能work，这时就很难讲文提出在那里了。问题可能出现的地方太多了，并且是逻辑(logical_问题而不是语法(syntactic)问题，很难通过传统软件中的单元测试解决，比如：
 - 数据增强环节中，你flip了图像，但忘记了对label进行flip——很可能最终网络拟合的很好，它学会了根据图像去flip label...
 - 自回归模型(autoregressive)中你不小心将输出作为输入进行训练
 - 应该对梯度进行clipping却对损失(loss)进行了clipping, 导致训练中直接忽略了样本中的outlier
 - 你导入了预训练的圈中却没有使用原本的均值
 - 超参数配置错误：正则化参数、学习率、模型大小等等

如果你的模型能跑出异常才是撞了大运了，大多数时候它只会默默的掉几个点...
因此，想“多快好省”的训练神经网络是不切实际的，只会带来无尽的痛苦。炼丹的过程痛苦是无法避免的，但遵循以下的原则是可以缓解的：
 - 细致(thorough)
 - 保守(defensive)
 - 多疑/偏执(paranoid)
 - 执着于可视化(一切可能被可视化的环节)

训练神经网络最重要的是**耐心**和**对细节的关注**

---
## 炼丹大法

当需要训练一个神经网络来解决新的问题时，要从从简到繁的构建并且在每一步先进行坚实的假设，而后利用实验来验证或者深入挖掘到发现问题为止。
在这个过程中要坚决避免同事引入多个“未验证”的难题而导致难以发现的错误。

*如果把炼丹的过程比作炼丹本身，那么你会希望使用很小的学习率，并且每次迭代都在完整的测试集上评估性能*

### 1. 与数据合二为一

此时不要写任何网络相关的代码，详细的检视你的输入数据。   
**这一步非常重要**  
人脑非常擅长扫描大量的数据，理解他们的分布并且搜寻数据的模式：
 - 重复的图片
 - 错误的图片/标签
 - 不平衡的数据分布
 - 根据你的需求判断，你需要什么样的网络结构？局部的特征就够了还是需要全局的上下文？
 - 数据之间的变化如何？那些变化其实是虚伪的(spurious)，预处理即可消除？
 - 空间位置重要吗？是否需要池化？
 - 细节的重要性如何？我们可以将采样到什么程度？
 - 标签有多少噪声？

同时，由于神经网络实际上是你数据集的一个压缩/提取(信息论)， 你可以观察网络的正确和错误预测就能明白错误可能来源于哪个环节。

对数据集有了定性的认识之后，你可以写一些简单的代码来对数据进行一些任何你能想到的操作(查找/过滤/排序), 并可视化它们的分布，和噪声。

### 2. 构建端到端的训练和评估框架 + dummy基线版本

现在你已经熟悉了数据，我们可以开始训练铉酷的多尺度注意力模型了吗？ nonono..

下一步时搭建完整的训练流程以及评测的框架，并且一步一个脚印的测试。此时最好先从一个简单到绝不会搞砸的模型开始——比如线性分类器、或者小型卷积网络——训练，可视化模型的损失，精度，模型的预测结果，并且不断的进行消融测试(ablation experiments)。

这一步的技巧：
 - **固定随机种子**  
``` python
import ramdom
random.seed(0)
```
减少程序的随机性，保证结果可稳定复现

 - **简化再简化**  
不要添加任何额外的非必要的功能，比如数据增强。  
> *数据增强本质上时一种正则化方法，会在后续环节引入，目前使用它只会增加搞砸的概率*
 - **增加评估的颗粒度**  
在对测试的损失画图表时，在整个测试集上进行验证——即使测试集比较大。目前我们只关注整个框架的正确，相信我，花点时间是值得的。
 - **初始化时验证模型的损失**  
训练开始之前，验证初始化的网络输出的损失是正确的。比如最后的softmax层应该输出`-log(1/n_classes)`，对于L2, Huber等等可以类推。
 - **正确的初始化**  
正确的初始化最后一层的权重。比如如果你回归的数据均值是50，那么你最终的偏差(bias)应该是50。如果你的数据集分类不平衡——正负样本是1：10——那么你最后一层`logits`的`bias`的初始化参数应该最终输出0.1的概率。  
正确的初始化可以加速网络的收敛，避免损失曲线的“曲棍球”现象——开始几轮模型只不过是在学习`bias`。
 - **人肉基线**  
对于你来说，应该更关注于可解释的尺度，比如`accuracy`而不是`loss`。如果可能的话，人肉测试一下你的`accuracy`和模型进行对比。你也可以手动标注测试集两次，对比一下差异(即标注的`accuracy`)。
 - **输入无关的基线**  
训练一个和输入无关的基线版本——比如输入置零的数据，它的性能应该比使用真实数据训练要差。如果不是这样，可能你的模型没有从数据中学到任何信息。
 - **过拟合一个批次**  
尝试对一个批次的数据(少至2个)在模型上进行过拟合。借此，我们可以增加模型的能力——比如增加layer或者filter——并且验证我们可以拟合到最小的损失(接近0)。  
我个人喜欢在一个图表里同时可视化真值和预测值,以此来确认当模型的损失接近最小时二者也一致了。如果达不到这个效果，那么框架里还有隐藏的bug，还不能进入下一步。
 - **确认训练时损失是下降的**  
在这个阶段，网络对于数据还是欠拟合的，因为我们仍在使用一个简单的模型。试试扩展一下它的能力，损失还会下降吗？
 - **可视化网络的*真实*输入**  
毫无疑问，正确的可视化数据应该在数据即将输入模型之前，即`y_hat = model(x)`，在任何预处理和数据增强之后。
 - **观察预测结果的变化**  
我很喜欢在训练时可视化一个固定的测试批次，从其预测输出的变化可以很直观的观察摩城训练的过程。有时候能感受到网络挣扎着对数据进行拟合——显示网络能力不足。过高或者过低的学习率同样能反映为抖动的程度。
 - **利用反向传播来调试**  
深度学习的代码经常包含复杂的向量化操作和广播操作。我遇到的一个常见bug是在该使用`transpose/permute`时错误的使用了`view`，因而错误的混用了不同批次见的数据。令人沮丧的是，网络训练后的结果依然可用，因为它学会了忽略其他数据。其中一个调试的方法是手动把模型的loss第`i`个样本输出的和，进行反向传播到输入层，那么你应该只在第`i`个样本的输入观察到非0的梯度。  
总的来说，梯度能反映你网络中信息传递的依赖关系，经常能帮助调试。

 - **不要基于实现通用代码**  
这其实是一个编程上的建议。不要一开始就尝试造轮子，或者写一些通用的功能性代码。我通常专注于当下要实现的功能，确认它可以运行，稍后再将它泛化。比如numpy/torch中的向量化方法，我总是先用最简单的循环实现，然后一层一层的改写成向量化代码。  

### 3. 过拟合
在这个阶段，我们应该对数据集有了比较好的了解，并且有了完整可行的训练+验证流程。对于任何一个给定的模型我们能够(可复现地)计算一个可信的评价尺度。通过训练一个和输入无关的基线，几个简单模型的基线，以及一个人肉基线，我们已经准备好在一个真正的好模型上训练了。  
我找一个好模型的通常分为两步：  找一个足够大的模型来过拟合(聚焦于训练的损失)，然后进行适当的正则化(牺牲训练集的损失，提升验证集的损失)。
我偏爱这两个过程的原因是，如果用任何模型都无法达到比较好的效果，那么很可能前面的步骤还存在着bug或者配置错误。  
以下是一些tips：

 - **选择模型**  
想获得最低的训练集损失，你需要为数据选择合适的网络结构。对此我的no.1建议是：**不要逞英雄**。很多人迫不及待地开始创新：把他们认为“合理”的各种奇怪的架构像堆乐高一般组成网络。在你项目的早期一定要抵制住这种诱惑。我经常建议人们去找到最相关的论文，只要复制粘贴他们最简单的网络架构来达到不错的性能。你可以在此基础上作修改，但是要稍后。
 - **adam很安全**  
在构建基线版本的早期，我喜欢使用Adam优化器以及[3e-4的学习率](https://twitter.com/karpathy/status/801621764144971776?lang=en)。我个人的经验是Adam优化器对超参数的容忍度很高，包括糟糕的学习率。对于卷积网络来说，一个调参好的SGD优化器通常会比Adam效果更好，但SGD优化器可正确收敛需要的学习率区间要岝的多，对于不同问题学习率也截然不同。注意，如果你在使用RNN或相关的序列模型，Adam更常用。  
总之，在项目早期，**不要逞英雄**，遵循大多数论文的方法吧。
 - **一步一个脚印**  
如果你有很多不同的信号可以输入你的分类器，那我建议你每次之增加一个，并且验证确保你得到了预想中的提升。不要一开始就一股脑的丢给你的模型。  
也有些其他方法不断的增加模型的复杂性，比如先输入较小的图片，随后逐渐增大，等等。
 - **不要相信默认的学习率衰减参数**  
如果你在复用其他领域的代码，那你要注意学习率衰减参数。不但训练时每轮的学习率衰减的设计对于不同的问题有所区别，它对于不同的训练阶段(轮次)，甚至不同的数据集也是不同的。比如Image在第30轮衰减10，如果你不使用ImageNet你不会使用这个参数。如果你不小心把你的消息率过早的衰减到0，你的网络就无法收敛。我个人通常禁用学习率衰减——即使用恒定学习率——并在非常后期的阶段对它进行微调。

### 4. 正则化
如果一切顺利，我们现在能够在训练集上拟合一个大模型了。是时候开始对它进行正则化来提升验证集上的精度(训练集上会下降，这是正常的)。  
以下是一些技巧：

 - **扩充数据集**  
最好的办法就是增加真实的训练数据了。  
一个常见的舞曲就是很小的数据集上花费很多时间，而不是去收集更多数据。据我所知增加数据几乎是唯一能保证提升网络性能的方法。
 - **数据增强**  
如果标注真实数据成本抬高，那么备选方案就是增强的假数据了。
 - **更逼真的数据增强**  
如果数据增强的假数据还不够，那么完全伪造的数据也可能有帮助。  
有一些更有创意的扩充数据集的方法，。比如使用[域随机化](https://openai.com/blog/learning-dexterity/) 或者[仿真](http://vladlen.info/publications/)
 - **预训练**  
适用预训练的网络几乎不会有什么负面影响，即使你有很多数据也是如此。
 - **坚持监督学习**  
不要对无监督与训练抱过高的期待：据我所知，它在现代计算机视觉领域没有过很好的效果——BERT等模型在NLP领域很棒，主要是由于文本天然具有较高的信噪比。
 - **减小输入维度**  
如果你的数据比较少，增加网络的输入只会增加过拟合的概率，因此需要尽量减少输入的维度可以去除一些含有虚伪信号的特征。  
比如如果底层特征对于你的应用并不重要，那么可以试试输入低分辨率的图片。
 - **减小模型大小**  
很多情况下，你可以尝试使用某个领域的先验/专家知识来减小你模型的尺寸。曾经在ImageNet后面使用全连接层(Fully Connectec Layers)非常流行，打败随后发现只需要简单的池化(pooling)就可以达到同样的效果，因而干掉了大量的参数。
 - **减小批次大小(batch size)**  
由于批标准化(batch normalization)的存在，更小的批次大小实际上相当于更强的正则化。原因是由于批次内的均值/方差会更接近数据集的实际分布，因而对应的scale/offset会使你的批次“抖动”的更多。
 - **dropout**  
对网络增加dropout可以减弱网络的过拟合，卷积网络可以使用spatial dropout。 注意dropout和批标准化可能会存在[冲突](https://arxiv.org/abs/1801.05134)。
 - **权重衰减(weight decay)**  
增加权重衰减的惩罚系数。
 - **早停(early stopping)**  
根据模型在验证集的损失来派端，在即将过拟合之前停止训练。
 - **试试更大的模型**  
大模型的过拟合能力更强大，但同样在“早停”时得到的模型性能会比小模型好得多。
### 5. 微调

这时候你应该在你数据上进行MIL(Model in the loop)的迭代测试不同的网络架构了，以下是一些trick:

 - **随机搜索而不是网格搜索**  
同时调试多个超参数时，对所有设置进行网格搜索听上去会很诱人，但记住[随机搜索往往效果更好](http://jmlr.csail.mit.edu/papers/volume13/bergstra12a/bergstra12a.pdf)。 从直觉上来说，这是因为神经网络的性能对于某些参数更加敏感，因此如果参数`a`更重要而参数`b`没有任何效果，那么你应该对`a`更精细的采样。
 - **超参数优化**  
有不少基于贝叶斯的超参数优化工具，据我一些朋友反馈有帮助；  
但我个人的经验是搜索模型和超参数的SOTA方法是找一个实习生（大误）

### 6. 榨干最后的性能
此时你已经找到了最佳的网络结构和超参数，有些方法可以帮助你榨干模型的最后一点性能。

 - **模型集成**  
模型集成(model ensemble)是几乎可以保证提升任何模型2%精度的方法。如果你test time的计算资源有限，可以试试知识蒸馏，参考这篇[dark knowledge](https://arxiv.org/abs/1503.02531).
 - **让它一直训练**  
通常大家在validation上的效果开始下降时就停止了训练。 在我个人经验，网络被反直觉地训练。有一次我放寒假前忘了停止训练，放假回来已经是SOTA了 :)

---
## 结论

到这里你已经有了炼丹的所有要素：你对技术，数据和要解决的问题有了深入的了解；你搭建了完整的训练/评测设施并对模型的精度有高度的信心；你尝试越来越复杂的模型，并且且逐步提升了效果。

现在，你已经准备好阅读大量的文献，进行大量的实验，并取得你自己的SOTA了，好运！