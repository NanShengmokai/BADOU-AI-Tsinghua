# 自写InceptionV3代码与TensorFlow或Pytorch官网源码的对比笔记
> [同论文]：为改动的目的是与论文中的语句或图表相一致<br>
> [同源码]：为论文中缺失相关描述，或认为源码更加合理<br>
> [均不同]：为取论文和源码折衷，或省略该部分内容

### (I). 省略了辅助分类器部分(auxiliary classifiers)
&emsp;&emsp;[均不同]省略辅助分类器相关代码部分

### (II). Module InceptionFig5(对应源码InceptionA):
<ol>
<li> [同论文] 将5×5卷积替换成“一个”3×3卷积(因为另一个分支已经是2个串联的3×3卷积了)</li>
<li> [同源码] pooling分支的输出pool_features参数保留，没有按照原文固定为64</li>
</ol>

### (III). Module InceptionFig5ToFig6(对应源码InceptionB):
<ol>
<li> [同源码] 第一个分支只保留了3×3卷积，去掉了论文Fig.10中在前面串联的1×1卷积</li>
</ol>

### (IV). Module InceptionFig6(对应源码InceptionC):
<ol>
<li> [同源码] Module内部隐藏层间传递时的feature map数量被一个参数channels_inside定义，在几个Module串联结构中依次增加</li>
</ol>

### (V). Module InceptionFig6ToFig7(对应源码InceptionD):
<ol>
<li> [同源码] 中间分支串联1×1 1×7 7×1 3×3 四个卷积，不同于Fig.10</li>
</ol>

### (VI). Whole Neural Networks
<ol>
<li> [同论文] 第5,6,7Conv层采用论文Table提供的layout结构，而不是源码中3层串联1×1 3×3 max pool结构</li>
<li> [同源码] InceptionFig6 串联4次，而非论文中的5次</li>
</ol>

# 结尾
&emsp;&emsp;我推测造成官方源码与论文叙述不同的原因可能实际使用过程中，根据实际情况做出调整，可能包括：
<ol>
<li>工作效果(准确率、收敛速率等)</li>
<li>条件限制(硬件匹配、内存优化、算力成本等)</li>
<li>减少过设计(辅助分类器个数、Inception串联数等)</li>
<li>……</li>
</ol>

在Kaggle上实际运行自写模型训练过程证明可以跑的通，其log信息存储在"Log of training process of MyInceptionv3.txt"文件中。

