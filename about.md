---
layout: page
title: 梁宇钦
---

<center>
Email: YuqinLiangX@gmail.com
</center>

## 教育经历

- 南京大学  视觉/三维重建    硕士   2019.09~2022.06
- 南京大学  电子信息科学技术    学士    2015.09~2019.06

## 工作经历

- 阿里巴巴集团 $\cdot$ 淘宝天猫商业集团 $\cdot$ 大淘宝技术 $\cdot$ Meta技术-3D重建  算法工程师   2022.06~Now
- 阿里巴巴集团 $\cdot$ 淘系技术部 $\cdot$ 淘系机器智能 $\cdot$ 基础算法-3D视觉   算法工程师(实习)  2021.05~2021.08

## 项目经历

### **阿里巴巴 – 低成本商品三维重建**  *算法工程师*  2022.07–Now

- **项目技术**: Python/Pytorch/C++/CUDA; NeRF, 体渲染, 可微渲染。
- **项目简介**: 探索和实现面向未来的三维重建方法和技术，实现三维内容的低成本规模化生产。
  - 对重建的物体，拍摄物体的各个视角图像序列, 采用 SFM 方法得到每张图像的位姿关系和相机内参;
  - 通过 NeRF 和体渲染技术, 输入所拍摄的图像序列以及位姿关系和相机内参, 重建出高精度的物体三维结果;
  - 可选: 1、将 NeRF 结果 bake 到体素网格, 进行渲染加速和压缩, 实现端上实时渲染和较小模型 (15MB)。
  - 2、从 NeRF(NeuS) 提取物体 mesh, 进行 UV 展开, 通过可微渲染得到 mesh 网格和纹理 (传统模型结果)。
- **主要工作**: 负责重建算法的效果和速度优化, 项目架构的重新设计和代码重构。
  - 实现 3D-BBox, Empty Space 以及 Dynamic Visible 采样点过滤,
    整体重建速度提升 4 倍以上 (30+h → 7+h); 去除 baking 阶段, 速度提升 7 倍以上 (27+h→4+h);
  - 使用 Grid Embedding + Tiny MLP 替换原来的大 MLP, 并解决 geometry field 连续性问题,
    整体重建速度提升 2 倍 (7+h → 4h);
    同时重建结果的 mesh 质量和纹理质量都得到大幅提升 (PSNR 提升 1 个点以上)。
  - 引入 depth (sparse 或者 dense) 监督, 一定程度上解决鞋和帽子内腔等部位无法正确凹陷的问题。
  - 重新设计和重构更加合理的项目架构, 独立完成了重建链路各阶段的解耦合; 项目可拓展性和可维护大大提升。
- **项目难点**: 从 2D 图像恢复 3D 模型, 是一个非常有挑战的问题, 目前不存在成熟解决方案。
  - NeRF 技术是比较新的研究技术, 很多问题需要解决;
  - NeRF 重建过程是一个黑盒的优化过程, 各种改进策略不能在理论上保证不影响重建结果, 需要充分验证;
  - NeRF 体渲染计算非常大, 做不到实时渲染; baking 到体素网格后, 需要兼顾渲染效率和模型大小。
  - 原来的项目架构存在较强的耦合, 并且重建链路较长, 项目的拓展性和可维护性较低。
- **个人收获**:
  - 了解了完整的重建链路 (项目重构), 特别是重建算法部分的大部分流程细节都有深入了解。
  - 深入掌握了 NeRF 这个新兴的技术, 我们研究和改进的点也一直保持在 NeRF 研究的前沿。
  - 进一步学习和熟悉了 CUDA (torch extension) 编程, 能够自己写出高效的 Pytorch 拓展。
  - 项目架构设计能力得到了实践锻炼, 对项目架构和流程的重要性有了进一步的理解。

### **南京大学 – 基于人体背部表面信息的脊柱三维重建** *项目负责人* 2019.10–-2021.05

- **项目技术**: Python/Pytorch/C++; 三维重建;SMPL variant; 链式运动;优化算法。
- **项目简介**: 实现一种基于人体背部表面信息的人体脊柱三维重建和姿态估计方法。
  - 根据人体脊柱解剖学关系,脊柱力学特性等先验知识建立标准人体脊柱三维运动模型。
  - 根据人体背部表面信息 (RGBD) 重建人体背部表面三维模型,提取人体脊柱相关信息。
  - 根据提取的信息以及标准人体脊柱三维运动模型,优化得到真实的人体脊柱模型,进行脊柱姿态估计。
- **项目成果**: 实现了依据人体背部表面信息而非 X 光等重建人体脊柱三维模型, 估计人体脊柱姿态。
  - 建立了标准人体脊柱模版模型和运动模型，实现了人体背部模型重建;
  - 能够根据人体背部标注的脊柱相关信息优化得到真实人体脊柱姿态。
- **项目难点**: 相关研究比较少, 从而相应的文献/数据就比较少; 问题本身难度很大。
  - 背部表面信息与脊柱之间的关联是模糊的, 并且信息是不足的; 需要充分结合脊柱本身的先验特性。
  - 数据稀缺。脊柱侧凸病人人体背部表面数据的采集很慢; 三维脊柱模型的数据的也比较少。
  - 目前脊柱姿态的健康状况都是从二维的指标来判断的, 还没有三维模型下的判断指标。
- **个人收获**: 从 0 到 1 把项目做了起来: 文献查阅, 算法 & 模型设计, 数据采集 (设备 & 流程), 算法实现。
  - 自己的各方面能力 (文献检索, 算法, 工程) 都得到很大的锻炼和提升; 不畏困难的意志力得到磨练。
  - 掌握了视觉三维重建的基本知识; 深入了解了 3D 旋转和链式旋转的各种表达方式, 以及链式旋转的优化。

### [**比赛项目 – 人体摔倒识别(检测)**](https://github.com/Yusnows/tumblerElf) *技术负责人* 2019.12--2020.05

- **项目技术**: Python/Pytorch; Detection; faster rcnn variant.
- **项目简介**: 利用人工智能技术实现高准确率的视觉图像人体摔倒检测器。
- **项目工作**: 将直觉上的姿态估计问题建模为视觉目标检测问题，解决了摔倒姿态的歧义问题。
  - 把人体摔倒检测建模为目标检测问题, 算法可充分利用图像中的地面等信息进行摔倒判断; 数据标注也更简单。
  - 我们把图像中的所有人都进行检测, 分为摔倒和不摔倒两类; 这使得任务与预训练模型 gap 更小,
    同时人体特征无论是摔倒还是不摔倒的, 具有明显的一致性, 同时检测反而对于检测模型来说是更加容易的。
  - 使用两阶段目标检测框架, 使用 DCN, FPN, Cascade 等技术手段提升检测准确度。
- **项目难点**: 赛题没有明确的技术方向指向, 也不提供任何训练数据。
  - 数据稀缺。摔倒检测相关的标注数据是没有的; 赛事主办方也不提供相关的训练数据, 只有测试图像数据。
  - 人体姿态检测和判断的技术路线, 输入图像的人体姿态也跟模型训练的有较大差别; 摔倒姿态判断也十分困难。
- **项目成果**: 实现了高准确率的人体摔倒检测器。
  1. 参加并获得中国华录杯人体摔倒识别算法赛第一名。
  2. 参加并获得 OPPO TOP 高校创新科技大赛 5G 赛季三等奖。
- **个人收获**: 深入了解了目标检测和人体姿态估计这两类计算机视觉任务, 拓展了自己的知识边界。

### [**深度学习图像分类**](https://github.com/Yusnows/imcls) *项目负责人*  2019.10--2020.03

- **项目技术**: Python/Pytorch; Classification; CNN.
- **项目简介**: 借鉴已有的网络框架，设计并实现一个灵活方便的图像分类框架。%方便选择不同骨干网络，方便各种tricks的消融实验。
  - 软件框架分层设计，主要分为BackBone, Neck 和 Head 三个部分，层次清晰，方便增加或者修改。
  - 可以通过 yaml 配置文件完成大部分配置，不需要修改代码。
- **项目成果**: 实现了大部分预期功能，代码开源: [代码](https://github.com/Yusnows/imcls)。
    1. 参加并获得[世界人工智能创新大赛-菁英挑战赛](https://www.cvmart.net/race/8/rank)第一名。

## 获得奖项

- [**中国华录杯人体摔倒识别算法赛**](https://www.kesci.com/home/competition/5df99c5aea206700353c5de8/leaderboard) *第一名* 2020.04
- [**世界人工智能创新大赛-菁英挑战赛**](https://www.cvmart.net/race/8/rank) *第一名* 2020.07
- **OPPO TOP 高校创新科技大赛 5G 赛季** *三等奖* 2020.06
- [**首届全国“云状识别”算法大赛**](https://www.datafountain.cn/competitions/357/ranking?isRedance=0\&sch=1437) *第三名* 2019.11
- **南京大学优秀毕业生** *Top 15%* 2019.06
- **2018 年全国大学生模拟电子邀请赛** *全国 三等奖* 2018.08
- **2016 年江苏省大学生电子设计大赛 — TI 杯** *江苏省 一等奖* 2016.08
