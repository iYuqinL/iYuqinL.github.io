---
layout: post
title: Samsung_x210_v3 嵌入式开发(二):触摸屏(touch screen)--tslib 移植
date: 2017-11-12
tags: 嵌入式 触摸屏 touch_screen arm Linux tslib
---

## 实验环境
### 1. 目标平台环境
#### 1.1 硬件环境
* CPU: Cortex-A8, 1GHz
* memery: 512MB
* Flash: 4GB inand-Flash
* 24-bit RGB interface
* Four USB HOST interface
* USB OTG interface
* Two RS232 interface
* Two SDcard interface
* Four LED
* Function Key
* One HDMI video interface
* Audio input/output interface
* Wired internet interface DM9000CEP
* Capacity touch screen
* Camera interface
* Real time clock
* Support for G-sensor, SPI, UART, USB/WIFI, GPS, GPRS, etc.
* 2D/3D high performance graphic process

#### 1.2 软件环境
* Operator System: Linux2.6.35
* File System: busy_box1.23.0

### 2. Host 环境
#### 2.1 操作系统
* ubuntu 16.04 LTS
* kernel 4.10.37

#### 2.2 编译器
* arm-linux-gcc (Freescale MAD -- Linaro 2011.07 -- Built at 2011/08/10 09:20) 4.6.2 20110630 (prerelease)
* GNU Make 4.1
* cmake version 3.5.1

## 实验原理

### 1. 触摸屏原理 [1]

#### 1.1 触摸屏分类

1. 常见的触摸屏分为 2 种：电阻式触摸屏和电容式触摸屏。早期用电阻式触摸屏，后来发明了电容式触摸屏。 
2. 这两种的特性不同、接口不同、编程方法不同、原理不同。

#### 1.2 电阻式触摸屏

电阻式触摸屏其实就是一种传感器，虽然已经用的不多了，但是还是有过很多的 LCD 模块采用电阻式触摸屏，这种屏幕可以用四线、 五线、七线或八线来产生屏幕偏置电压，同时读回触摸点的电压， 在这里主要以四线为例进行说明。  

##### 1.2.1 电阻屏电平转换原理

ITO 是一种材料，其实是一种涂料，特点就是透明、导电、均匀涂抹。(如图 1 所示) 本来玻璃和塑料都是不导电的，但是涂上 ITO 之后就变成导电了（同时还保持着原来透明的特性）。ITO 不但导电而且有电阻，所以中间均匀涂抹了 ITO 之后就相当于在同一层的两边之间接了一个电阻。因为 ITO 形成的等效电阻在整个板上是均匀分布的，所在在板子上某一点的电压值和这一点的位置值成正比。触摸屏经过操作，按下之后要的就是按下的坐标，坐标其实就是位置信息，这个位置信息和电压成正比了，而这一点的电压可以通过 AD 转换得到。这就是整个电阻式触摸屏的工作原理。

>
![电阻屏](/images/posts/Samsung_x210/touchscreen/01.png  "电阻屏")

在第一个面板的一对电极上加电压，然后在另一个面板的一个电极和第一个面板的地之间去测量。在没有按下时测试无结果，但是在有人按下时在按下的那一点 2 个面板接触，接触会导致第二个面板上整体的电压值和接触处的电压值相等，所以此时测量到的电压就是接触处在第一个面板上的电压值。以上过程在一个方向进行一次即可测得该方向的坐标值，进行完之后撤掉电压然后在另一个方向的电极上加电压，故伎重施，即可得到另一个方向的坐标。至此一次触摸事件结束。例如下图所示：我们先在 X+ 和 X-之间加上一个电压，当有人按下触摸屏之后就会在相应的位置形成一个触点，那么此时我们去测量 Y+与 GND（或者是 Y-与 GND）之间的电压，那么其实得到的电压值就是发生触点处的电压值，因为电阻是均匀分布的， 所以可以算出该点在 x 方向上的位置；同理测量 Y 轴也是一样的道理。

>
![电阻屏](/images/posts/Samsung_x210/touchscreen/02.png  "电阻屏")


