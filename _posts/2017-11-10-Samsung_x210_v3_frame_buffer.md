---
layout: post
title: Samsung_x210_v3 嵌入式开发(二):Frame Buffer
date: 2017-11-10
tags: 嵌入式Linux Samsung_x210
---

本项目的github地址: <a href="https://github.com/Yusnows/Samsung_X210v3">github</a> ，对应的代码在 LCD目录中。

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

### 1. Frame Buffer 原理
Frame Buffer 是出现在 Linux-2.2.xx 内核当中的一种驱动程序接口。Linux 是工作在保护模式下，所以用户态进程是无法像 DOS 那样使用显卡BIOS 里提供的中断调用来实现直接写屏， Linux 抽象出 Frame Buffer 这个设备来供用户态进程实现直接写屏。Frame buffer 机制模仿显卡的功能，将显卡硬件结构抽象掉，可以通过 Frame Buffer 的读写直接对显存进行操作。用户可以将Frame Buffer 看成是显示内存的一个映像，将其映射到进程地址空间之后，就可以直接进行读写操作，而写操作可以立即反应在屏幕上。这种操作是抽象的，统一的。用户不必关心物理显存的位置、换页机制等等具体细节，这些都是由Frame Buffer 设备驱动来完成的。 [1]   

按照显示屏的生能或显示模式区分，显示屏可以以单色或彩色显示。单色用 1 位来表示 (单色并不等于黑与白两种颜色，而是说只能以两种颜色来表示。通常取允许范围内颜色对比度最大的两种颜色)。彩色有 2、 4、 8、 16、 24、 32等位色。这些色调代表整个屏幕所有像素的颜色取值范围。如采用 8 位色/ 像素的显示模式，显示屏上能够出现的颜色种类最多只能有 256 种。究竞应该采取什么显示模式首先必须根据显示屏的性能，然后再由显示的需要来决定。这些因素会影响 Frame Buffer 空间的大小，因为 Frame Buffer 空间的计算大小是以屏幕的大小和显示模式来决定的。另一个影响 Frame Buffer 空间大小的因素是显示屏的单/双屏幕模式。   

单屏模式表示屏幕的显示范围是整个屏幕。这种显示模式只需一个 Frame Buffer 来存储整个屏幕的显示内容，并且只需一个通道来将 Frame Buffer 内容传输到显示屏上(Frame Buffer 的内容可能需要被处理后再传输到显示屏)。双屏模式则将整个屏幕划分成两部分。它有别于将两个独立的显示屏组织成一个显示屏。 单看其中一部分，它们的显示方式是与单屏方式一致的，并且两部分同时扫描，工作方式是独立的。这两部分都各自有 Frame Buffer，且他们的地址无需连续(这里指的是下半部的 Frame Buffer 的首地址无需紧跟在上半部的地址未端)，并且同时具有独立的两个通道将 Frame Buffer 的数据传输到显示屏。 [2]  

### 2. Frame Buffer 相关操作
在应用程序中，一般通过将 Frame Buffer 设备映射到进程地址空间的方式使用，比如下面的程序就打开 /dev/fb0 设备，并通过 ` mmap()` 系统调用进行地址映射，随后用` memset() `将屏幕清空。因此，对 Frame Buffer 的操作其实与对变量（数组变量）的操作没有什么差别。我们想要在屏幕上正确显示我们想要显示的内容，关键是找到映射到程序空间的内存与像素之间的对应关系。  

#### 2.1 获取 Frame Buffer 的大小

Frame Buffer 的大小可以通过以下公式计算：
$$ FrameBufferSize= \frac{Width \times Height \times BitperPixel}{8} $$其中，Width、Height 分别对应显示屏的宽和高的像素值，Bits_per_Pixel是显示屏的位色，其值可能为上面提到的 2、4、8、16、24 或 32。  

有了 Width、Height 和 Bits_per_Pixel 这三个量，我们就可以算出 Frame Buffer的大小了。 该三者的值可以通过 ioctl()系统调用获得。  

主要的流程如下代码所示：
```
struct fb_var_screeninfo vinfo;
ioctl(fd,FBIOGET_VSCREENINFO,&(vinfo));
xres=vinfo.xres;
yres=vinfo.yres;
bits_per_pixel=vinfo.bits_per_pixel;
ScreenSize=((xres)*(yres)*(bits_per_pixel))/8;
```

#### 2.2 Frame Buffer 地址映射

得到了 Frame Buffer 的大小之后，我们就可完成 Fame Buffer 与用户程序内存之间的地址映射了。  

通过 mmap() 系统调用便可以将 Frame Buffer 与内存的地址映射起来，mmap() 返回的是一个 void 指针。代码如下：
```
fbp=(unsigned char*)mmap(0,ScreenSize,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
```

#### 2.3 Frame Buffer 画点

将 Frame Buffer 进行地址映射之后，我们便有可能通过简单的对数组赋值操作在屏幕上显示我们想要的内容。显然，无论多么复杂的显示任务，都是由一个个点构成的。 因此， 我们首先要实现的是画点的方法。  

想要在屏幕上画一个点， 首先要知道 Frame Buffer 映射到的内存的每一个 bit 与屏幕每一个像素的 RGB 的对应关系。这个关系一般可以通过显示屏的数据手册 (datasheet) 获得，也可以通过程序的方式获得。通过程序获得一般需要一些经验和猜测： 首先我们可以获得 Bits_per_Pixel, 然后根据一般计算机色位的组织方式可以大概推测 RGB 各对应多少位以及 RGB 的排列顺序，然后通过实现现象做相应的调整即可。  

画点的主要代码部分如下：

```
inline int lcddev_t::drawPoint(const unsigned int& x,const unsigned int& y,const Color_t color)
{
  ...............................
  if(bits_per_pixel==24)
  {
    offset=(y*(xres)+x)*(bits_per_pixel)/8;
    *(unsigned char *)(fbp+offset+0)=(unsigned char)color.B;
    *(unsigned char *)(fbp+offset+1)=(unsigned char)color.G;
    *(unsigned char *)(fbp+offset+2)=(unsigned char)color.R;
  } 
  else if(bits_per_pixel==16)
  {
    offset=(y*xres+x)*(bits_per_pixel)/8;
    int color_i=(color.R<<11)|(color.G<<5)|(color.B&0x1f);
    *(unsigned char *)(fbp+offset+0)=(unsigned char)((color_i)&0xff);
    *(unsigned char *)(fbp+offset+1)=(unsigned char)((color_i>>8)&0xff);
  } 
  else if(bits_per_pixel==32)
  {
    offset=(y*xres+x)*(bits_per_pixel)/8;
    *(unsigned char *)(fbp+offset+0)=(unsigned char)color.B;
    *(unsigned char *)(fbp+offset+1)=(unsigned char)color.G;
    *(unsigned char *)(fbp+offset+2)=(unsigned char)color.R;
    *(unsigned char *)(fbp+offset+3)=(unsigned char)color.A;
  } 
  return 0;
}
```

## 实验步骤

### 1. 程序编写

Frame Buffer 操作的程序采用 C++面向对象的方式进行编写。主要数据成员如下：
```
protected:
	int numid;
	char isInit;
	char isFbpMap; //记录是否进行了内存映射
	unsigned char *fbp; //内存映射后的指针
	int fd; //打开的文件的 id（ 如 fd=open(“/dev/fb0”)）
	unsigned int ScreenSize;
	unsigned int xres;
	unsigned int yres;
	unsigned int bits_per_pixel;
	char* devfile; //所打开的文件路径以及文件名
```

主要的成员函数（操作接口）有：
```
inline int showChar(......); //显示字符函数
inline int drawCircle(......); //画圆函数
inline int drawRect(......); //画矩形函数
inline int drawLine(......); //画线函数
inline int drawPoint(......); //画点函数
inline int clearPoint(......); //清除点函数， 用处不大， 可用画点函数替代
inline int drawAera(......); //画一个特定区域， 填充形式
inline int clearAeratoWhite(......); //将一个区域清除为白色
inline int clearAeratoBlack(......); //讲一个区域清除为黑色
inline int clearAlltoWhite(); //将整个屏幕清除为白色
inline int clearAlltoBlack(); //将整个屏幕清除为黑色
inline int clearAll(); //将整个屏幕清除为黑色
```
注： 以上函数的参数均已经省略，因为实际程序中将这些函数都重载了多种参数形式的，方便不同场景下调用。另外，出于效率方面考虑，这些函数均声明为 inline 函数。以上便是 Frame Buffer 较为底层操作的程序的主要数据成员以及函数，具体程序源码可以在<a href="https://github.com/Yusnows/Samsung_X210v3/tree/master/LCD/">github</a>中下载。其中，有效的文件为tftlcd.cpp 以及 tftlcd.hpp 。main.cpp 为测试文件。



### 参考文献

[1] hedtao “Frame buffer 详解” CSDN 博客
[2] 南京大学 电子科学与工程学院 嵌入式系统 X210v3 实验指导书