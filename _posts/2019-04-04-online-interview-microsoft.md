---
layout: post
title: 2019 Microsoft Internship on line code test I
date: 2019-04-02
tags: algorithm OJ internship
---

### 1. Playing with Beads

 There are N people in a group labeled from 1 to N. People are connected to each other via threads in the following manner:

The person with label K is connected to all the people with label J such that J exactly divides K. Beads can be passed through the threads. if a person P has a bead, in how many ways can the bead be passed in the network of threads so that it return to the same person within X moves on less.

*MOVE*: Passing the bead from one person to the other.

**Input Specification:**

```
input1: N, denoting the number of people
input2: P, label of the people having bead
input3: X, Maximum number of moves that can be made
```

**output Specification:**

```
Your function should return the total number of ways in which the bead will retrun to the initial position within X moves.
```

**Example 1:**

```
input1: 3
input2: 2
input3: 2
```

```
output: 1
```

**Explanation:**

```
Only one way:
2->1->2
```

**Example 2:**

```
input1: 3
input2: 2
input3: 4
```

```
output: 3
```

**Explanation:**

```
Ways:
2->1->2
2->1->2->1->2
2->1->3->1->2
```

#### 解题思路：

这题的题目有点绕。其实这题就是一个图的可达路径问题。

![graph-K-strides](/images/posts/MicrosoftInternship/graph-K-strides.jpg)

按照上面的算法，我们很容就写出代码了：

```c++
#include <iostream>
#include <vector>
#include<stdlib.h>

using namespace std;
typedef vector<vector<int> > mat_t;
mat_t matmul(mat_t& matA, mat_t& matB)
{
    int rows = matA.size();
    int cols = rows;
    mat_t ret(rows, vector<int>(cols, 0));
    for(int i=0;i<rows;++i)
    {
        for(int j=0;j<cols;++j)
        {
            for(int k=0; k<cols; ++k)
                ret[i][j] += matA[i][k] * matB[k][j];
        }
    }
    return ret;
}

int maxCircles(int input1, int input2, int input3)
{
    // 构建图的邻接矩阵
    mat_t map(input1, vector<int>(input1, 0));
    mat_t res(input1, vector<int>(input1, 0));
    for(int i=0; i<input1; ++i)
    {
		for(int j=0; j<input1;++j)
		{
			if(((i+1)%(j+1)==0||(j+1)%(i+1)==0)&&i!=j)
            {
                map[i][j] = 1;
                res[i][j] = 1;
            }
		}
	}
    int ways = 0;
    for(int i=0;i<input3-1;++i)
    {
        res = matmul(res, map);
        ways += res[input2-1][input2-1];
    }
    return ways;
}

int main(int argc, char const *argv[])
{
    if(argc != 4)
        return -1;
    int argv_i[3];
    for(int i=1; i<argc; ++i)
    {
        argv_i[i-1] = atoi(argv[i]);
    }
	int ret = maxCircles(argv_i[0], argv_i[1], argv_i[2]);
	cout<<ret<<endl;
	return 0;
}
```

很容易看出，上面算法的时间复杂度为$ O(n^3m)$，n为图矩阵的大小，m为最多移动的步数。

这个时间复杂度应该说不是最优的，因为这种方法不仅计算了所需节点的路径数，而是计算了所有节点到达所有节点的路径数。 

我们其实可以用深度优先搜索的方法来做这一道题。代码如下：

```c++
#include<iostream>
#include<vector>
#include<stdlib.h>
using namespace std;

int travelMap(int it, int owner, int times, vector<vector<int> > & map);

int maxCircles(int input1, int input2, int input3)
{
    vector<vector<int> > map;
    int ret = 0;
    for(int i=1; i <= input1; ++i)
    {
        vector<int> tmp;
        for(int j=input1; j>0;--j)
        {
            if((i%j==0||j%i==0)&&i!=j)
                tmp.push_back(j);
        }
        map.push_back(tmp);
    }
    int it = input2;
    for(int i=0; i<map[it-1].size();++i)
    {
        ret += travelMap(map[it-1][i], input2-1, input3-1, map);
    }
    return ret;
}


// 深度优先
int travelMap(int it, int owner, int times, vector<vector<int> > & map)
{
    if(times <=0)
        return it-1 == owner;
    int res = 0;
    if(it-1 == owner) //遇到了一个环
        res ++;
    times = times - 1;
    for(int i=0; i<map[it-1].size();++i)
    {
        res += travelMap(map[it-1][i], owner, times, map);
    }
    return res;
}

int main(int argc, char const *argv[])
{
    if(argc != 4)
        return -1;
    int argv_i[3];
    for(int i=1; i<argc; ++i)
    {
        argv_i[i-1] = atoi(argv[i]);
    }
    int ret = maxCircles(argv_i[0], argv_i[1], argv_i[2]);
    cout<<ret<<endl;
    return 0;
}
```

这个算法的时间复杂度为 $ O(VEm)$ 。V为图的节点个数，这里就是people的个数；E为图的边数，这里即为threads的个数；m为m为最多移动的步数。因为这里的图的边数是有限制条件的，一个节点只联通能够整除它和被他整除的数。所以 $E$ 跟 $V^2$ 不是一个数量级的。

如下是people为20的时候的邻接矩阵的情况：

```
0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 
1 0 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 
1 0 0 0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0 
1 1 0 0 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1 
1 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 
1 1 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 
1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 
1 1 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 
1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 
1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 
1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
1 1 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
1 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 
1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
1 1 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 
1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
1 1 1 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 
1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
1 1 0 1 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 
```

如下是people为40的时候的邻接矩阵的情况：

```
0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 
1 0 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 
1 0 0 0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 
1 1 0 0 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1 
1 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 
1 1 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 
1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 
1 1 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 
1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 
1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 
1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 
1 1 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 
1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 
1 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 
1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 
1 1 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 
1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 
1 1 1 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 
1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 
1 1 0 1 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 
1 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
1 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
1 1 1 1 0 1 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
1 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
1 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
1 1 0 1 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
1 1 1 0 1 1 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
1 1 0 1 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
1 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
1 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
1 1 1 1 0 1 0 0 1 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
1 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
1 1 0 1 1 0 0 1 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
```

但是但你测试的时候，你会惊奇的发现，第二个算法居然比第一个慢很多很多。说明我们的时间复杂度的分析错误了。

为什么呢，因为这个图是有环的，因此它的时间复杂度并不是 $ O(VEm)$ , 而是 $ O(VE^m)$ 。

因此还是矩阵的相乘的方法更加优秀。其实在有环图的情况下，深度优先搜寻如果不加标志的搜索往往会陷入指数级的时间复杂度。