---
layout: post
title: LeetCode Path Sum Serials I II III
date: 2019-04-02
tags: leetcode algorithm OJ
---

### 112. Path Sum

Given a binary tree and a sum, determine if the tree has a root-to-leaf path such that adding up all the values along the path equals the given sum.

**Note:** A leaf is a node with no children.

**Example:**

Given the below binary tree and `sum = 22`,

```
      5
     / \
    4   8
   /   / \
  11  13  4
 /  \      \
7    2      1
```

return true, as there exist a root-to-leaf path `5->4->11->2` which sum is 22.

#### 解题思路：

这题很简单，题目要求的路径是 root-to-leaf 的，所以很显然只需要深度优先搜索一遍就可以了，同时，由于题目只是问存不存在，因此只要一旦发现满足条件的路径就返回True就可以了，其他的节点就不用继续搜索了。

这题的一个细节就是：判断是否为叶子节点一点要在当前节点判断，而不能在下去一层判断是否为NULL。

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    bool hasPathSum(TreeNode* root, int sum) 
    {
        if(root==NULL)
            return false;
        if(root->left==NULL && root->right==NULL)
       		return root->val == sum;
        return hasPathSum(root->left, sum-root->val) || hasPathSum(root->right, sum-root->val);
    }
};
```

### 113. Path Sum II

Given a binary tree and a sum, find all root-to-leaf paths where each path's sum equals the given sum.

**Note:** A leaf is a node with no children.

**Example:**

Given the below binary tree and `sum = 22`,

```
      5
     / \
    4   8
   /   / \
  11  13  4
 /  \    / \
7    2  5   1
```

Return:

```
[
   [5,4,11,2],
   [5,8,4,5]
]
```

#### 解题思路

这题跟上面一题其实没有太大的区别，就是这里要求返回所有的满足要求的路径。我们只需要在第一题的基础上把路径记录下来（用一个vector\<int>就可以了），如果满足题目要求，就把记录的路径放到待返回的vector里面。

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<vector<int>> pathSum(TreeNode* root, int sum) 
    {
        vector<int> curpath;
        vector<vector<int>> pathes;
        pathSum(root, sum, curpath, pathes);
        return pathes;
    }
private:
    int pathSum(TreeNode* node, int sum, vector<int>& curpath, vector<vector<int>>& pathes)
    {
        if(node == NULL)
            return 0;
        curpath.push_back(node->val);
        if(node->left==NULL && node->right == NULL)
            if(node->val == sum)
                pathes.push_back(curpath);
        pathSum(node->left, sum-(node->val), curpath, pathes);
        pathSum(node->right, sum-(node->val), curpath, pathes);
        curpath.pop_back();
        return 0;
    }
};
```



### 437. Path Sum III

You are given a binary tree in which each node contains an integer value.

Find the number of paths that sum to a given value.

The path does not need to start or end at the root or a leaf, but it must go downwards (traveling only from parent nodes to child nodes).

The tree has no more than 1,000 nodes and the values are in the range -1,000,000 to 1,000,000.

**Example:**

```
root = [10,5,-3,3,2,null,11,3,-2,null,1], sum = 8

      10
     /  \
    5   -3
   / \    \
  3   2   11
 / \   \
3  -2   1

Return 3. The paths that sum to 8 are:

1.  5 -> 3
2.  5 -> 2 -> 1
3. -3 -> 11
```

#### 解题思路

这一题相比前面两题，我认为是要难一些，但是leetcode上面这题的难度是easy，而上面第二题的难度是medium。我也不知道是不是我自己的思维清奇了。

回到题目，为什么说这题更难一点呢，因为这题的路径是更加自由的：可以从任意节点开始，任意节点结束，只要是向下的就可以了。这样的路径明显满足要求的可能路径更加多了，在进行深度搜索的时候要考虑的情况也更加的多了。（至于为甚么是深度优先搜索，其实从路径是向下的就可以知道应该用深度优先搜索。但是这不是一个必要条件，只能说这个能给我们一些启示）。

我们先来看看，对于每一个节点，我们要考虑的问题有哪些：

1. 当前节点是否可能是一个路径的结束节点
2. 当前节点是否可能是一个路径的开始节点

对于第一个问题，还是跟第一第二题差不多（一样）的确定条件，只是现在是每一个节点都要确认一遍。对于第二个问题，其实我们没有办法在进行子节点的搜索之前就确定它是还是不是，所以我们可以一律认为它可能是，然后转化为子节点的第一个问题。因此我们就可以写出我们的代码：

```c++
class Solution {
public:
    int pathSum(TreeNode* root, int sum) 
    {
        pathSum(root, 0, sum, 0);
        return pathnum;
    }
private:
    int pathnum = 0;
    int pathSum(TreeNode* node, int presum, int sum, int flag)
    {
        if(node==NULL)
            return 0;
        int cursum = presum + node->val;
        if(cursum==sum) // 判断是否是某个路径的结束节点
        {
            pathnum++;
        }
        pathSum(node->left, cursum, sum, flag+1); //这里flag+1很重要，因为我们只希望在亲儿子节点处理第二种情况，而不能把问题一直延续孙子节点
        pathSum(node->right, cursum, sum, flag+1);
        if(flag==1)
        {
            pathSum(node, 0, sum, 0);
        }
        return 0;
    }
};
```

上面的flag主要是用来标志是否是作为“根节点”（因为路径不要求root-to-leaf，因此所有的节点都有可能成为“根节点”或者“叶节点”）以及是否作为子节点（不包括孙子节点）。因为我们希望如果是子节点的话，我们把它作为可能的路径开头，但是不能是孙子节点也这样处理，因为会重复。

上面的代码可能不够好理解，写得简洁一些就是：

```c++
class Solution {
public:
    int pathSum(TreeNode* root, int sum) 
    {
        if(root==NULL)
            return 0;
        pathSum(root, 0, sum);
        pathSum(root->left, sum);
        pathSum(root->right, sum);
        return pathnum;
    }
private:
    int pathnum = 0;
    int pathSum(TreeNode* node, int presum, int sum)
    {
        if(node==NULL)
            return 0;
        int cursum = presum + node->val;
        if(cursum==sum) // 判断是否是某个路径的结束节点
        {
            pathnum++;
        }
        pathSum(node->left, cursum, sum);
        pathSum(node->right, cursum, sum);
        return 0;
    }
};
```

通过把子节点作为开始节点的问题放到第一个函数，而第二个函数仅仅是从“根节点”一直搜索到叶节点。

但是其实我们可以发现，上面的算法的时间复杂度是$ O(n^2) $ 的。直觉高速我们还有时间复杂度上更加优秀的算法。

仔细分析，其实可以发现，上面的算法有很多的重复计算。这些重复计算是怎么产生的呢？用下面这个例子来分析。

```    
          10
         /  \
        5   -3
       / \    \
      3   2   11
     / \   \
    3  -2   1
```
以10作为根节点，计算到5这个节点的时候，这时会有两种情况，即开始提到的两种情况产生：

1. 当前节点5作为子节点继续向它的子节点搜索
2. 当前节点5重新作为“根节点”开始搜索

那么当到达节点3跟节点2的时候，就会有上面的来各种情况在做几乎相同的计算，唯一不同的地方就是presum可能不一样。

如果我们想要一次搜索就得到所有可能的路径，只要我们能够知道presum就可以了。比如到达3这个节点，如果我们知道了前面10-5这个路径前缀的所有presum，那么他可以很容易的知道自己是不是末尾节点，以及是多少条路径的末尾节点（开始节点不一样）。算法是$ curcum - 某一段的presum  == sum $。代码如下：

```c++
class Solution {
public:
    int pathSum(TreeNode* root, int sum) 
    {
        if(root==NULL)
            return 0;
        unordered_map<int, int> presum_hash;
        presum_hash[0] = 1;
        return pathSum(root, 0, sum, presum_hash);
    }
private:
    int pathSum(TreeNode* node, int presum, int sum, unordered_map<int, int>& presum_hash)
    {
        if(node==NULL)
            return 0;
        int cursum = presum + node->val;
        int res = presum_hash[cursum- sum];
        presum_hash[cursum]++;
        res += pathSum(node->left, cursum, sum, presum_hash);
        res += pathSum(node->right, cursum, sum, presum_hash);
        presum_hash[cursum]--;
        return res;
    }
};

```

