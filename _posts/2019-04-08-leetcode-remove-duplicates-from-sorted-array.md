---
layout: post
title: LeetCode Move Zeroes & Remove Element & Remove Duplicates from Sorted Array
date: 2019-04-08
tags: leetcode
---

### 283. Move Zeroes

Given an array `nums`, write a function to move all `0`'s to the end of it
while maintaining the relative order of the non-zero elements.

**Example:**

```plaintxt
Input: [0,1,0,3,12]
Output: [1,3,12,0,0]
```

**Note**:

1. You must do this **in-place** without making a copy of the array.
2. Minimize the total number of operations.

#### **解题思路：**

这题的题目很好理解，就是把所有的零放到数组的末尾。
一个最直接的办法就是新开一个数组ret，然后遍历所给的数组，遇到不是零的元素就放到新开的数组ret的末尾，
遍历结束后再push $nums.size() - ret.size()$ 个0到ret的末尾 ，然后令 $nums=ret$ 即可。

但是这个算法不符合题目 **in-place** 的要求。我们把上面的算法换一个思路，变成 **in-place** 的。
只需要增加一个指针$lastzero$，这个指针始终指向遍历过程中最后一个有可能是0的元素。
每次遇到不是0的元素都执行 $nums[lastzero++] = nums[i]$。遍历结束后，再从$lastzero$开始往后将所有元素置零。

```c++
class Solution {
public:
    void moveZeroes(vector<int>& nums) 
    {
        int lastzero = 0;
        for(int i=0;i<nums.size();++i)
        {
            if(nums[i] != 0)
                nums[lastzero++] = nums[i];
        }
        for(int i=lastzero;i<nums.size();++i)
            nums[i] = 0;
    }
};
```

### 27. Remove Element

Given an array *nums* and a value *val*, remove all instances of that value
[**in-place**](https://en.wikipedia.org/wiki/In-place_algorithm) and
return the new length.

Do not allocate extra space for another array, you must do this by
**modifying the input array in-place** with O(1) extra memory.

The order of elements can be changed. It doesn't matter what
you leave beyond the new length.

**Example 1:**

```plaintxt
Given nums = [3,2,2,3], val = 3,

Your function should return length = 2, with the first two elements of nums being 2.

It doesn't matter what you leave beyond the returned length.
```

**Example 2:**

```plaintxt
Given nums = [0,1,2,2,3,0,4,2], val = 2,

Your function should return length = 5,
with the first five elements of nums containing 0, 1, 3, 0, and 4.

Note that the order of those five elements can be arbitrary.

It doesn't matter what values are set beyond the returned length.
```

**Clarification:**

Confused why the returned value is an integer but your answer is an array?

Note that the input array is passed in by **reference**,
which means modification to the input array will be known to the caller as well.

Internally you can think of this:

```plaintxt
// nums is passed in by reference. (i.e., without making a copy)
int len = removeElement(nums, val);

// any modification to nums in your function would be known by the caller.
// using the length returned by your function, it prints the first len elements.
for (int i = 0; i < len; i++) {
    print(nums[i]);
}
```

#### **解题思路**

这题跟上一题的解题思路是十分类似的。

```c++
class Solution {
public:
    int removeElement(vector<int>& nums, int val) 
    {
        int lastNotVal = 0;
        for(int i=0;i<nums.size();++i)
        {
            if(nums[i] != val)
                nums[lastNotVal++] = nums[i];
        }
        return lastNotVal;
    }
};
```

### 26. Remove Duplicates from Sorted Array

Given a sorted array *nums*, remove the duplicates
[**in-place**](https://en.wikipedia.org/wiki/In-place_algorithm)
such that each element appear only *once* and return the new length.

Do not allocate extra space for another array, you must do this by
**modifying the input array in-place** with O(1) extra memory.

**Example 1:**

```plaintxt
Given nums = [1,1,2],

Your function should return length = 2,
with the first two elements of nums being 1 and 2 respectively.

It doesn't matter what you leave beyond the returned length.
```

**Example 2:**

```plaintxt
Given nums = [0,0,1,1,1,2,2,3,3,4],

Your function should return length = 5,
with the first five elements of nums being modified to 0, 1, 2, 3, and 4 respectively.

It doesn't matter what values are set beyond the returned length.
```

**Clarification:**

Confused why the returned value is an integer but your answer is an array?

Note that the input array is passed in by **reference**,
which means modification to the input array will be known to the caller as well.

Internally you can think of this:

```plaintxt
// nums is passed in by reference. (i.e., without making a copy)
int len = removeDuplicates(nums);

// any modification to nums in your function would be known by the caller.
// using the length returned by your function, it prints the first len elements.
for (int i = 0; i < len; i++) {
    print(nums[i]);
}
```

#### **解题思路**

经过前面两题，这一题其实也是很简单的。

```c++
class Solution {
public:
    int removeDuplicates(vector<int>& nums) 
    {
        if(nums.size()<=1)
            return nums.size();
        int notdupl= 1;
        for(int i=1;i<nums.size();++i)
        {
            if(nums[i]!=nums[notdupl-1])
                nums[notdupl++] = nums[i];
        }
        return notdupl;
    }
};
```

### 80. Remove Duplicates from Sorted Array II

Given a sorted array *nums*, remove the duplicates
[**in-place**](https://en.wikipedia.org/wiki/In-place_algorithm)
such that duplicates appeared at most *twice* and return the new length.

Do not allocate extra space for another array, you must do this by
**modifying the input array in-place** with O(1) extra memory.

**Example 1:**

```plaintxt
Given nums = [1,1,1,2,2,3],

Your function should return length = 5,
with the first five elements of nums being 1, 1, 2, 2 and 3 respectively.

It doesn't matter what you leave beyond the returned length.
```

**Example 2:**

```plaintxt
Given nums = [0,0,1,1,1,1,2,3,3],

Your function should return length = 7,
with the first seven elements of nums being modified to 0, 0, 1, 1, 2, 3 and 3 respectively.

It doesn't matter what values are set beyond the returned length.
```

**Clarification:**

Confused why the returned value is an integer but your answer is an array?

Note that the input array is passed in by **reference**,
which means modification to the input array will be known to the caller as well.

Internally you can think of this:

```plaintxt
// nums is passed in by reference. (i.e., without making a copy)
int len = removeDuplicates(nums);

// any modification to nums in your function would be known by the caller.
// using the length returned by your function, it prints the first len elements.
for (int i = 0; i < len; i++) {
    print(nums[i]);
}
```

#### **解题思路**

与上题类似，这里只需要把notdupl改成2就行了。

```c++
class Solution {
public:
    int removeDuplicates(vector<int>& nums) 
    {
        if(nums.size()<=2)
            return nums.size();
        int notdupl= 2;
        for(int i=2;i<nums.size();++i)
        {
            if(nums[i]!=nums[notdupl-2])
                nums[notdupl++] = nums[i];
        }
        return notdupl;
    }
};
```
