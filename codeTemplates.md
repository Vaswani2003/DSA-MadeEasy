# 🧰 DSA Code Templates (Python + C++)

## 🔄 Two Pointers (Two Inputs)

**When to use:** Merging two sorted arrays, finding intersection/union of arrays, or comparing elements from two different data structures.

**Time Complexity:** O(n + m)  
**Space Complexity:** O(1) for comparison, O(n + m) for merged result clean and reusable code templates for common DSA patterns. Each section includes a quick explanation, followed by both **Python** and **C++** code snippets.

---

## 📁 Table of Contents

- [🫡 Two Pointers (One Input, Opposite Ends)](#-two-pointers-one-input-opposite-ends)
- [🔄 Two Pointers (Two Inputs)](#-two-pointers-two-inputs)
- [🚪 Sliding Window](#-sliding-window)
- [➕ Prefix Sum](#-prefix-sum)
- [🌟 Efficient String Building](#-efficient-string-building)
- [🐢 Linked List (Fast & Slow Pointer)](#-linked-list-fast--slow-pointer)
- [🔁 Reversing a Linked List](#-reversing-a-linked-list)
- [🔍 Find Subarrays with Criteria](#-find-subarrays-with-criteria)
- [🔽 Monotonic Stack](#-monotonic-stack)
- [🌳 Binary Tree DFS (Recursive)](#-binary-tree-dfs-recursive)
- [🌲 Binary Tree DFS (Iterative)](#-binary-tree-dfs-iterative)
- [🌳 Binary Tree BFS](#-binary-tree-bfs)
- [📚 Graph DFS/BFS](#-graph-dfsbfs)
- [🔁 Backtracking](#-backtracking)
- [🔍 Binary Search Variants](#-binary-search-variants)
- [🎓 DP - Top Down (Memoization)](#-dp---top-down-memoization)
- [📊 Heap Usage](#-heap-usage)

---

## 🫡 Two Pointers (One Input, Opposite Ends)

**When to use:** Problems involving palindromes, pairs with target sum, or comparing elements from both ends of an array.

**Time Complexity:** O(n)  
**Space Complexity:** O(1)

### Python

```python
def two_pointers(nums):
    left, right = 0, len(nums) - 1
    while left < right:
        # Do something with nums[left] and nums[right]
        left += 1
        right -= 1
```

### C++

```cpp
void twoPointers(vector<int>& nums) {
    int left = 0, right = nums.size() - 1;
    while (left < right) {
        // Do something with nums[left] and nums[right]
        ++left;
        --right;
    }
}
```

---

## � Two Pointers (Two Inputs)

### Python

```python
def merge_two_sorted(arr1, arr2):
    i, j = 0, 0
    merged = []
    while i < len(arr1) and j < len(arr2):
        if arr1[i] < arr2[j]:
            merged.append(arr1[i])
            i += 1
        else:
            merged.append(arr2[j])
            j += 1
    merged.extend(arr1[i:])
    merged.extend(arr2[j:])
    return merged
```

### C++

```cpp
vector<int> mergeTwoSorted(vector<int>& arr1, vector<int>& arr2) {
    int i = 0, j = 0;
    vector<int> merged;
    while (i < arr1.size() && j < arr2.size()) {
        if (arr1[i] < arr2[j])
            merged.push_back(arr1[i++]);
        else
            merged.push_back(arr2[j++]);
    }
    while (i < arr1.size()) merged.push_back(arr1[i++]);
    while (j < arr2.size()) merged.push_back(arr2[j++]);
    return merged;
}
```

---

## 🚪 Sliding Window

**When to use:** Finding subarrays with specific properties (max sum, contains all characters, etc.), substring problems.

**Time Complexity:** O(n)  
**Space Complexity:** O(1) to O(k) depending on window contents

### Python

```python
def max_subarray_sum(nums, k):
    window_sum = sum(nums[:k])
    max_sum = window_sum
    for i in range(k, len(nums)):
        window_sum += nums[i] - nums[i - k]
        max_sum = max(max_sum, window_sum)
    return max_sum
```

### C++

```cpp
int maxSubarraySum(vector<int>& nums, int k) {
    int window_sum = 0, max_sum = INT_MIN;
    for (int i = 0; i < k; ++i) window_sum += nums[i];
    max_sum = window_sum;
    for (int i = k; i < nums.size(); ++i) {
        window_sum += nums[i] - nums[i - k];
        max_sum = max(max_sum, window_sum);
    }
    return max_sum;
}
```

---

## ➕ Prefix Sum

**When to use:** Range sum queries, subarray sum problems, or when you need to quickly calculate cumulative values.

**Time Complexity:** O(n) to build, O(1) for range queries  
**Space Complexity:** O(n)

### Python

```python
```python
def prefix_sum(nums):
    prefix = [0]
    for num in nums:
        prefix.append(prefix[-1] + num)
    return prefix

def range_sum_query(prefix, left, right):
    """Query sum of elements from index left to right (inclusive)"""
    return prefix[right + 1] - prefix[left]
```

### C++

```cpp
vector<int> prefixSum(const vector<int>& nums) {
    vector<int> prefix(nums.size() + 1, 0);
    for (int i = 0; i < nums.size(); ++i)
        prefix[i + 1] = prefix[i] + nums[i];
    return prefix;
}

int rangeSumQuery(const vector<int>& prefix, int left, int right) {
    // Query sum of elements from index left to right (inclusive)
    return prefix[right + 1] - prefix[left];
}
```

---

## 🌟 Efficient String Building

**When to use:** Concatenating many strings, building result strings character by character, avoiding O(n²) string concatenation.

**Time Complexity:** O(n) where n is total length  
**Space Complexity:** O(n)

### Python

```python
def build_string(parts):
    return ''.join(parts)

# Alternative using list for character-by-character building
def build_string_chars(chars):
    return ''.join(chars)
```

### C++

```cpp
#include <sstream>
string buildString(const vector<string>& parts) {
    ostringstream oss;
    for (const auto& part : parts) oss << part;
    return oss.str();
}
```

---

## 🐢 Linked List (Fast & Slow Pointer)

**When to use:** Detecting cycles, finding middle element, checking if list has specific length properties.

**Time Complexity:** O(n)  
**Space Complexity:** O(1)

### Python

```python
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

def find_middle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow
```

### C++

```cpp
struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};
bool hasCycle(ListNode* head) {
    ListNode *slow = head, *fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) return true;
    }
    return false;
}
```

---

## 🔁 Reversing a Linked List

**When to use:** Reversing entire list, reversing portions of a list, palindrome checking.

**Time Complexity:** O(n)  
**Space Complexity:** O(1)

### Python

```python
def reverse_list(head):
    prev = None
    curr = head
    while curr:
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt
    return prev
```

### C++

```cpp
ListNode* reverseList(ListNode* head) {
    ListNode* prev = nullptr;
    while (head) {
        ListNode* nxt = head->next;
        head->next = prev;
        prev = head;
        head = nxt;
    }
    return prev;
}
```

---

## 🔍 Find Subarrays with Criteria

**When to use:** Finding all subarrays that meet specific conditions, brute force subarray enumeration.

**Time Complexity:** O(n²) to O(n³)  
**Space Complexity:** O(1) to O(n) depending on storage

### Python

```python
def find_subarrays(nums, k):
    result = []
    for i in range(len(nums)):
        for j in range(i, len(nums)):
            subarray = nums[i:j+1]
            if meets_criteria(subarray, k):  # Define your criteria
                result.append(subarray)
    return result

def count_subarrays_with_sum(nums, target):
    count = 0
    for i in range(len(nums)):
        current_sum = 0
        for j in range(i, len(nums)):
            current_sum += nums[j]
            if current_sum == target:
                count += 1
    return count
```

### C++

```cpp
vector<vector<int>> findSubarrays(vector<int>& nums, int k) {
    vector<vector<int>> result;
    for (int i = 0; i < nums.size(); ++i) {
        for (int j = i; j < nums.size(); ++j) {
            if (meetsCriteria(nums, i, j, k))
                result.push_back(vector<int>(nums.begin() + i, nums.begin() + j + 1));
        }
    }
    return result;
}
```

---

## 🔽 Monotonic Stack

**When to use:** Next/previous greater/smaller element problems, finding spans, calculating areas in histograms.

**Time Complexity:** O(n)  
**Space Complexity:** O(n)

### Python

```python
def next_greater(nums):
    stack = []  # Store indices
    res = [-1] * len(nums)
    for i, num in enumerate(nums):
        # While stack not empty and current element > element at stack top
        while stack and nums[stack[-1]] < num:
            res[stack.pop()] = num
        stack.append(i)
    return res

def daily_temperatures(temperatures):
    stack = []
    result = [0] * len(temperatures)
    for i, temp in enumerate(temperatures):
        while stack and temperatures[stack[-1]] < temp:
            prev_index = stack.pop()
            result[prev_index] = i - prev_index
        stack.append(i)
    return result
```

### C++

```cpp
vector<int> nextGreater(vector<int>& nums) {
    vector<int> res(nums.size(), -1);
    stack<int> st;
    for (int i = 0; i < nums.size(); ++i) {
        while (!st.empty() && nums[st.top()] < nums[i]) {
            res[st.top()] = nums[i];
            st.pop();
        }
        st.push(i);
    }
    return res;
}
```

---

## 🌳 Binary Tree DFS (Recursive)

**When to use:** Tree traversal, path finding, calculating tree properties (height, diameter), tree modification.

**Time Complexity:** O(n) where n is number of nodes  
**Space Complexity:** O(h) where h is height (O(log n) for balanced, O(n) for skewed)

### Python

```python
def dfs_recursive(node):
    if not node:
        return
    # Preorder: Process node first
    print(node.val)
    dfs_recursive(node.left)
    dfs_recursive(node.right)

def max_depth(root):
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))

def has_path_sum(root, target_sum):
    if not root:
        return False
    if not root.left and not root.right:  # Leaf node
        return root.val == target_sum
    return (has_path_sum(root.left, target_sum - root.val) or 
            has_path_sum(root.right, target_sum - root.val))
```

### C++

```cpp
void dfsRecursive(TreeNode* node) {
    if (!node) return;
    // Visit node
    dfsRecursive(node->left);
    dfsRecursive(node->right);
}
```

---

## 🌲 Binary Tree DFS (Iterative)

**When to use:** When recursion might cause stack overflow, or when you need explicit control over traversal order.

**Time Complexity:** O(n)  
**Space Complexity:** O(h) for stack

### Python

```python
def dfs_iterative(root):
    if not root:
        return
    stack = [root]
    while stack:
        node = stack.pop()
        print(node.val)  # Process node
        # Add right first, then left (so left is processed first)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)

def inorder_iterative(root):
    stack = []
    current = root
    while stack or current:
        # Go to leftmost node
        while current:
            stack.append(current)
            current = current.left
        # Process node
        current = stack.pop()
        print(current.val)
        # Move to right subtree
        current = current.right
```

### C++

```cpp
void dfsIterative(TreeNode* root) {
    stack<TreeNode*> st;
    st.push(root);
    while (!st.empty()) {
        TreeNode* node = st.top(); st.pop();
        if (node) {
            // Visit node
            st.push(node->right);
            st.push(node->left);
        }
    }
}
```

---

## 🌳 Binary Tree BFS

**When to use:** Level-order traversal, finding shortest path in unweighted trees, level-by-level processing.

**Time Complexity:** O(n)  
**Space Complexity:** O(w) where w is maximum width of tree

### Python

```python
def bfs(root):
    if not root:
        return
    from collections import deque
    queue = deque([root])
    while queue:
        node = queue.popleft()
        print(node.val)  # Process node
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)

def level_order_traversal(root):
    if not root:
        return []
    from collections import deque
    result = []
    queue = deque([root])
    while queue:
        level_size = len(queue)
        level = []
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    return result
```

### C++

```cpp
void bfs(TreeNode* root) {
    queue<TreeNode*> q;
    q.push(root);
    while (!q.empty()) {
        TreeNode* node = q.front(); q.pop();
        if (node) {
            // Visit node
            q.push(node->left);
            q.push(node->right);
        }
    }
}
```

### Python

```python
```python
def dfs(node, visited, graph):
    if node in visited:
        return
    visited.add(node)
    print(node)  # Process node
    for neighbor in graph[node]:
        dfs(neighbor, visited, graph)

def bfs(start, graph):
    from collections import deque
    queue = deque([start])
    visited = set([start])
    while queue:
        node = queue.popleft()
        print(node)  # Process node
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

### C++

```cpp
void dfs(int node, unordered_set<int>& visited, vector<vector<int>>& graph) {
    if (visited.count(node)) return;
    visited.insert(node);
    for (int neighbor : graph[node]) {
        dfs(neighbor, visited, graph);
    }
}

void bfs(int start, vector<vector<int>>& graph) {
    queue<int> q;
    unordered_set<int> visited;
    q.push(start);
    visited.insert(start);
    while (!q.empty()) {
        int node = q.front(); q.pop();
        for (int neighbor : graph[node]) {
            if (!visited.count(neighbor)) {
                visited.insert(neighbor);
                q.push(neighbor);
            }
        }
    }
}
```

---

## 🔁 Backtracking

**When to use:** Finding all solutions, generating permutations/combinations, solving constraint satisfaction problems.

**Time Complexity:** Exponential - depends on problem (O(2^n), O(n!), etc.)  
**Space Complexity:** O(depth of recursion)

### Python

```python
def backtrack(path, options, result):
    if len(path) == target_length:  # Base case
        result.append(path[:])  # Make a copy
        return
    
    for i in range(len(options)):
        # Choose
        path.append(options[i])
        # Recurse with remaining options
        backtrack(path, options[:i] + options[i+1:], result)
        # Undo (backtrack)
        path.pop()

def generate_permutations(nums):
    result = []
    backtrack([], nums, result)
    return result

def generate_subsets(nums):
    result = []
    def backtrack_subsets(start, path):
        result.append(path[:])  # Add current subset
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack_subsets(i + 1, path)
            path.pop()
    
    backtrack_subsets(0, [])
    return result
```

### C++

```cpp
void backtrack(vector<int>& path, vector<int>& options, vector<vector<int>>& result) {
    if (END_CONDITION) {
        result.push_back(path);
        return;
    }
    for (int i = 0; i < options.size(); ++i) {
        path.push_back(options[i]);
        vector<int> next = options;
        next.erase(next.begin() + i);
        backtrack(path, next, result);
        path.pop_back();
    }
}
```

---

## 🔍 Binary Search Variants

**When to use:** Searching in sorted arrays, finding insertion points, searching in rotated arrays, optimization problems.

**Time Complexity:** O(log n)  
**Space Complexity:** O(1)

### Python: Multiple Binary Search Patterns

```python
def binary_search(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def search_insert(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return left

def find_first_occurrence(nums, target):
    left, right = 0, len(nums) - 1
    result = -1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            result = mid
            right = mid - 1  # Continue searching left
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return result
```

### C++: Search Insert Position

```cpp
int searchInsert(vector<int>& nums, int target) {
    int left = 0, right = nums.size() - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] < target)
            left = mid + 1;
        else
            right = mid - 1;
    }
    return left;
}
```

---

## 🎓 DP - Top Down (Memoization)

**When to use:** Optimization problems with overlapping subproblems, recursive solutions that can be cached.

**Time Complexity:** O(number of unique subproblems × time per subproblem)  
**Space Complexity:** O(number of unique subproblems)

### Python

```python
def dp(i, memo):
    if i in memo:
        return memo[i]
    if i <= 1:  # Base case
        return i
    memo[i] = dp(i-1, memo) + dp(i-2, memo)
    return memo[i]

def fibonacci(n):
    return dp(n, {})

# Using functools.lru_cache decorator (cleaner syntax)
from functools import lru_cache

@lru_cache(maxsize=None)
def fib_clean(n):
    if n <= 1:
        return n
    return fib_clean(n-1) + fib_clean(n-2)

def coin_change(coins, amount):
    @lru_cache(maxsize=None)
    def dp(remaining):
        if remaining == 0:
            return 0
        if remaining < 0:
            return float('inf')
        
        min_coins = float('inf')
        for coin in coins:
            min_coins = min(min_coins, 1 + dp(remaining - coin))
        return min_coins
    
    result = dp(amount)
    return result if result != float('inf') else -1
```

### C++

```cpp
int dp(int i, unordered_map<int, int>& memo) {
    if (memo.count(i)) return memo[i];
    if (BASE_CASE) return VALUE;
    memo[i] = dp(i-1, memo) + dp(i-2, memo);
    return memo[i];
}
```

---

## 📊 Heap Usage

**When to use:** Finding kth largest/smallest elements, priority queues, merging sorted lists, scheduling problems.

**Time Complexity:** O(log n) for push/pop, O(1) for peek  
**Space Complexity:** O(n)

### Python: Min Heap & Max Heap

```python
import heapq

# Min Heap (default in Python)
minheap = []
heapq.heappush(minheap, value)
min_val = heapq.heappop(minheap)
min_val = minheap[0]  # Peek without removing

# Max Heap (negate values)
maxheap = []
heapq.heappush(maxheap, -value)
max_val = -heapq.heappop(maxheap)

# Heapify existing list
nums = [3, 1, 4, 1, 5]
heapq.heapify(nums)

# Find k largest elements
def find_k_largest(nums, k):
    return heapq.nlargest(k, nums)

# Find k smallest elements  
def find_k_smallest(nums, k):
    return heapq.nsmallest(k, nums)
```

### C++: Min Heap

```cpp
#include <queue>

priority_queue<int, vector<int>, greater<int>> minheap;
minheap.push(value);
int min_val = minheap.top(); minheap.pop();
```
