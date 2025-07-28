class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> hashMap;
        int cur, i, n = nums.size();

        for(i = 0; i < n; i++){
            cur = nums[i];

            if (hashMap.count(target-cur)){
                return {hashMap[target-cur], i};
            }
            else{
                hashMap[cur] = i;
            }
        }
        return {};
    }
};
