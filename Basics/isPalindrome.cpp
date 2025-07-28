class Solution {
public:
    bool isPalindrome(int x) {
        if (x < 0){
            return false;
        }

        int temp = x, cur = 0;

        while(temp){
            if (cur > INT_MAX/10){ return false;}
            cur = cur*10 + temp%10;
            temp /= 10;
        }

        return cur==x;
    }
};
