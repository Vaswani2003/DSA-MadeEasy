class Solution {
public:
    bool isValid(string s) {
        stack<char> paranthStack;
        unordered_map<char, char> paranth_maps = {
            {')', '('},
            {'}', '{'},
            {']', '['}
        };

        for(char ch: s){
            if (ch == '(' || ch == '{' || ch == '['){
                paranthStack.push(ch);
            }
            else {
                if (paranthStack.empty()){return false;}

                else if(paranthStack.top() != paranth_maps[ch]){ return false; }
                
                else{ paranthStack.pop(); }
            }
        }
        return paranthStack.empty();
    }
};
