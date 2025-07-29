class Solution {
public:
    vector<string> fizzBuzz(int n) {
        string FIZZ = "Fizz", BUZZ = "Buzz", FIZZBUZZ = "FizzBuzz";
        vector<string> answer;
        bool fizz, buzz;


        for(int i = 1; i <= n; i++){

            fizz = (i%3==0);
            buzz = (i%5==0);
            
            if (fizz and buzz){
                answer.push_back(FIZZBUZZ);
            }
            else if (fizz){
                answer.push_back(FIZZ);
            }
            else if(buzz){
                answer.push_back(BUZZ);
            }
            else{
                answer.push_back(to_string(i));
            }
        }

        return answer;
    }
};
