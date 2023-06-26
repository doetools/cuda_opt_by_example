#include <iostream>
#include <functional>
using namespace std;

int add(int &a, int &b)
{
    return a + b;
}

int multiply(int &a, int &b)
{
    return a * b;
}

typedef int (*FUNC_TYPE)(int &, int &);
// this is equivalent to int (*f)(int &, int &)
int pass_fn_pointer(FUNC_TYPE f, int &a, int &b)
{
    cout << f(a, b) << endl;
    return 0;
}

// use std::function
int pass_fn_object(function<int(int &, int &)> f, int &a, int &b)
{
    cout << f(a, b) << endl;
    return 0;
}

int main()
{
    int a = 1;
    int b = 2;

    // parsing &f or f is the same thing
    pass_fn_pointer(&multiply, a, b);
    pass_fn_object(&add, a, b);
    // printf("address of function multiply() is :%p\n", multiply);
    // printf("address of function &multiply() is : %p\n", &multiply);
}
