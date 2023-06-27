#include <iostream>
#include <vector>
using namespace std;

class Foo
{
public:
    int a = 5;
    int b = 6;

    Foo(int a, int b) : a(a), b(b)
    {
    }

    // global operator
    friend Foo operator+(const Foo &x, const Foo &y);

    // this operator applies to x+ Foo (LHS)
    // string x = "foo" + string...
    friend Foo operator+(int &x, const Foo &y);

    // this operator applies only to Foo + x (RHS)
    // nor the other way around x + Foo (LHS)

    Foo operator+(int &x)
    {
        return Foo(this->a + x, this->b + x);
    }
};

// global operator, and it does not have to be a friend
// a friend fn can access private members inside an object
Foo operator+(int &x, const Foo &y)
{
    return Foo(x + y.a, x + y.b);
}

// below will not work as operator = is not a global one
// Foo operator=(const Foo &y){};

int main()
{
    Foo foo1(5, 6), foo2(10, 12);
    int x = 5;

    // call member operator +
    Foo res = foo1 + x;
    cout << "member operator+: " << res.a << " " << res.b << endl;

    // this wont work unless the friend operator is defined.
    Foo res1 = x + foo1;
    cout << "global operator+: " << res1.a << " " << res1.b << endl;
}