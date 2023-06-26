#include <iostream>

using namespace std;

class Foo
{
private:
    int a = 1;

public:
    // a friend (non-member) function
    friend void print_a(Foo);

    // a declaration of member function
    void print_private();

    // a static function that can be called w/o instantiation
    // static here implies using the class as a namespace
    static void print(int);
};

// non member that can access the private member
void print_a(Foo foo)
{
    cout << "friend fn: " << foo.a << endl;
}

void Foo::print_private()
{
    cout << "private print: " << a + 1 << endl;
}

// define a member function declared inside a class
void Foo::print(int a)
{
    cout << "member function: " << a << endl;
}

int main()
{
    Foo foo;
    // since it is a non-member, do not need scope resolution operator
    print_a(foo);
    // call member function
    foo.print_private();
    // call static function
    Foo::print(6);
}