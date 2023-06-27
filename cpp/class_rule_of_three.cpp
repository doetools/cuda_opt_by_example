/*******************************************************************************
this example is to demonstrate the rule of three or five, which says
If you need to explicitly declare either the destructor, copy constructor
or copy assignment operator yourself, you probably need to explicitly declare
all three of them.

The idea is that copying a object means copying its members. When there is pointer
or dynamically allocated memory, the pointer, not, its data will be copied. This
will cause a lot of problems in the later operator, like double freeing memory.

class Foo
{
...

public:
    Foo();                                      // parameterized constructor
    Foo(const Foo &) = default;                // 1/5: Copy Ctor
    Foo(Foo &&) noexcept = default;            // 4/5: Move Ctor
    Foo& operator=(const Foo &) = default;     // 2/5: Copy Assignment
    Foo& operator=(Foo &&) noexcept = default; // 5/5: Move Assignment
    ~Foo() noexcept = default;                 // 3/5: Dtor
};

For details check out
https://stackoverflow.com/questions/4172722/what-is-the-rule-of-three

********************************************************************************/

#include <iostream>
#include <vector>
using namespace std;

class Foo
{
public:
    int n = 0;
    int *T;

public:
    // below is function overloading, basically a fancy term
    // to say a multile function of same name will created to
    // handle different input argument types

    Foo() {}

    Foo(int n) : n(n)
    {
        T = new int[n];
    }

    // declare copy constructor operator
    Foo(const Foo &that);

    // overaloding assignment operator
    // check out class_overload_operator.cpp
    Foo &operator=(const Foo &that)
    {
        this->n = that.n;
        this->T = new int(*that.T);
        cout << "copy assign " << that.T << endl;
        return *this;
    }

    ~Foo()
    {
        cout << "Foo: deallocating: " << T << endl;
        delete T;
    }
};

Foo::Foo(const Foo &that)
{
    this->n = that.n;
    cout << "copy constructor " << that.T << endl;
    this->T = new int(*that.T);
}

int main()
{
    // instantiate a class
    Foo foo(5), foo1(5);

    // copy contructor (implicit)
    // see https://stackoverflow.com/questions/28357825/what-goes-on-in-the-background-when-we-assign-object1-object2-of-the-same-clas
    // for detailed explanation
    Foo foo_copy = foo;

    // copy constructor (explicit)
    Foo foo_copy_constructor(foo);

    // copy assiginment operator
    foo1 = foo;

    // all three memory will be the same
    // meaning when destructing a doube free will happen
    // and a core dump might happen
    cout << "memory: " << foo.T << endl;
    cout << "memory foo copy: " << foo_copy.T << endl;
    cout << "memory foo copy constructor: " << foo_copy_constructor.T << endl;

    cout << "obj memory: " << &foo << endl;
    cout << "obj memory foo copy: " << &foo_copy << endl;
    cout << "obj memory foo copy constructor: " << &foo_copy_constructor << endl;

    try
    {
        Foo foo_copy_constructor1(foo);
        Foo foo_copy_constructor2(foo);
        Foo foo_copy_constructor3(foo);
        Foo foo_copy_constructor4(foo);
        Foo foo_copy_constructor5(foo);
        Foo foo_copy_constructor6(foo);
    }
    catch (std::exception &e)
    {
        cout << e.what() << endl;
    };
}