1. Deciper the typename before function argument

   Checkthis [blog](https://c-faq.com/decl/spiral.anderson.html) and this [stackoverflow answer](https://stackoverflow.com/questions/1143262/what-is-the-difference-between-const-int-const-int-const-and-int-const#:~:text=So%20in%20your%20question%2C%20%22int,to%20the%20thing%20after%20it.).

```cpp
//READ BACKWARD! THAT IS THE KEY.
float const * -> a pointer to constant float
const float * -> same as the first one
float * const -> a constant pointer to float (the pointer cannot point to other variable)
const * float -> a float pointer to constant...does not make sense, illegal.
```

2. Access memember variable in a base class template

```cpp
template <T>
class Base {
    // public var
    public:
        int a;
        int b;
        // constructor to initiate
        Base(int a, int b): a(a), b(b){};
}

tempelate <T>
class Foo : public Base <T>{
    // public var
    public:
        int c;
        int d;
        int *a = new (int);
        // constructor to initiate
        Base(int c, int d): Base<T>(c, d), c(c), d(d) {

            cout << a << endl; // error, a will not be found
            cout << this->a << endl; // ok
            cout << Base<T>::a << endl; // ok. :: is scope resolution operator
        };

        // destructor
        ~Base(){
            // do something
        }
}
```

3. Callimg function on member variable in class declaration is not allowed

```cpp
#include <vector>

template <T>
class Base {
    // public var
    public:
        int a;
        int b;
        vector<T> c(10, 0); // error, create a vector of 10 zeros
                            // the member of c is called on.
        vector<T> c = vector<T>(10, 0); // good, c is assigned, not called on
}

```

4. Friend function that can access private and protected members but not a member function

```cpp
class Base {
    // private var, only accessible within Base class
    private:
        int a;
    public:
        friend int add_one(Base);
}

// a friend function to increment the private member a by one
int add_one(Base& b){
    b.a += 1;
}

```

5. Lambda function and function binding

```cpp

#include <functional>
using namspace std;

// std::function is a class template

// function add_one takes a integer reference
// and returns null.

funcion<void (int &)> const add_one{
    [](int & a){
        a += 1;
    };
}

// [] can have either & or = inside, with the & as
// references to external variables and = as copies

// the lambda function is handy when a function is needed
// for once

int math(int &a){

    int b = 2;

    funcion<void (int &)> const add_one{
        [&](int & a){
            a += b;
        }
    };

    return 0;
}

// another application of lambda function

vector<int> v(10, 0);
// for each takes three argument,
// the first and second ones: iterators
// the last one a function to handle the value from the looping
// note that a reference is taken.
std:: for_each(v.begin(), v.end(), [](int &a){
    a += 1;
});

// function binding

```

6. Reference and pointer are similar concept except that reference is a constant pointer and does not need to dereference(`*`) to get the value. Reference is mainly used as function arguments to avoid copy a big chuck of data in function call.

```cpp

int a = 0;
int &b = a; // b is an alias and share the same address as a.

int c = 2;
int &b = c; // error, reference canno be changed to point to other variable

int& add_one(int& a){

    int b = 0;

    b = a;
    b ++;

    return b; // meaningless and could lead to segmentation error
              //  after add_one is executed, the local variable b will
              // be deleted. Except making b a static variable, this code
              // won't work.
}
```
