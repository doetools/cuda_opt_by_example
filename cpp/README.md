1. Deciper the typename before function argument

   Checkthis [blog](https://c-faq.com/decl/spiral.anderson.html) and this [stackoverflow answer](https://stackoverflow.com/questions/1143262/what-is-the-difference-between-const-int-const-int-const-and-int-const#:~:text=So%20in%20your%20question%2C%20%22int,to%20the%20thing%20after%20it.).

```
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

7. A member function prefixed with `static` can be called without instantiating the class. The prefix `static` indicates that class is only a scope resolution. Broadly, a function prefixed with `static` or placed with an unamed namepsace is only by functions in THE file.

```cpp
using namespace std;
class Foo{
    ...

    static void print(int a){
        cout << a << endl;
    }
}

// fn_1 and fn_2 are only accessible in the file
// or the translation unit as opposed to every translation unit
static void fn_1();
namespace {
    void fn_2();
}

int main(){
    int a = 1;
    Foo::print(a);
}

```

8. Virtual function and pure virtual function can be defined in base class and overwritten (by in-time binding ) in derived class. Unlike the pure virtual function which only needs to be **declared**, the virtual function has to be **declared** and **defined**.

Check out the [example]("./virtual_fn.cpp");

```cpp

class Animal {
    public:
        // bark is a virtual function
        // virtual bark(); will not work
        // as it does not provide the definition.
        virtual void bark(){
            cout << "i am an animal" <<endl;
        }

        // get_legs is a pure virtual function
        // which makes Animal an abstract class
        // the derived class will continue to be
        // an abstract class unless it completes
        // the definition/implementation of get_legs.
        virtual void get_legs() const=0;
}

class Dog: public Animal{
    ...

    // overwrite the non-virutal function won't work

    // optional we can overwrite virutal function
    // this is called run-time binding
    void bark()
    {
        cout << "i am a dog" << endl;
    }

    // use override to overwrite the pure virtual function
    virtual void get_legs() const override
    {
        cout << "i have 4 legs" << endl;
    }
}

```

9. Pass a function as input argument for another function. This normally happens when a uniform interface is needed to perform some tests, for example, measuring the execution speed of a few functions.

The first way is like `C`, which is to pass the function by its pointer (`f` or `&f`). The second way is to pass the function as an object using `std::function`.

```cpp
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
}
```

10. Function binding.

```cpp

```

11. Static memory and dynamic memory. Any memory allocations done by `allocate` or `new` are dynamic and thus saved in HEAP (v.s. STACK). When not needed, all dynamically allocated memories have to be released by using `free` or `delete`. A carvet is that for array-like object, `delete []` needed to be used in lieu of `delete`.

An important thing to remember is that memory allocation could be hierarchical, as illustrated below. Too much memory management could be a headache, especially with a vector of pointers. Then, people started to using some kinds of enhanced pointers, like, `boost::shared_array`.

```cpp
// a vector container that store 10 int pointer
vector<int *> data(10, NULL);
// populate, so that vector stores 10 10 elememnt array
for_each(data.begin(),data.end(),[](int* i){
    i = new array[10];
})
// a naive of vector.clear wont do the job
// as this will only destroy the pointer
data.clear()

// instead, will have to destroy array by array
for_each(data.begin(),data.end(),[](int* i){
    delete [] i;
})

// you can then empty the vector using the swap technique
vector<int *>().swap(data)

```

So, do not make an object too deep or pointer-based... that will not work well from the memory management standpoint of view.

```cpp


```

12. Relating to **11**, use smart pointers, like `unique_ptr`, `shared_ptr`.

```cpp
#include <vector>
#include <iostream>
#include <memory>

struct T
{
    T(int n) :x(n) {};
    int print() { return x; };
private:
    int x;
};

int main(int argv, char** argc)
{
    std::vector<std::unique_ptr<T>> t;
    t.push_back(std::make_unique<T>(1));
    t.push_back(std::make_unique<T>(2));
    std::cout << t.size() << std::endl;
    std::cout << t.back()->print() << std::endl;
    t.pop_back();
    std::cout << t.size() << std::endl;
    std::cout << t.back()->print() << std::endl;
    return 0;
}

```

13. To supercede the point 12, try not using raw pointer. Otherwise, you are at the mercy of [RULE OF THREE](https://stackoverflow.com/questions/4172722/what-is-the-rule-of-three)
14. Access a local variable by its pointer or reference outside the (fn) scope will cause undefined bevavior. If a local vairable needs to be return, then return its value.

```cpp
// would fail
int foo(){
    int a = 0;

    return [&a](){
        cout << a << end;
    }
}

// would succeed
int foo_good(){
    int a = 0;

    return [=a](){
        cout << a << end;
    }
}

// accessing a free memory is undefined behavior
auto f = foo();
f();

```

15. Overlopping, not only for functions, but also for operators (within class). The class operator overloading can be global scope and class (member) scope. Class scope operator overloadig is only good for operations started with LHS of the operator being the class type; otherwise, a global operator overloading is needed. Operator `=` is inherently class scope.

```cpp

// simple function overloading

int print (int a){
    cout << "integet: "<<  a << endl;
}

int print (float a){
    cout << "float: "<< a << endl;
}

print (5); // calls the first print
print (5.0); // class the second print


class Foo{
    public:
        int a = 5;

    Foo(){}


    // member operator overloading
    // good for Foo + x
    // not food for x + Foo
    Foo operator+(int &x){
        return Foo(x+this->a);
    }
}
// global operator overloading
// good for x + food
Foo operator+(int &x, Foo &that){
    return Foo(x+that.a);
}

```

16. ellipsis in function/class template parameter. When put to the left of the parameter, it indicates a pack. When put to the right of the parameter, it indicates an expansion. This [link](https://learn.microsoft.com/en-us/cpp/cpp/ellipses-and-variadic-templates?view=msvc-170) has a good introduction.

```cpp


```

16. Weird terminologies and their counterpart (less weird).
    `lvalue` -- locator value, named variable with identifiable address
    `rvalue`-- opposite of `lvalue`, unnamed objects, like constant 5
    `&` -- reference of `lvalue`
    `&&` -- reference of `rvalue`
    `*` -- pointer of `lvalue` that can be obtained using operator `&`

```cpp
int a; // lvalue
a = 5; // 5 is rvalue (unnamed)
        // illegal &(5)

// expect the reference to a rvalue
void print (int && x){
    cout << x<< endl;
}

// good
print(5);

// not good
print (a);

```
