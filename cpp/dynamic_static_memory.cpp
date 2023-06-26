#include <iostream>
#include <vector>
using namespace std;

int static_stack_memory()
{
    vector<int> data(10, 10);

    cout << "stack: before clear, size in byte: " << sizeof(vector<int>) + (sizeof(int) * data.size()) << endl;
    cout << "stack: before clear, capacity: " << data.capacity() << endl;
    cout << "stack: before clear, size: " << data.size() << endl;

    data.clear();

    // Printing the vector
    for (auto it = data.begin(); it != data.end();
         ++it)
        cout << ' ' << *it << "\n";

    cout << "stack: after clear, size in byte: " << sizeof(vector<int>) + (sizeof(int) * data.size()) << endl;
    cout << "stack: after clear, capacity: " << data.capacity() << endl;
    cout << "stack: after clear, size: " << data.size() << endl;

    // this is UB -- undefined behavor, with which anything can happen.
    cout << data[0] << endl;
    // this wil lead to an out of bound error
    // with a try throw and catch machenism
    try
    {
        cout << data.at(0) << endl;
        // throw std::out_of_range("balalala");
    }
    catch (std::out_of_range &e)
    {
        cout << e.what() << endl;
    };

    // make container zero capacity
    vector<int>().swap(data);

    return 0;
}

class Foo
{
public:
    int n = 0;
    int *T;

public:
    Foo() {}
    Foo(int n) : n(n)
    {
        T = new int[n];
    }

    ~Foo()
    {
        cout << "Foo: deallocating: " << n << endl;
        delete T;
    }
};

int dynamic_heap_memory()
{

    vector<Foo> vec_data(10);

    // vector<Foo> data;
    for (int i = 1; i < 10; i++)
        vec_data[i] = Foo(i);

    return 0;
}

int main()
{
    static_stack_memory();
    dynamic_heap_memory();
}
