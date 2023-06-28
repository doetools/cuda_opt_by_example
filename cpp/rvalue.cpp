#include <iostream>
using namespace std;

void print(int &&x)
{
    cout << x << endl;
}

int main()
{

    int a = 5;

    // not good
    // print(a);

    // good
    print(5);
}