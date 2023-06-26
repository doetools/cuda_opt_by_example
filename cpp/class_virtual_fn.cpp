#include <iostream>

using namespace std;

class Animal
{

public:
    virtual void bark()
    {
        cout << "i am an animal" << endl;
    }

    virtual void get_legs() const = 0;
};

class Dog : public Animal
{
public:
    // override is used here for clarity
    // it is not necessary
    void bark() override
    {
        cout << "i am a dog" << endl;
    }

    virtual void get_legs() const override
    {
        cout << "i have 4 legs" << endl;
    }
};

int main()
{
    // function returning abstract class "Animal" is not
    //  allowed:C/C++(323)

    // Animal animal();
    // a carvet Dog dog(); will throws as an error
    // as the compiler think it is a declaration of dog function.
    Dog dog;
    dog.bark();
    dog.get_legs();

    // we can check if base class is overwritten
    // here we are casting a Dog pointer to an Animal pointer.
    // it is different from Animal animal = dog;
    Animal *animal = &dog;
    animal->bark();
}