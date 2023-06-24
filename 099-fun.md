## How to deciper the typename before function argument

READ BACKWARD!
Checkthis [blog](https://c-faq.com/decl/spiral.anderson.html) and this [stackoverflow answer](https://stackoverflow.com/questions/1143262/what-is-the-difference-between-const-int-const-int-const-and-int-const#:~:text=So%20in%20your%20question%2C%20%22int,to%20the%20thing%20after%20it.).

```
float const * -> a pointer to constant float
const float * -> same as the first one
float * const -> a constant pointer to float (the pointer cannot point to other variable)
const * float -> a float pointer to constant...does not make sense, illegal.
```
