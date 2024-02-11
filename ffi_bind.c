#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void* __ffi_bind(){
  
  return NULL;

  
}

extern void* test(unsigned int* len_ptr){
    void* mem = malloc((size_t) *len_ptr);

    memset(mem, 88, (size_t) *len_ptr);

    return mem;
}

extern char* gets_r(size_t len){
    char* a = (char*)malloc(len);
    scanf("%s", a);
    return a;
}