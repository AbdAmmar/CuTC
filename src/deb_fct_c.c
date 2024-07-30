#include <stdio.h>

int deb_fct_c(int n, double *Tin, double *Tout) {

    int i;

    printf("Hello from C fct \n");

    printf("n = %d\n", n);

    printf("Tin:\n");
    for(i = 0; i < n; i++) {
        printf("Tin[%d] = %f\n", i, Tin[i]);
    } 

    for(i = 0; i < n; i++) {
        Tout[i] = Tin[i] + (double) i;
    } 

    return 0;
}


