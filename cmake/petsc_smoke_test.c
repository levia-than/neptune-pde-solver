#include <petscsys.h>

int main(int argc, char **argv) {
  PetscInitialize(&argc, &argv, NULL, NULL);
  PetscFinalize();
  return 0;
}
