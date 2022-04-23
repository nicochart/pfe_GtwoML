/*
Exemple basique du Linux Magazine
compilation :   "g++ -fopenmp helloworld.c -o" helloworld (gcc ne marche pas car "iostream" non trouvé)
                -fopenmp pour utiliser openmp
*/

#include <iostream> 

using namespace std;

int main(int argc, char** argv)
{
  #pragma omp parallel for
  for (int i = 0; i < argc; ++i)
  {
    #pragma omp critical
    cout << "L'argument " << i << " a pour valeur" << " '" << argv[i] << "'" << endl;
  }
  return 0;
}
