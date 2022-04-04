  /* Structures permettant de stocker les matrices et les informations sur les matrices, en COO et CSR, et fonctions permettant de faire des opérations sur les matrices */
  /* à compléter */
  /*Nicolas HOCHART*/

  #define matrixstruct

  #include <stdio.h>
  #include <stdlib.h>
  #include <mpi.h>
  #include <assert.h>

  #define NULL ((void *)0)

  /*-------------------------------------------------------------------
  --- Structures pour le stockage des matrices au format COO et CSR ---
  -------------------------------------------------------------------*/

  struct IntCOOMatrix
  {
       int * Row; //vecteur de taille len_values = "nombre d'éléments non nuls dans la matrice"
       int * Column; //vecteur de taille len_values = "nombre d'éléments non nuls dans la matrice"
       int * Value; //vecteur de taille len_values = "nombre d'éléments non nuls dans la matrice"
       long dim_l; //nombre de lignes
       long dim_c; //nombre de colonnes
       long len_values; //taille des vecteurs Row, Column et Value
  };
  typedef struct IntCOOMatrix IntCOOMatrix;

  struct IntCSRMatrix
  {
       int * Row; //vecteur de taille "nombre de lignes + 1" (dim_l + 1)
       int * Column; //vecteur de taille len_values = "nombre d'éléments non nuls dans la matrice"
       int * Value; //vecteur de taille len_values = "nombre d'éléments non nuls dans la matrice"
       long dim_l; //nombre de lignes
       long dim_c; //nombre de colonnes
       long len_values;  //taille des vecteurs Column et Value
  };
  typedef struct IntCSRMatrix IntCSRMatrix;

  struct DoubleCSRMatrix
  {
       int * Row; //vecteur de taille "nombre de lignes + 1" (dim_l + 1)
       int * Column; //vecteur de taille len_values = "nombre d'éléments non nuls dans la matrice"
       double * Value; //vecteur de taille len_values = "nombre d'éléments non nuls dans la matrice"
       long dim_l; //nombre de lignes
       long dim_c; //nombre de colonnes
       long len_values; //taille des vecteurs Column et Value
  };
  typedef struct DoubleCSRMatrix DoubleCSRMatrix;

  /*-----------------------------------------------------------
  --- Structures pour les blocks (sur les différents cores) ---
  -----------------------------------------------------------*/

  struct MatrixBlock
  {
       int indl; //Indice de ligne du block
       int indc; //Indice de colonne du block
       long dim_l; //nombre de lignes dans le block
       long dim_c; //nombre de colonnes dans le block
       long startRow; //Indice de départ en ligne (inclu)
       long startColumn; //Indice de départ en colonne (inclu)
       long endRow; //Indice de fin en ligne (inclu)
       long endColumn; //Indice de fin en colonne (inclu)

       int pr_result_redistribution_root; //Indice de colonne du block "root" (source) de la communication-redistribution du vecteur résultat
       int result_vector_calculation_group; //Indice de groupe de calcul du vecteur résultat
       int indc_in_result_vector_calculation_group; //Indice de colonne du block dans le groupe de calcul du vecteur résultat (groupes de blocks colonnes)
       int inter_result_vector_need_group_communicaton_group; //Indice du Groupe de communication inter-groupe de besoin (utile pour récupérer le résultat final)
       long startColumn_in_result_vector_calculation_group; //Indice de départ en colonne dans le groupe de calcul du vecteur résultat (inclu)
       long startRow_in_result_vector_calculation_group; //Indice de départ en ligne dans le groupe de calcul du vecteur résultat (inclu), utile dans le PageRank pour aller chercher des valeurs dans le vecteur q
       int my_result_vector_calculation_group_rank; //my_rank dans le groupe de calcul du vecteur résultat
  };
  typedef struct MatrixBlock MatrixBlock;

  /*---------------------------------
  --- Opérations sur les matrices ---
  ---------------------------------*/

  int get_csr_matrix_value_int(long indl, long indc, IntCSRMatrix * M_CSR)
  {
      /*
      Renvoie la valeur [indl,indc] de la matrice M_CSR stockée au format CSR.
      Le vecteur à l'adresse (*M_CSR).Value doit être un vecteur d'entiers.
      */
      int *Row,*Column,*Value;
      Row = (*M_CSR).Row; Column = (*M_CSR).Column; Value = (*M_CSR).Value;
      if (indl >= (*M_CSR).dim_l || indc >= (*M_CSR).dim_c)
      {
          perror("ATTENTION : des indices incohérents ont été fournis dans la fonction get_sparce_matrix_value()\n");
          return -1;
      }
      long i;
      long nb_values = Row[indl+1] - Row[indl]; //nombre de valeurs dans la ligne
      for (i=Row[indl];i<Row[indl]+nb_values;i++)
      {
          if (Column[i] == indc) {return Value[i];}
      }
      return 0; //<=> on a parcouru la ligne et on a pas trouvé de valeur dans la colonne
  }

  double get_csr_matrix_value_double(long indl, long indc, DoubleCSRMatrix * M_CSR)
  {
      /*
      Renvoie la valeur [indl,indc] de la matrice M_CSR stockée au format CSR.
      Le vecteur à l'adresse (*M_CSR).Value doit être un vecteur de doubles.
      */
      int *Row,*Column; double *Value;
      Row = (*M_CSR).Row; Column = (*M_CSR).Column; Value = (*M_CSR).Value;
      if (indl >= (*M_CSR).dim_l || indc >= (*M_CSR).dim_c)
      {
          perror("ATTENTION : des indices incohérents ont été fournis dans la fonction get_sparce_matrix_value()\n");
          return -1;
      }
      long i;
      long nb_values = Row[indl+1] - Row[indl]; //nombre de valeurs dans la ligne
      for (i=Row[indl];i<Row[indl]+nb_values;i++)
      {
          if (Column[i] == indc) {return Value[i];}
      }
      return 0; //<=> on a parcouru la ligne et on a pas trouvé de valeur dans la colonne
  }

  void coo_to_csr_matrix(IntCOOMatrix * M_COO, IntCSRMatrix * M_CSR)
  {
      /*
      Traduit le vecteur Row de la matrice M_COO stockée au format COO en vecteur Row format CSR dans la matrice M_CSR
      A la fin : COO_Column=CSR_Column (adresses), COO_Value=CSR_Value (adresses), et CSR_Row est la traduction en CSR de COO_Row (adresses et valeurs différentes)
      L'allocation mémoire pour CSR_Row (taille dim_l + 1) doit être faite au préalable
      Attention : dim_c, dim_l et len_values ne sont pas modifiés dans le processus, et doivent être remplis au préalable.
      */
      long i;
      for (i=0;i<(*M_COO).len_values;i++) //on parcours les vecteurs Column et Value de taille "nombre d'éléments non nuls de la matrice" = len_values
      {
          (*M_CSR).Column[i] = (*M_COO).Column[i];
          (*M_CSR).Value[i] = (*M_COO).Value[i];
      }

      int * COO_Row = (*M_COO).Row;
      int * CSR_Row = (*M_CSR).Row;
      long current_indl = 0;
      *(CSR_Row + current_indl) = 0;
      while(COO_Row[0] != current_indl) //cas particulier : première ligne de la matrice remplie de 0 (<=> indice de la première ligne, 0, différent du premier indice de ligne du vecteur Row)
      {
          *(CSR_Row + current_indl) = 0;
          current_indl++;
      }
      for (i=0;i<(*M_COO).len_values;i++)
      {
          if (COO_Row[i] != current_indl)
          {
              *(CSR_Row + current_indl + 1) = i;
              while (COO_Row[i] != current_indl + 1) //cas particulier : ligne de la matrice vide (<=> indice de Row qui passe d'un nombre i à un nombre j supérieur à i+1)
              {
                  current_indl++;
                  *(CSR_Row + current_indl + 1) = i;
              }
              current_indl = COO_Row[i];
          }
      }
      *(CSR_Row + current_indl + 1) = (*M_COO).len_values;
      if (current_indl < (*M_CSR).dim_l) //cas particulier : dernière(s) ligne(s) remplie(s) de 0
      {
          current_indl++;
          *(CSR_Row + current_indl + 1) = i;
      }
  }

  long get_nnz_row(IntCSRMatrix * M, long irow)
  {
      /*
      Renvoie le nombre de non-zero à la ligne "irow" de la matrice M d'entiers stockée en CSR
      On suppose que irow est compris entre 0 et (*M).dim_l inclu
      Renvoie un long qui correspond au nnz de la ligne.
      */
      return (long) (*M).Row[irow+1] - (*M).Row[irow];
  }

  long * get_nnz_rows_local(IntCSRMatrix * M)
  {
      /*
      Renvoie un pointeur vers un vecteur de long de taille (*M).dim_l (nombre de ligne dans la matrice M locale) contenant le nnz dans chaque ligne.
      On suppose que la matrice M a ses membres remplis (notamment (*M).dim_l et (*M).Row qui sont utilisés)
      */
      long * nnz_rows = (long *)malloc((*M).dim_l * sizeof(long));

      long i;
      for (i=0; i<(*M).dim_l; i++)
      {
          nnz_rows[i] = get_nnz_row(M, i);
      }
      return nnz_rows;
  }

  long * get_nnz_rows_local_double(DoubleCSRMatrix * M_double)
  {
      /*
      Renvoie un pointeur vers un vecteur de long de taille (*M).dim_l (nombre de ligne dans la matrice M locale) contenant le nnz dans chaque ligne.
      On suppose que la matrice M a ses membres remplis (notamment (*M).dim_l et (*M).Row qui sont utilisés)
      */
      IntCSRMatrix M_int;
      M_int.dim_l = (*M_double).dim_l;
      M_int.len_values = (*M_double).len_values;
      M_int.Row = (*M_double).Row;
      return get_nnz_rows_local(&M_int);
  }

  long * get_nnz_rows(IntCSRMatrix * M, MatrixBlock BlockInfo, long dim_l)
  {
      /*
      Renvoie un pointeur vers un vecteur de long de taille dim_l contenant le nnz de chaque ligne de la matrice globale (dimension dim_l)
      On suppose que la matrice M a ses membres remplis (notamment (*M).dim_l et (*M).Row qui sont utilisés)
      dim_l est le nombre de ligne de la matrice globale
      3 malloc sont fait dont 2 free
      1 malloc non free : le vecteur retourné, de taille dim_l
      à l'indice i du vecteur retourné se trouve le nombre d'éléments non nulles de la ligne i de la matrice globale
      */
      /* Communicateurs par ligne et colonne */
      MPI_Comm ROW_COMM,COLUMN_COMM;
      MPI_Comm_split(MPI_COMM_WORLD, BlockInfo.indl, BlockInfo.indc, &ROW_COMM);
      MPI_Comm_split(MPI_COMM_WORLD, BlockInfo.indc, BlockInfo.indl, &COLUMN_COMM);

      /*Poiteurs utilisés pour les vecteurs nnz_rows sur les processus d'une même ligne de la grille de processus*/
      long * nnz_rows_local = get_nnz_rows_local(M); //malloc de taille (*M).dim_l dans la fonction
      long * nnz_rows = (long *)malloc((*M).dim_l * sizeof(long));

      /*remplissage des nombres de non-zero dans nnz_rows (sur les processus d'une même ligne de la grille de processus)*/
      MPI_Allreduce(nnz_rows_local, nnz_rows, (*M).dim_l, MPI_LONG, MPI_SUM, ROW_COMM); //somme MPI_SUM de tout les nnz_rows_local dans nnz_rows, sur les lignes de la grille de processus.
      free(nnz_rows_local);

      long * nnz_rows_global = (long *)malloc(dim_l * sizeof(long));
      MPI_Allgather(nnz_rows, (*M).dim_l, MPI_LONG, nnz_rows_global, (*M).dim_l, MPI_LONG, COLUMN_COMM);
      free(nnz_rows);

      return nnz_rows_global;
  }

  long * get_nnz_columns_local(IntCSRMatrix * M)
  {
      /*
      Renvoie un pointeur vers un vecteur de long de taille (*M).dim_c (nombre de colonne dans la matrice M locale) contenant le nnz dans chaque colonne.
      On suppose que la matrice M a ses membres remplis (notamment (*M).dim_c, (*M).len_values et (*M).Column qui sont utilisés)
      */
      long * nnz_columns = (long *)malloc((*M).dim_c * sizeof(long));

      long i;
      for (i=0;i<(*M).dim_c;i++) //initialisation du vecteur sum_vector
      {
          nnz_columns[i] = 0;
      }

      for (i=0;i<(*M).len_values;i++) //on parcours le vecteur Column et Value, et on ajoute 1 à la somme des nnz (à l'indice correspondant)
      {
          nnz_columns[(*M).Column[i]] += 1;
      }

      return nnz_columns;
  }

  long * get_nnz_columns_local_double(DoubleCSRMatrix * M_double)
  {
      /*
      Renvoie un pointeur vers un vecteur de long de taille (*M).dim_c (nombre de colonne dans la matrice M locale) contenant le nnz dans chaque colonne.
      On suppose que la matrice M a ses membres remplis (notamment (*M).dim_c, (*M).len_values et (*M).Column qui sont utilisés)
      */
      IntCSRMatrix M_int;
      M_int.dim_c = (*M_double).dim_c;
      M_int.len_values = (*M_double).len_values;
      M_int.Column = (*M_double).Column;
      return get_nnz_columns_local(&M_int);
  }

  long * get_nnz_columns(IntCSRMatrix * M, MatrixBlock BlockInfo, long dim_c)
  {
      /*
      Renvoie un pointeur vers un vecteur de long de taille dim_c (passé en paramètre) contenant le nnz de chaque colonne de la matrice complète (tout processus réunis - dimension dim_c)
      On suppose que la matrice M a ses membres remplis (notamment (*M).dim_c, (*M).len_values et (*M).Column qui sont utilisés)
      dim_c est le nombre de ligne de la matrice globale
      3 malloc sont fait dont 2 free
      1 malloc non free : le vecteur retourné, de taille dim_c
      à l'indice i du vecteur retourné se trouve le nombre d'éléments non nulles de la colonne i de la matrice globale
      */
      /* Communicateurs par ligne et colonne */
      MPI_Comm ROW_COMM,COLUMN_COMM;
      MPI_Comm_split(MPI_COMM_WORLD, BlockInfo.indl, BlockInfo.indc, &ROW_COMM);
      MPI_Comm_split(MPI_COMM_WORLD, BlockInfo.indc, BlockInfo.indl, &COLUMN_COMM);

      /*Poiteurs utilisés pour les vecteurs nnz_columns sur les processus d'une même ligne de la grille de processus*/
      long * nnz_columns_local = get_nnz_columns_local(M); //malloc de taille (*M).dim_c dans la fonction
      long * nnz_columns = (long *)malloc((*M).dim_c * sizeof(long));

      /*remplissage des nombres de non-zero dans nnz_rows (sur les processus d'une même ligne de la grille de processus)*/
      MPI_Allreduce(nnz_columns_local, nnz_columns, (*M).dim_c, MPI_LONG, MPI_SUM, COLUMN_COMM); //somme MPI_SUM de tout les nnz_columns_local dans nnz_columns, sur les colonnes de la grille de processus
      free(nnz_columns_local);

      long * nnz_columns_global = (long *)malloc(dim_c * sizeof(long));
      MPI_Allgather(nnz_columns, (*M).dim_c, MPI_LONG, nnz_columns_global, (*M).dim_c, MPI_LONG, ROW_COMM);
      free(nnz_columns);

      return nnz_columns_global;
  }

  void normalize_csr_binary_matrix_on_columns(DoubleCSRMatrix * M, MatrixBlock BlockInfo)
  {
      /*
      Normalise la matrice CSR M sur les colonnes.
      La normalisation est faite au global : on divise par la somme des colonnes, après avoir reduce sur les colonnes pour avoir la somme globale (tout processus réunis)
      BlockInfo permet de définir le communicateur sur les colonnes, pour le reduce.
      */
      /* Communicateurs par colonne */
      MPI_Comm COLUMN_COMM;
      MPI_Comm_split(MPI_COMM_WORLD, BlockInfo.indc, BlockInfo.indl, &COLUMN_COMM);
      long i;

      /* Calcul de la somme sur les colonnes */
      long * nnz_columns_local = get_nnz_columns_local_double(M);
      long * nnz_columns = (long *)malloc((*M).dim_c * sizeof(long));
      MPI_Allreduce(nnz_columns_local, nnz_columns, (*M).dim_c, MPI_LONG, MPI_SUM, COLUMN_COMM); //somme MPI_SUM de tout les nnz_columns_local dans nnz_columns, sur les colonnes de la grille de processus
      free(nnz_columns_local);

      /* Division de chaque colonne par sa somme */
      for (i=0;i<(*M).len_values;i++) //on parcours le vecteur Column et Value, et on divise chaque valeur (de Value) par la somme (dans nnz_columns) de la colonne correspondante
      {
          (*M).Value[i] = (*M).Value[i] / nnz_columns[(*M).Column[i]];
      }
      free(nnz_columns);
  }

  void normalize_csr_binary_matrix_on_rows_global_sum_vector(DoubleCSRMatrix * M_CSR, MatrixBlock BlockInfo, long * row_sum_vector)
  {
      /*
      Normalise la matrice parallèle CSR M_CSR, stockée en parallèle, sur les lignes.
      Le vecteur row_sum_vector doit contenir le nombre d'éléments sur chaque ligne (vecteur de longueur matrix_dim_l).

      Les nombre de lignes dans la matrice (complète - tout processus réunis) doit être de matrix_dim_l.
      matrix_dim_l est inconnu dans la fonction, mais on va jusqu'à cet indice en utilisant BlockInfo.startRow (permet de prendre en compte le décalage en ligne, en parallèle)
      Même si un processus ne possède pas un morceau de chaque ligne, row_sum_vector concerne toutes les lignes (y compris celles des autres processus).
      On utilise donc le décalage en ligne BlockInfo.startRow pour prendre en compte la parallélisation.
      */
      long i,j;
      for(i=0; i<(*M_CSR).dim_l; i++) //parcours du vecteur row
      {
          for (j=(*M_CSR).Row[i]; j<(*M_CSR).Row[i+1]; j++) //parcours du vecteur column
          {
              (*M_CSR).Value[j] = 1.0/row_sum_vector[BlockInfo.startRow+i];
          }
      }
  }

  void normalize_csr_binary_matrix_on_rows(DoubleCSRMatrix * M, MatrixBlock BlockInfo)
  {
      /*
      Normalise la matrice parallèle CSR M_CSR, stockée en parallèle, sur les lignes.
      La normalisation est faite au global : on divise par la somme des colonnes totale, après avoir reduce sur les colonnes pour avoir la somme complète (tout processus réunis)
      BlockInfo permet de définir le communicateur sur les lignes, pour le reduce.
      */
      /* Communicateur par ligne */
      MPI_Comm ROW_COMM;
      MPI_Comm_split(MPI_COMM_WORLD, BlockInfo.indl, BlockInfo.indc, &ROW_COMM);
      long i,j;

      /* Calcul de la somme sur les lignes */
      long * nnz_rows_local = get_nnz_rows_local_double(M);
      long * nnz_rows = (long *)malloc((*M).dim_l * sizeof(long));
      MPI_Allreduce(nnz_rows_local, nnz_rows, (*M).dim_l, MPI_LONG, MPI_SUM, ROW_COMM); //somme MPI_SUM de tout les nnz_rows_local dans nnz_rows, sur les lignes de la grille de processus
      free(nnz_rows_local);

      /* Division de chaque ligne par sa somme */
      for(i=0; i<(*M).dim_l; i++) //parcours du vecteur row
      {
          for (j=(*M).Row[i]; j<(*M).Row[i+1]; j++) //parcours du vecteur column
          {
              (*M).Value[j] = 1.0/nnz_rows[i];
          }
      }
      free(nnz_rows);
  }

  void printf_csr_matrix_int_maxdim(IntCSRMatrix * M, int max_dim)
  {
      /*
      Imprime la matrice d'entiers stockée en format CSR passée en paramètre
      Si une des dimension de la matrice dépasse max_dim, le message "trop grande pour être affichée" est affiché
      */
      long i,j;

      if ((*M).dim_l <= max_dim && (*M).dim_c <=max_dim) //si l'une des dimensions de la matrice est supérieure à max_dim, on n'affiche pas la matrice
      {
          for (i=0;i<(*M).dim_l;i++)
          {
              for (j=0;j<(*M).dim_c;j++)
              {
                  printf("%i ", get_csr_matrix_value_int(i, j, M));
              }
              printf("\n");
          }
      }
      else
      {
          printf("Matrice trop grande pour être affichée..\n");
      }
  }

  double * local_csr_matrix_vector_product(DoubleCSRMatrix *M, double *x)
  {
      /*
      Effectue le produit matrice vecteur y = M.x, et retourne y. M doit être une matrice stockée au format CSR, ses champs doivent être correctement remplis, x doit être de taille (*M).dim_c = nombre de colonnes dans la matrice
      Le produit matrice_vecteur est effectué en local, et de manière non-optimale. Aucune parallélisation n'est prise en compte
      */
      long i,j;
      double * y = (double *)malloc((*M).dim_c * sizeof(double));

      for (i=0;i<(*M).dim_l;i++) //parcours des lignes
      {
          y[i] = 0;
          for (j=(*M).Row[i]; j<(*M).Row[i+1]; j++) //parcours des colonnes ; for (j=0;j<nb_col;j++)  y[i] += A[i*nb_col+j] * x[j]
          {
              y[i] += (*M).Value[j] * x[(*M).Column[j]];
          }
      }
      return y;
  }

  void printf_csr_matrix_int(IntCSRMatrix * M)
  {
      /*
      Imprime la matrice d'entiers stockée en format CSR passée en paramètre
      Si une des dimension de la matrice dépasse 32, le message "trop grande pour être affichée" est affiché
      */
      printf_csr_matrix_int_maxdim(M, 32);
  }

  MatrixBlock fill_matrix_block_info(int rank, int nb_blocks_row, int nb_blocks_column, long n)
  {
      /*
      Rempli une structure MatrixBlock en fonction des paramètres reçu, et la retourne
      Les informations remplies sont uniquement les informations basiques
      */
      struct MatrixBlock MBlock;
      MBlock.indl = rank / nb_blocks_column; //indice de ligne dans la grille 2D de processus
      MBlock.indc = rank % nb_blocks_column; //indice de colonne dans la grille 2D de processus
      MBlock.dim_l = n/nb_blocks_row; //nombre de lignes dans un block
      MBlock.dim_c = n/nb_blocks_column; //nombre de colonnes dans un block
      MBlock.startRow = MBlock.indl*MBlock.dim_l;
      MBlock.endRow = (MBlock.indl+1)*MBlock.dim_l-1;
      MBlock.startColumn = MBlock.indc*MBlock.dim_c;
      MBlock.endColumn = (MBlock.indc+1)*MBlock.dim_c-1;
      return MBlock;
  }

  MatrixBlock fill_matrix_block_info_adjacency_prv_pagerank(int rank, int nb_blocks_row, int nb_blocks_column, long n)
  {
      /*
      Rempli une structure MatrixBlock en fonction des paramètres reçu, et la retourne
      La structure est remplie pour un PageRank avec vecteur résultat réparti, appliqué à une matrice d'adjacence
      Les explications sur ce remplissage sont disponible dans le pdf "explication du PageRank version 6"
      */
      int pgcd_nbr_nbc, local_result_vector_size_row_blocks, local_result_vector_size_column_blocks;
      long local_result_vector_size_column;
      double grid_dim_factor;

      int tmp_r = nb_blocks_row, tmp_c = nb_blocks_column;
      while (tmp_c!=0) {pgcd_nbr_nbc = tmp_r % tmp_c; tmp_r = tmp_c; tmp_c = pgcd_nbr_nbc;}
      pgcd_nbr_nbc = tmp_r;

      struct MatrixBlock MBlock = fill_matrix_block_info(rank, nb_blocks_row, nb_blocks_column, n);

      grid_dim_factor = (double) nb_blocks_row / (double) nb_blocks_column;
      MBlock.pr_result_redistribution_root = (int) MBlock.indl / grid_dim_factor;
      local_result_vector_size_column_blocks = nb_blocks_column /pgcd_nbr_nbc;
      local_result_vector_size_row_blocks = nb_blocks_row / pgcd_nbr_nbc;
      local_result_vector_size_column = local_result_vector_size_column_blocks * MBlock.dim_c;
      MBlock.result_vector_calculation_group = MBlock.indc / local_result_vector_size_column_blocks;
      MBlock.indc_in_result_vector_calculation_group = MBlock.indc % local_result_vector_size_column_blocks;
      MBlock.inter_result_vector_need_group_communicaton_group = (MBlock.indl % local_result_vector_size_row_blocks) * nb_blocks_column + MBlock.indc;
      MBlock.startColumn_in_result_vector_calculation_group = MBlock.dim_c * MBlock.indc_in_result_vector_calculation_group;
      MBlock.startRow_in_result_vector_calculation_group = MBlock.dim_l * MBlock.indl % local_result_vector_size_row_blocks;
      MBlock.my_result_vector_calculation_group_rank = MBlock.indc_in_result_vector_calculation_group + MBlock.indl * local_result_vector_size_column_blocks;
      return MBlock;
  }
