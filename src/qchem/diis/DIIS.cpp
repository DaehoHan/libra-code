#include "DIIS.h"

#include <Eigen/LU>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Core>

//#include <Eigen/SVD>
using namespace Eigen;


/**********************************************************************

  DIIS::DIIS(int _N_diis_max)

  void DIIS::update_diis_coefficients()
  void DIIS::add_diis_matrices(MATRIX* _X, MATRIX* _err)

  void DIIS::extrapolate_matrix(MATRIX* X_ext)

**********************************************************************/

DIIS::DIIS(int _N_diis_max,int Norb){

  int n;

  // Setup parmeters and variables
  N_diis = 0;
  N_diis_eff = 0;
  N_diis_max = _N_diis_max;

  // Clear memory if used
  if(diis_X.size()>0){ diis_X.clear(); }
  if(diis_err.size()>0){ diis_err.clear(); }


  // Allocate memory
  diis_c = vector<double>(N_diis_max,0.0);
  for(n=0;n<N_diis_max;n++){
    MATRIX* x; x = new MATRIX(Norb,Norb); *x = 0.0;
    diis_X.push_back(x);
  }
  for(n=0;n<N_diis_max;n++){
    MATRIX* x; x = new MATRIX(Norb,Norb); *x = 0.0;
    diis_err.push_back(x);
  }

}// DIIS::DIIS(int _N_diis_max)



void DIIS::update_diis_coefficients(){

    // General case - holds true for both N_diis<N_diis_max and for N_diis==N_diis_max (after preliminary corrections )
    // Starting at this point we compute extrapolation coefficients:
    // Solving Ax = b

  int N_diis_curr;
  int rank;
  int i,j;
  int debug_flag = 0;
  double diis_damp = 0.0; // see [Hamilton,Pulay, JCP 84, 5728 (1986) ] - scale diagonal element of B matrix by (1+diis_damp) to
                          // avoid numerical problems associated with large diis coefficents


  MatrixXd* A;  // will be of size N_diis+2
  VectorXd* b; //(N_diis+2);        
  VectorXd* x; //(N_diis+2);


  if(debug_flag){    cout<<"*diis_err[N_diis]= "<<*diis_err[N_diis]<<endl; }


  // Actual size of DIIS matrices at given call
  // We save this number, because N_diis can reduce in the following run, to give full-rank matrices
  N_diis_curr = N_diis;
 
  // Initial guess 
  A = new MatrixXd(N_diis+2,N_diis+2);

  for(i=0;i<=N_diis;i++){
    for(j=0;j<=N_diis;j++){            
      (*A)(i,j) = ((*diis_err[i]).T() * (*diis_err[j])).tr();

      if(i==j){ (*A)(i,j) *= (1.0+diis_damp); }
    }
    (*A)(i,N_diis+1) = -1.0;
    (*A)(N_diis+1,i) = -1.0;
  } 
  (*A)(N_diis+1,N_diis+1) = 0.0;


  // Determine the rank of the DIIS matrix A
  FullPivLU<MatrixXd> lu_decomp(*A);
  rank = lu_decomp.rank();

  if(rank==(N_diis+2)){  if(debug_flag){    cout<<"Matrix is well-conditioned\n"<<*A<<endl;  }  }
  else{     if(debug_flag){    cout<<"Matrix is ill-conditioned\n"<<*A<<endl;   }

    int min_indx = 0;

    // Reduce DIIS matrix by removing older iterates untill it is well-conditioned
    while(rank!=(N_diis+2)){

      N_diis--;
      min_indx++; // this will hide first min_indx diis_err matrices from consideration 

      MatrixXd tempA(N_diis+2,N_diis+2);  

      for(i=0;i<=N_diis;i++){
        for(j=0;j<=N_diis;j++){            
          tempA(i,j) = ((*diis_err[min_indx+i]).T() * (*diis_err[min_indx+j])).tr();

          if(i==j){ tempA(i,j) *= (1.0+diis_damp); }

        }
        tempA(i,N_diis+1) = -1.0;
        tempA(N_diis+1,i) = -1.0;
      } 
      tempA(N_diis+1,N_diis+1) = 0.0;

      FullPivLU<MatrixXd> tmp_lu_decomp(tempA);
      rank = tmp_lu_decomp.rank();

      if(debug_flag){
        cout<<"Reduction iteration "<<min_indx<<endl;
        cout<<"Reduced matrix = \n"<<tempA<<endl;
        cout<<"Reduced matrix rank = "<<rank<<endl;
      }

      if(rank==N_diis+2){ A = new MatrixXd(N_diis+2,N_diis+2);  *A = tempA; }

    }// while

  }// else rank!=Ndiis+2


  b = new VectorXd(N_diis+2);  for(i=0;i<=N_diis;i++){  (*b)(i) = 0.0;  }  (*b)(N_diis+1) = -1.0;
  x = new VectorXd(N_diis+2);

  // Solve linear algebra to get coefficients
  *x = A->lu().solve(*b); for(i=0;i<=N_diis;i++){  diis_c[i] = (*x)[i];  }

  if(debug_flag){
    cout<<"A*x = "<<(*A)*(*x)<<endl;
    cout<<"b = "<<*b<<endl;
    cout<<"x = "<<*x<<endl;
  }

  // Free memory
  delete A;
  delete b;
  delete x;


  // Compute effective size of the DIIS matrix
  N_diis_eff = (N_diis_curr - N_diis); // keep in mind that N_diis is a current one (updated)   
  N_diis = N_diis_curr;


}//void DIIS::update_diis_coefficients()




void DIIS::add_diis_matrices(MATRIX* X, MATRIX* err){

//  cout<<"Starting add_diis_matrices\n  N_diis = "<<N_diis<<endl;
  int i,j;
//  int N_diis_curr;

  if(N_diis==0){ // The very first iterate (most likely this will be used right after guess)

    *diis_X[0] = *X;        
    *diis_err[0] = *err;
    diis_c[0]     = 1.0;

//    N_diis_curr = N_diis;

  }// N_diis==0

  else if(N_diis>0){  // we already have a few matrices stored

    if(N_diis>=N_diis_max){ // this happens because last call of add_diis_matrices incremented N_diis

      N_diis = N_diis_max-1;  // so we set N_diis to maximal valid index, pointing to the last entries in the arrays


    // Rotate matrices
    // so only last entry X[N_diis_max-1] is old 
    // in the following sections we will update it
      for(i=0;i<N_diis;i++){
        
        *diis_X[i] = *diis_X[i+1];
        *diis_err[i] = *diis_err[i+1];
        diis_c[i]    = diis_c[i+1];

      }// for i


    }// N_diis==N_diis_max  (so it will actually always be N_diis_max)
    else{
      // do nothing special
    }


    // N_diis - always points to the last valid element of DIIS lists (arrays)
    // so the size of the matrices is (N_diis+1) x (N_diis+1)
    *diis_X[N_diis] = *X;    
    *diis_err[N_diis] = *err;

    // Now we need to update coefficients for given set of DIIS matrices
    update_diis_coefficients();
 
  }// N_diis>0

  N_diis++;

//  cout<<"Finish add_diis_matrices\n  N_diis = "<<N_diis<<endl;

}// void DIIS::add_diis_matrices(MATRIX* _X, MATRIX* _err)


void DIIS::extrapolate_matrix(MATRIX* X_ext){

  // Extrapolate X matrix
  // Note the Fock matrix constructed below (extrapolated) will only be used to obtain density
  // It will not be stored in diis_Fao_... (timing/sequence of function calls in scf() procedure is very important!!! )

  //!!!!!!!!!! Assume it is called just after add_diis_matrices !!!!!!!!!!!
  // so we need to used decremented N_diis value!!!!!

  *X_ext = 0.0;

  for(int i=0;i<=((N_diis-1)-N_diis_eff);i++){
    *X_ext += diis_c[i] * (*diis_X[N_diis_eff + i]);
  }

}// void DIIS::extrapolate_matrix(MATRIX* X_ext)


