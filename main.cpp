//
// Created by Vitto Resnick on 2/21/24.
//
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <fstream>
#include <armadillo>

using namespace std;
using namespace arma;

// Count words in input file to know which question algorithm to run
tuple<int,int> count_words(string file_name){
    ifstream inFile;
    inFile.open(file_name);

    string line;
    int numWords = 0;
    int numLines = 0;

    while(getline(inFile, line)){
        stringstream lineStream(line);
        while(getline(lineStream, line, ' ')){
            numWords++;
        }
        numLines++;
    }

    inFile.close();
    return make_tuple(numWords,numLines);
}

// Check if a number is an integer
bool is_integer(double k){
    return floor(k) == k;
}


// Gaussian G(X) centered at X
double G(double x, double X, double alpha, double l){
    return pow(x-X,l)*exp(-alpha*pow(x-X,2));
}

// Factorial function
int factorial(int n){
    int res = 1,i;
    for (i=1;i<=n;i++){
        res *= i;
    }
    return res;
}

// Double factorial
int dfact(int n){
    int i;double res=1.0;
    for(i=n;i>=1;i-=2){
        res *=i;
    }
    return res;
}

// Bionomial Coefficient Example: m choose n
int binomCoeff(int m, int n){
    return factorial(m)/(factorial(n)*factorial(m-n));
}

// Calculation one of three directional components for SAB
double SxABComp(double XA, double XB, double alpha, double beta, double lA, double lB){

    double P  = exp(-alpha*beta*pow(XA-XB,2)/(alpha + beta)); // Calculate prefactor
    double XP = (alpha*XA + beta*XB)/(alpha + beta);
    double doubleSum = 0; // Initialize double sum

    // Compute  double sound
    for     (int i = 0; i < lA+1; i++){
        double innerSum = 0;
        for (int j = 0; j < lB+1; j++){
            if ((i+j)% 2 == 0){ // Only do even i+j terms
                double summand = binomCoeff(lA,i)*binomCoeff(lB,j)*dfact(i+j-1)*pow(XP-XA,lA-i)*pow(XP-XB,lB-j)/pow(2*(alpha+beta),(i+j)/2);
                innerSum += summand;
            }
        }
        doubleSum += innerSum;
    }
    return P*sqrt(M_PI/(alpha + beta))*doubleSum;
}

// Find normalization constant for a given k primitive gauss
double NormConst(double X, double Y, double Z, double alpha_k, double l, double m, double n){
    double SAA = 1; // Initialize SAB product
    SAA *= SxABComp(X,X,alpha_k,alpha_k,l,l); // Compute SxAA
    SAA *= SxABComp(Y,Y,alpha_k,alpha_k,m,m); // Compute SyAA
    SAA *= SxABComp(Z,Z,alpha_k,alpha_k,n,n); // Compute SzAA

    // normalization constants are defined such that the overlap of each primitive with itself is equal to one
    double N_k_lmn = 1/sqrt(SAA);
    return N_k_lmn;
}

// Question floating point or double? loss of info?
// Find constants for a given basis function
tuple<vec,vec,vec,vec,vec> FindConsts(int atom, vec R_center,string orbital){
    // Exponent Data
    const vec H_alpha_1s = vec({3.42525091,0.62391373,0.16885540});
    const vec C_alpha_2s_2p = vec({2.94124940,0.68348310,0.22228990});
    // Contraction Coefficient Data
    const vec H_d_1s = vec({0.15432897,0.53532814,0.44463454});
    const vec C_d_2s = vec({-0.09996723,0.39951283,0.70011547});
    const vec C_d_2p = vec({0.15591627,0.60768372,0.39195739});
    // Quantum Numbers
    const vec lms_s = vec({0,0,0});
    const vec lms_px = vec({1,0,0});
    const vec lms_py = vec({0,1,0});
    const vec lms_pz = vec({0,0,1});

    // Initialize Output Vectors
    vec exponents;
    vec contraCoeffs;
    vec quantNums;
    if (atom == 1){
        exponents    = H_alpha_1s;
        contraCoeffs = H_d_1s;
        quantNums    = lms_s;
    } else if (atom == 6){
        if        (orbital == "2s" ){
            exponents    = C_alpha_2s_2p ;
            contraCoeffs = C_d_2s;
            quantNums    = lms_s;
        } else if (orbital == "2px"){
            exponents    = C_alpha_2s_2p;
            contraCoeffs = C_d_2p;
            quantNums    = lms_px;
        } else if (orbital == "2py"){
            exponents    = C_alpha_2s_2p;
            contraCoeffs = C_d_2p;
            quantNums    = lms_py;
        } else if (orbital == "2pz"){
            exponents    = C_alpha_2s_2p;
            contraCoeffs = C_d_2p;
            quantNums    = lms_pz;
        }
    }
    // For each basis function, ωμ(r) (for μ = 1 · · · N ) there will be ...

    // (i) a center, R,
    double X = R_center(0);
    double Y = R_center(1);
    double Z = R_center(2);

    // (ii) 3 quantum numbers, (l , m, n),
    double l = quantNums(0);
    double m = quantNums(1);
    double n = quantNums(2);

    // and (iii) information about 3 primitive functions:
    // 3 exponents, αk,
    // 3 corresponding contraction coefficients, dk, and
    // 3 normalization constants, N_k_lmn
    int K = contraCoeffs.size();
    vec normConsts(K, fill::ones);
    for (int k = 0; k < K; k++){
        double alpha_k = exponents(k);
        double N_k_lmn = NormConst(X,Y,Z,alpha_k,l,m,n);
        normConsts(k) = N_k_lmn;
    }
    return make_tuple(R_center,quantNums,exponents,contraCoeffs,normConsts);
}

// Not used, functional omega_mu calculation
double omega_mu(vec r_input,tuple<vec,vec,vec,vec,vec> constants){
    auto [R_center,quantNums,exponents,contraCoeffs,normConsts] = constants;

    double x = r_input(0);
    double y = r_input(1);
    double z = r_input(2);

    double X = R_center(0);
    double Y = R_center(1);
    double Z = R_center(2);

    double l = quantNums(0);
    double m = quantNums(1);
    double n = quantNums(2);

    // contracted function
    double omega_mu_r = 0;

    int K = contraCoeffs.size();
    for (int k = 0; k < K; k++){
        double d_k_mu  = contraCoeffs(k);
        double alpha_k = exponents(k);
        double N_k_lmn = normConsts(k); // normalization constants

        double omega_k_r;
        double Gx = G(x,X,alpha_k,l);
        double Gy = G(y,Y,alpha_k,m);
        double Gz = G(z,Z,alpha_k,n);

        // The coefficients, dkμ, mix together the primitive gaussian functions given by ...
        omega_k_r = N_k_lmn * Gx * Gy * Gz;

        // contracted function as a linear combination of primitive gaussian functions
        omega_mu_r += d_k_mu * omega_k_r;
    }
    return omega_mu_r;
}

double diag_h_select(int atom, string orbital){
    const double h_H    = -13.6; // eV, H
    const double h_C_2s = -21.4; // eV, C 2s
    const double h_C_2p = -11.4; // eV, C 2p

    if (atom == 1){
        return h_H;
    } else if (atom == 6 && orbital == "2s"){
        return h_C_2s;
    } else if (atom == 6 && (orbital == "2px" || orbital == "2py" || orbital == "2pz")){
        return h_C_2p;
    }
}

int main() {
    string file_name = "/Users/vittor/Documents/CLASSES/SPRING 2024/CHEM_179_HW3/test_cases/og.txt";
    auto [numWords,numLines] = count_words(file_name);
    // Count number of words in file to determine which question to do.
    // Question 1

    // Read in the coordinates, in the format: E X Y Z for each atom, where E is the element (handle at least H and C).

    // Read in input file
    ifstream inputFile(file_name);

    // Throw error if file was not opened correctly
    if (!inputFile) {
        cerr << "Error opening file." << endl;
    }

    // Initialize vars
    int num_atoms;          // Initialize total number of atoms = n
    int charge;
    inputFile >> num_atoms >> charge; // Set total number of atoms = n
    const double Bohr_A = 0.52917706; // angstroms in 1 bohr
    int a = 0; // Number carbons
    int b = 0;
    vector<vector<double>> xyz_list;       // Initialize list for atoms' xyz coordinates
    vector<int> atom_list;              // Initialize list for atoms' identities
    vector<vector<double>> basis_xyz_list;       // Initialize list for atoms' xyz coordinates
    vector<int> basis_atom_list;              // Initialize list for atoms' identities

    // Read in atom identity and xyz coordinates
    for (int i = 0; i < num_atoms; ++i) {          // Iterate through every atom
        int atom;
        double x, y, z;                    // Initialize atom identity and xyz coordinates
        inputFile >> atom >> x >> y >> z ; // Set atomic number/atom identity and xyz coordinates
        x = x/Bohr_A;
        y = y/Bohr_A;
        z = z/Bohr_A;
        if (atom != 6 && atom != 1) {                  // If a given atom is not gold, throw an error
            cerr << "Atom No." << i+1 << ": This atom is not a carbon or hydrogen!" << endl;
        } else if (atom == 6){
            a = a + 1;
            basis_atom_list.insert(basis_atom_list.end(), {6,6,6,6});
            basis_xyz_list.insert(basis_xyz_list.end(), { {x, y, z},{x, y, z},{x, y, z},{x, y, z} });
        } else if (atom == 1){
            b = b + 1;
            basis_atom_list.push_back(atom);
            basis_xyz_list.push_back({x, y, z});
        }

        atom_list.push_back(atom);         // Append this atom's atomic number/atom identity to list
        xyz_list.push_back({x, y, z});     // Append this atom's xyz coordinates to list
    }
    inputFile.close();                     // Close the txt file

    // Evaluate the number of basis functions, N from the molecular formula, Ca Hb , where
    // the relation is N = 4a + b. Your matrices, such as S, H, X, etc, will be N × N , so this will
    // enable you to define them.
    int N = 4*a+b;

    // Evaluate the number of electrons 2n = 4a + b. Throw an error if the number of electron
    // pairs n = 2a +b/2 is not an integer. Knowing n is necessary to evaluate the energy later.
    int n;
    if (!is_integer(2*a+b/2)){
        cerr << "The number of electron pairs n = 2a +b/2 is not an integer!" << endl;
    } else {
        n = 2*a + b/2;
    }

    // Build a list of the basis functions, which are contracted gaussians,
    // Iterate through N AOs and construct basis functions
    vector<string> C_orbital_bank = {"2s","2px","2py","2pz"};
    vector<tuple<vec,vec,vec,vec,vec>> basis_func_constants;
    vector<string> basis_orbital_list;

    for (int i = 0; i < N; i++){

        int atom = basis_atom_list[i];
        vector<double> center = basis_xyz_list[i];

        string orbital;
        if (atom == 1){
            orbital = "1s";
        } else if (atom == 6){
            orbital = C_orbital_bank[0];
            C_orbital_bank.erase(C_orbital_bank.begin());
        }
        if (C_orbital_bank.size() == 0){
            C_orbital_bank = {"2s","2px","2py","2pz"};
        }
        basis_orbital_list.push_back(orbital);

        //auto[exponents,contraCoeffs,quantNums,R_center,normConsts] = FindConsts(atom, center, orbital);
        basis_func_constants.push_back(FindConsts(atom, center, orbital));
    }

    // Question 2

    mat All_Normalization_Constants(3,N,fill::zeros);

    // Run a loop over all your basis functions
    for (int i = 0; i < N; i++){
        // get the normalization constants for the 3 primitives that make up each basis function
        auto[R_center,quantNums,exponents,contraCoeffs,normConsts] = basis_func_constants[i];

        // and save them in an array.
        All_Normalization_Constants.col(i) = normConsts;
    }


    // Contracted overlap integral S
    mat S(N,N,fill::zeros);

    // Iterate over pairs of basis functions
    for (int mu = 0; mu < N; mu++){
        for (int nu = 0; nu < N; nu++){
            // For this given pair of basis functions
            // R_center,quantNums,exponents,contraCoeffs,normConsts
            auto[R_mu,lmn_mu,a_mu,d_mu,N_mu] = basis_func_constants[mu];
            auto[R_nu,lmn_nu,a_nu,d_nu,N_nu] = basis_func_constants[nu];

            int K = d_mu.size();
            int L = d_nu.size();

            double S_mu_nu = 0;

            for (int k = 0; k < K; k++){
                for (int l = 0; l < L; l++){
                    double d_k_mu     = d_mu(k);
                    double d_l_nu     = d_nu(l);
                    double N_k_mu     = N_mu(k);
                    double N_l_nu     = N_nu(l);
                    double alpha_k_mu = a_mu(k);
                    double alpha_l_nu = a_nu(l);

                    double Xm = R_mu(0);
                    double Ym = R_mu(1);
                    double Zm = R_mu(2);

                    double Xn = R_nu(0);
                    double Yn = R_nu(1);
                    double Zn = R_nu(2);

                    double lm = lmn_mu(0);
                    double mm = lmn_mu(1);
                    double nm = lmn_mu(2);

                    double ln = lmn_nu(0);
                    double mn = lmn_nu(1);
                    double nn = lmn_nu(2);

                    double S_k_l = 1; // Initialize SAB product
                    S_k_l *= SxABComp(Xm,Xn,alpha_k_mu,alpha_l_nu,lm,ln); // Compute Sxkl
                    S_k_l *= SxABComp(Ym,Yn,alpha_k_mu,alpha_l_nu,mm,mn); // Compute Sykl
                    S_k_l *= SxABComp(Zm,Zn,alpha_k_mu,alpha_l_nu,nm,nn); // Compute Szkl

                    S_mu_nu += d_k_mu * d_l_nu * N_k_mu * N_l_nu * S_k_l;
                }
            }
            S(mu,nu) = S_mu_nu;
        }
    }

    //S.print("Contracted overlap integral S");

    // Question 3

    mat H(N,N,fill::zeros);

    const double K = 1.75;
    // Iterate over pairs of basis functions
    for (int mu = 0; mu < N; mu++) {
        for (int nu = 0; nu < N; nu++) {
            if (mu == nu){ // The diagonal elements
                H(mu,nu) = diag_h_select(basis_atom_list[mu],basis_orbital_list[mu]);

            } else { //
                double h_mu_nu;
                double h_mu_mu = diag_h_select(basis_atom_list[mu],basis_orbital_list[mu]);
                double h_nu_nu = diag_h_select(basis_atom_list[nu],basis_orbital_list[nu]);
                h_mu_nu = 0.5*K*(h_mu_mu+h_nu_nu)*S(mu,nu);
                H(mu,nu) = h_mu_nu;
            }
        }
    }

    //H.print("Hamiltonian!");

    // Solve the generalized eigenvalue problem to obtain the molecular orbital coefficients, C and the eigenvalues ε.

    // Make the orthogonalization transformation

    S = mat("1 0.004 0 0 -0.06; 0.004 1 0 0 0; 0 0 1 0 0; 0 0 0 1 0; -0.06 0 0 0 1");
    H = mat("-13.6 -0.19 0 0 1.66; -0.19 -40 0 0 0; 0 0 -18 0 0; 0 0 0 -18 0; 1.66 0 0 0 -18");
    N = S.n_cols;


    vec eigval; // main diagonal of D matrix
    mat eigvec; // U matrix
    eig_sym(eigval, eigvec, S,"dc");

    mat U = eigvec;
    mat s = diagmat(eigval);

    mat s_inv_sq(N,N,fill::zeros);

    for (int mu = 0; mu < N; mu++){
        for (int nu = 0; nu < N; nu++){
            if (s(mu,nu) == 0){
                s_inv_sq(mu,nu) = 0;
            } else {
                s_inv_sq(mu,nu) = 1/sqrt(s(mu,nu));
            }
        }
    }
    mat X = U.t()*s_inv_sq*U;

    // Form the hamiltonian in the orthogonalized basis: H = XT HX
    mat orth_H = X.t()*H*X;

    vec e; // main diagonal of D matrix
    mat V; // U matrix
    eig_sym(e, V, orth_H,"dc");
    mat E = diagmat(e);

    mat C = X*V;

    S.print("S");
    eigvec.print("eigenvec");
    eigval.print("eigenval");
    U.print("U");
    s.print("s");
    s_inv_sq.print("s^-1/2");
    X.print("X");
    orth_H.print("Fancy H");
    E.print("E");
    C.print("C");


    //Solve for eigenvals/vecs of S with Armadillo:
}
