#include <bits/stdc++.h>

#define endl "\n"
#define pb push_back
#define mp make_pair
#define ll long long int
#define fi first
#define se second

const int nx[4] = {1, -1, 0, 0};
const int ny[4] = {0, 0, 1, -1};

using namespace std;

double epsilon = 1e-7;

template<typename T>
class Matrix
{
protected:
    int n, m;
    vector<vector<T>> grid;
public:
    Matrix()
    {
        n = 0;
        m = 0;
    }
    Matrix(int n0, int m0)
    {
        n = n0;
        m = m0;

        grid.assign(n, vector<T>(m, 0));
    }

    friend istream& operator>>(istream& is, Matrix<T> &A);
    friend ostream& operator<<(ostream& os, Matrix<T> &A);

    int getN()
    {
        return n;
    }

    int getM()
    {
        return m;
    }

    void setAtIndex(int i, int j, T v)
    {
        grid[i][j] = v;
    }

    T getAtIndex(int i, int j)
    {
        return grid[i][j];
    }

    vector<vector<T>>* getGrid()
    {
        return &grid;
    }


    bool operator==(Matrix &A)
    {
        if(n != A.getN() || m != A.getM())
        {
            return false;
        }

        vector<vector<T>> *temp = A.getGrid();
        for(int i = 0; i<n; i++)
        {
            for(int j = 0; j<m; j++)
            {
                if(grid[i][j] != (*temp)[i][j])
                    return false;
            }
        }

        return true;
    }

    void operator=(Matrix &A)
    {
        if(n != A.getN() || m != A.getM())
        {
            cout << "Error: the dimensional problem occurred in =" << endl;
            return;
        }

        vector<vector<T>> *temp = A.getGrid();
        for(int i = 0; i<n; i++)
        {
            for(int j = 0; j<m; j++)
            {
                grid[i][j] = (*temp)[i][j];
            }
        }
    }

    Matrix operator+(Matrix &A)
    {
        if(n != A.getN() || m != A.getM())
        {
            cout << "Error: the dimensional problem occurred in +" << endl;
            Matrix err(0, 0);
            return err;
        }

        Matrix D(n, m);

        vector<vector<T>> *tempA = A.getGrid();
        vector<vector<T>> *tempD = D.getGrid();
        for(int i = 0; i<n; i++)
        {
            for(int j = 0; j<m; j++)
            {
                (*tempD)[i][j] = grid[i][j] + (*tempA)[i][j];
            }
        }

        return D;
    }

    Matrix operator-(Matrix &A)
    {
        if(n != A.getN() || m != A.getM())
        {
            cout << "Error: the dimensional problem occurred in -" << endl;
            Matrix err(0, 0);
            return err;
        }

        Matrix E(n, m);

        vector<vector<T>> *tempA = A.getGrid();
        vector<vector<T>> *tempE = E.getGrid();
        for(int i = 0; i<n; i++)
        {
            for(int j = 0; j<m; j++)
            {
                (*tempE)[i][j] = grid[i][j] - (*tempA)[i][j];
            }
        }

        return E;
    }

    Matrix operator*(Matrix &A)
    {
        if(m != A.getN())
        {
            cout << "Error: the dimensional problem occurred in x" << endl;
            Matrix err(0, 0);
            return err;
        }

        Matrix F(n, A.getM());

        vector<vector<T>> *tempA = A.getGrid();
        vector<vector<T>> *tempF = F.getGrid();

        for(int i = 0; i<n; i++)
        {
            for(int j = 0; j<A.getM(); j++)
            {
                int counter = 0;
                while(counter < m)
                {
                    (*tempF)[i][j] += (fabs(grid[i][counter]*(*tempA)[counter][j]) > epsilon)?grid[i][counter]*(*tempA)[counter][j]: 0.0;
                    counter++;
                }
            }
        }

        return F;
    }

    Matrix operator-()
    {
        Matrix temp = *this;
        for(int i = 0; i<n; i++)
        {
            for(int j = 0; j<m; j++)
            {
                temp.setAtIndex(i, j, -grid[i][j]);
            }
        }

        return temp;
    }

    Matrix transpose()
    {
        Matrix G(m, n);
        vector<vector<T>> *temp = G.getGrid();

        for(int i = 0; i<n; i++)
        {
            for(int j = 0; j<m; j++)
            {
                (*temp)[j][i] = grid[i][j];
            }
        }

        return G;
    }

    Matrix* augmentedMatrix()
    {
        Matrix* aug = new Matrix(n, 2*m);

        vector<vector<T>> *temp = aug->getGrid();

        for(int i = 0; i<n; i++)
        {
            for(int j = 0; j<m; j++)
            {
                (*temp)[i][j] = grid[i][j];
            }
        }

        for(int i = 0; i<n; i++)
        {
            (*temp)[i][i+m] = 1;
        }

        return aug;
    }
};

template<typename T>
class SquareMatrix: public Matrix<T>
{
public:
    SquareMatrix()
    {
        Matrix<T>::n = 0;
        Matrix<T>::m = 0;
    }

    SquareMatrix(int n0)
    {
        Matrix<T>::n = n0;
        Matrix<T>::m = n0;
        Matrix<T>::grid.assign(Matrix<T>::n, vector<T>(Matrix<T>::n, 0));
    }
};

template<typename T>
class IdentityMatrix: public SquareMatrix<T>
{
public:
    IdentityMatrix()
    {
        Matrix<T>::n = 0;
        Matrix<T>::m = 0;
    }

    IdentityMatrix(int n0)
    {
        Matrix<T>::n = n0;
        Matrix<T>::m = n0;
        Matrix<T>::grid.assign(Matrix<T>::n, vector<T>(Matrix<T>::n, 0));

        for(int i = 0; i<Matrix<T>::n; i++)
        {
            Matrix<T>::grid[i][i] = 1;
        }
    }
};

template<typename T>
class EliminationMatrix: public SquareMatrix<T>
{
public:
    EliminationMatrix()
    {
        Matrix<T>::n = 0;
        Matrix<T>::m = 0;
    }

    EliminationMatrix(int n0)
    {
        Matrix<T>::n = n0;
        Matrix<T>::m = n0;
        Matrix<T>::grid.assign(Matrix<T>::n, vector<T>(Matrix<T>::n, 0));

        for(int i = 0; i<Matrix<T>::n; i++)
        {
            Matrix<T>::grid[i][i] = 1;
        }
    }

    EliminationMatrix(int n0, Matrix<T>* M, int i, int j)
    {
        Matrix<T>::n = n0;
        Matrix<T>::m = n0;
        Matrix<T>::grid.assign(Matrix<T>::n, vector<T>(Matrix<T>::n, 0));

        for(int k = 0; k<Matrix<T>::n; k++)
        {
            Matrix<T>::grid[k][k] = 1;
        }

        vector<vector<T>>* temp = M->getGrid();
        double c = (*temp)[i-1][j-1]/(*temp)[j-1][j-1];

        Matrix<T>::grid[i-1][j-1] = -c;
    }

    void eliminate(Matrix<T>* M, int i, int j)
    {
        vector<vector<T>>* temp = M->getGrid();
        double c = (*temp)[i-1][j-1]/(*temp)[j-1][j-1];

        Matrix<T>::grid[i-1][j-1] = -c;
    }
};

template<typename T>
class PermutationMatrix: public SquareMatrix<T>
{
public:
    PermutationMatrix()
    {
        Matrix<T>::n = 0;
        Matrix<T>::m = 0;
    }

    PermutationMatrix(int n0)
    {
        Matrix<T>::n = n0;
        Matrix<T>::m = n0;
        Matrix<T>::grid.assign(Matrix<T>::n, vector<T>(Matrix<T>::n, 0));

        for(int i = 0; i<Matrix<T>::n; i++)
        {
            Matrix<T>::grid[i][i] = 1;
        }
    }

    PermutationMatrix(int n0, Matrix<T>* M, int i, int j)
    {
        Matrix<T>::n = n0;
        Matrix<T>::m = n0;
        Matrix<T>::grid.assign(Matrix<T>::n, vector<T>(Matrix<T>::n, 0));

        for(int k = 0; k<Matrix<T>::n; k++)
        {
            Matrix<T>::grid[k][k] = 1;
        }

        swap(Matrix<T>::grid[i-1], Matrix<T>::grid[j-1]);
    }

    void permute(Matrix<T>* M, int i, int j)
    {
        swap(Matrix<T>::grid[i-1], Matrix<T>::grid[j-1]);
    }
};

istream& operator>>(istream &is, Matrix<double> *A)
{
    vector<vector<double>> *temp = A->getGrid();
    for(int i = 0; i<A->getN(); i++)
    {
        for(int j = 0; j<A->getM(); j++)
        {
            is >> (*temp)[i][j];
        }
    }

    return is;
}

ostream& operator<<(ostream &os, Matrix<double> &A)
{
    vector<vector<double>> *temp = A.getGrid();
    for(int i = 0; i<A.getN(); i++)
    {
        for(int j = 0; j<A.getM(); j++)
        {
            os << fixed << setprecision(6) << (*temp)[i][j];
            if(j <A.getM()-1)
            {
                cout << ' ';
            }
        }
        cout << endl;
    }

    return os;
}

Matrix<double> calculateInverse(Matrix<double> A, int n)
{
    Matrix<double> aug = *A.augmentedMatrix();

    int step = 1;
    for(int j = 1; j<=n-1; j++)
    {
        int r1 = j, r2 = j;
        for(int i = j+1; i<=n; i++)
        {
            double temp1 = (*(A.getGrid()))[r2-1][j-1];
            double temp2 = (*(A.getGrid()))[i-1][j-1];

            if(fabs(temp2) > fabs(temp1))
            {
                r2 = i;
            }
        }

        if(r1 != r2)
        {
            Matrix<double>* P = new PermutationMatrix<double>(n, &A, r1, r2);
            Matrix<double> temp = (*P)*(A);
            A = temp;

            Matrix<double> temp2 = (*P)*(aug);
            aug = temp2;

            step++;
        }

        for(int i = j+1; i<=n; i++)
        {
            Matrix<double>* E = new EliminationMatrix<double>(n, &A, i, j);
            Matrix<double> temp = (*E)*(A);

            if(A == temp)
            {
                continue;
            }

            A = temp;

            Matrix<double> temp2 = (*E)*(aug);
            aug = temp2;

            step++;
        }
    }


    for(int j = n; j>=2; j--)
    {
        for(int i = j-1; i >= 1; i--)
        {
            Matrix<double>* E = new EliminationMatrix<double>(n, &A, i, j);

            Matrix<double> temp = (*E)*(A);

            if(A == temp)
            {
                continue;
            }

            A = temp;

            Matrix<double> temp2 = (*E)*(aug);
            aug = temp2;

            step++;
        }
    }

    for(int i = 1; i<=n; i++)
    {
        for(int j = i+1; j<=2*n; j++)
        {
            (*(aug.getGrid()))[i-1][j-1] /= (*(aug.getGrid()))[i-1][i-1];
        }

        (*(aug.getGrid()))[i-1][i-1] = 1;
    }

    for(int i = 1; i<=n; i++)
    {
        for(int j = 1; j<=n; j++)
        {
            (*(A.getGrid()))[i-1][j-1] = (*(aug.getGrid()))[i-1][j-1+n];
        }
    }

    return A;
}


bool interior_point_solution_found(int iteration)
{
    if(iteration >= 30)
        return true;
    return false;
}

Matrix<double> *generateRandomPoint(Matrix<double>* A, Matrix<double>* minBoundries, Matrix<double>* maxBoundries){
    int m = A->getN();
    int n = A->getM() - m;
    Matrix<double> *generatedPoint = new Matrix<double>(n+m, 1);

    // Generate a random point within the boundries of each variable
    for(int i=0; i<n; i++){
        double min = minBoundries->getAtIndex(0, i);
        double max = maxBoundries->getAtIndex(0, i);
        double generatedValue = min + static_cast<double>(rand()) / RAND_MAX * (max - min);
        generatedPoint->setAtIndex(i, 0, generatedValue);
    }

    return generatedPoint;
}

bool inBoundPoint(Matrix<double>* A, Matrix<double>* b, Matrix<double>* point){
    // Check if the point is in the bounded region
    // Verify the validity of the constraints
    int m = A->getN();
    int n = A->getM() - m;
    for(int i=0; i<m; i++){
        // Constraint: a_1.x_1 + a_2.x_2 + a_3.x_3 + .. + a_n.x_n < b_m
        double RHS = b->getAtIndex(i, 0);
        double LHS = 0;
        for(int j=0; j<n; j++){
            LHS = LHS + (A->getAtIndex(i, j)*point->getAtIndex(j, 0));
        }
        if(LHS >= RHS) return false;

        point->setAtIndex(i+n, 0, RHS-LHS);
    }
    return true;
}

Matrix<double> *findInitialPoint(Matrix<double> *A, Matrix<double>* b){
    int m = A->getN();
    int n = A->getM() - m;

    // Matrices to store the minimum and maximum boundries of each varible in the objective function
    Matrix<double> *minBoundries = new Matrix<double>(1, n+m);
    Matrix<double> *maxBoundries = new Matrix<double>(1, n+m);

    // Calculate the maximum boundry for each variable in the objective function
    for(int i=0; i<n; i++){
        maxBoundries->setAtIndex(0, i, 100);
    }


    // Generate random points within the boundries of each variable
    // Check if the point is a feasible solution in the bounded region
    int maxGenerated = 100000;
    int generated = 1;
    Matrix<double>* point = generateRandomPoint(A, minBoundries, maxBoundries);
    while(!(inBoundPoint(A, b, point)) && generated < maxGenerated){
        free(point);
        point = generateRandomPoint(A, minBoundries, maxBoundries);
        generated++;
    }
    if(generated < maxGenerated){
        Matrix<double> *feasibleSolution = new Matrix<double>(n+m, 1); 
        for(int i=0; i<n+m; i++){
            feasibleSolution->setAtIndex(i, 0, point->getAtIndex(i, 0));
        }
    
        return feasibleSolution;
    }

    // No feasible solution in the bounded region is found
    return nullptr;
}

bool isOptimal(double prevObjectiveValue, double currentObjectiveValue, double precision) {
    double change = fabs(prevObjectiveValue - currentObjectiveValue);
    return change <= precision;
}

void interiorPointAlgorithm(Matrix<double>* C, Matrix<double>* A, Matrix<double>* b, double precision, string purpose, double alpha)
{
    int m = A->getN(); //number of constraints
    int n = A->getM() - m; //number of coefficients in the original objective function
    //the size of C is n+m
    //The size of b is m


    // Minimum case
    if(purpose == "minimum")
    {
        for(int i = 0; i<n+m; i++)
        {
            C->setAtIndex(i, 0, -(C->getAtIndex(i, 0)));
        }
    }
    else if(purpose != "maximum")
    {
        cout << "You have to chose either minimum or maximum!" << endl;
        return;
    }
  
    // Step 1: check if the programming problem is linear or quadratic
    // In our case, the programming problem is always linear as we only get the input for the coefficients
    // but now for the terms.

    // Step 2: Verify the objective function is diffrentiable and continuous over the feasible region
    // The LPP is always linear in this program, therefore it is always differentiable and continuous
    // over the feasible region

    Matrix<double> *initial_point = new Matrix<double>(n+m, 1);
    initial_point = findInitialPoint(A, b);
    
    
    if(initial_point == nullptr) // If the feasable region is empty
    {
        cout << "The problem does not have a solution!" << endl;
        return;
    }
    Matrix<double> x = *initial_point;
    cout << "The initial trial solution that lies in the interior of the feasible region:" << endl;
    cout << x << endl;
    cout << endl;
    int iteration = 0;

    double prevObjectiveValue = numeric_limits<double>::max();
    //For each iteration
    while(true)
    {
        cout << "Iteration " << iteration << ':' << endl << endl;
        //Check if the problem has a solution (most probably about unbounded problem) (The job of tester 2)
        //Step 1: Calculate D
        Matrix<double> *D = new Matrix<double>(n+m, n+m);
        for(int i = 0; i<n+m; i++)
        {
            D->setAtIndex(i, i, x.getAtIndex(i, 0));
        }

        //Step 2: Calculate AA and cc
        Matrix<double> AA = (*A) * (*D); //[m x n+m]
        Matrix<double> cc = (*D) * (*C); // [n+m x 1]



        //Step 3: Calculate P and cp
        Matrix<double> *I = new IdentityMatrix<double>(n+m); //[n+m x n+m]
        Matrix<double> AAT = AA.transpose(); //[n+m x m]
        Matrix<double> AA_AAT = AA * AAT; //[m x m]
        Matrix<double> AA_AAT_inverse = calculateInverse(AA_AAT, AA_AAT.getN()); // [m x m]
        Matrix<double> checker = AA_AAT_inverse*AA_AAT;
        Matrix<double> temp1 = AAT * AA_AAT_inverse; //[n+m x m]
        Matrix<double> temp2 = temp1 * AA; //[n+m x n+m]
        Matrix<double> P = *I - temp2; //[n+m x n+m]

        cout << "P in iteration " << iteration << ':' << endl;
        cout << P;


        Matrix<double> cp = P * cc; // [n+m x 1]
        cout << "cp in iteration " << iteration << ':' << endl;

        //Step 4: Identify the most negative element in cp and set v to its abs value
        double v = 1.0; //1.0 is the flag value
        for(int i=0; i<n+m; i++)
        {
            if(cp.getAtIndex(i, 0) < 0)
                if(cp.getAtIndex(i, 0) < v)
                    v = cp.getAtIndex(i, 0);
        }
        v = fabs(v);
        cout << "the value of v in the iteration " << iteration << ':' << endl;
        cout << v << endl << endl;
        //Calculate xx
        Matrix<double> *one_vector = new Matrix<double>(n+m, 1);
        Matrix<double> *cp_alpha_v = new Matrix<double>(n+m, 1);
        for(int i = 0; i<n+m; i++)
        {
            one_vector->setAtIndex(i, 0, 1);
            cp_alpha_v->setAtIndex(i, 0, (alpha/v)*cp.getAtIndex(i, 0));
        }
        Matrix<double> xx = (*one_vector) + (*cp_alpha_v);


        //Step 5: calculate x
        Matrix<double> temp = (*D) * xx;
        x = temp;

        cout << "Current vector x:" << endl;
        cout << x << endl;
        //Check the precision of x
        Matrix<double> CT = C->transpose();

        Matrix<double> res = CT * x;
        cout << "Current result:" << endl;
        cout << res;

        if(isOptimal(prevObjectiveValue, res.getAtIndex(0, 0), precision) || interior_point_solution_found(iteration))
        {
            cout << "Solution found with interior point algorithm in the last iteration " << iteration << '!' << endl;
            cout << "x:" << endl;
            cout << x << endl;

            cout << "The " << purpose << " is:" << endl;
            cout << res.getAtIndex(0, 0) << endl;
            cout << endl;
            break;
        }
        else
        {
            cout << "x for iteration " << iteration << ':' << endl;
            cout << x << endl;
            cout << endl;
        }
        prevObjectiveValue = res.getAtIndex(0, 0);
        iteration++;
        // turns out that the number of iterations is close to "infinity" => the problem is not bounded
        if(iteration > 1000){
            cout << "Interior Point Method inapplicable!" << endl;
            exit(0);
        }
    }

}

int main()
{
    cout << "NOTE: The problem must be in standard form! If it is not, rewrite the problem in standard form." << endl;
    cout << "-----------------------------------------------------------------------------------------------" << endl;
    int n;
    cout << "Type the number of variables in the objective function:" << endl;
    cin >> n;


    int m;
    cout << "Type the number of constraint functions:" << endl;
    cin >> m;

    Matrix<double> *C = new Matrix<double>(n+m, 1);
    cout << "Type the " << n+m << " coefficients of the augmented objective function, seperated by spaces:" << endl;
    cin >> C;

    Matrix<double> *A = new Matrix<double>(m, n+m);
    if (m == 1) cout << "Type the coefficients of the constraint function. The " << m << " line must contain " << n+m << " coefficients, seperated by spaces:" << endl;
    else cout << "Type the coefficients of the constraint function. Each of the " << m << " lines must contain " << n+m << " coefficients, seperated by spaces:" << endl;
    cin >> A;

    Matrix<double> *b = new Matrix<double>(m, 1);
    if (m == 1) cout << "Type the right hand side of the constraint containing one number:" << endl;
    else cout << "Type the right hand side of the constraints containing " << m << " numbers, seperated by spaces:" << endl;
    cin >> b;

    double precision;
    cout << "Type the approximation accuracy:" << endl;
    cin >> precision;

    string purpose;
    cout << "Type maximum or minimum:" << endl;
    cin >> purpose;

    cout << "------------------------------------------------------------------------------------------------------------------------------------" << endl;

    cout << "------------------------------------------------------------------------------------------------------------------------------------" << endl;
    cout << "Solving for alpha = 0.5..." << endl;
    interiorPointAlgorithm(C, A, b, precision, purpose, 0.5);
    cout << "------------------------------------------------------------------------------------------------------------------------------------" << endl;
    cout << "Solving for alpha = 0.9..." << endl;
    interiorPointAlgorithm(C, A, b, precision, purpose, 0.9);
    cout << "------------------------------------------------------------------------------------------------------------------------------------" << endl;

    return 0;
}