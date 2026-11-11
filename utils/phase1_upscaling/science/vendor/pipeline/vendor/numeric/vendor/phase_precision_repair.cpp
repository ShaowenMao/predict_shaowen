#include "mex.h"
#include <quadmath.h>
#include <vector>
#include <cstring>

// Keep the pressure correction and conservative face sums in binary128.
// The double sparse matrix is only the approximate correction solver.
static const double* field(const mxArray* p, const char* s) {
    const mxArray* a = mxGetField(p, 0, s);
    if (!a || !mxIsDouble(a) || mxIsComplex(a)) mexErrMsgIdAndTxt("Study:Input", "Invalid field %s", s);
    return mxGetPr(a);
}
static mwSize count(const mxArray* p, const char* s) {
    return mxGetNumberOfElements(mxGetField(p, 0, s));
}
static mwIndex index(double v, mwSize n) {
    if (v < 1 || v > n || v != (double)(mwIndex)v) mexErrMsgIdAndTxt("Study:Index", "Invalid node index");
    return (mwIndex)v - 1;
}
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs != 3 || !mxIsChar(prhs[0])) mexErrMsgIdAndTxt("Study:Input", "Expected action, data, pressure/correction");
    char action[16]; mxGetString(prhs[0], action, sizeof(action));
    if (!strcmp(action, "apply")) {
        if (nlhs != 1 || !mxIsStruct(prhs[1]) || !mxIsDouble(prhs[2]))
            mexErrMsgIdAndTxt("Study:Input", "Invalid matrix-free apply");
        const mxArray* p = prhs[1];
        mwSize n = mxGetNumberOfElements(prhs[2]);
        if (count(p, "scale") != n) mexErrMsgIdAndTxt("Study:Input", "Scale dimension differs");
        const double* v = mxGetPr(prhs[2]); const double* scale = field(p, "scale");
        std::vector<__float128> x(n), y(n, 0);
        for (mwSize k = 0; k < n; ++k) x[k] = (__float128)v[k] * (__float128)scale[k];
        const double* i = field(p, "i"); const double* j = field(p, "j");
        const double* t = field(p, "tt");
        for (mwSize k = 0; k < count(p, "tt"); ++k) {
            mwIndex a = index(i[k], n), b = index(j[k], n);
            __float128 flux = (__float128)t[k] * (x[a] - x[b]);
            y[a] += flux; y[b] -= flux;
        }
        i = field(p, "ii"); t = field(p, "tin");
        for (mwSize k = 0; k < count(p, "tin"); ++k) {
            mwIndex a = index(i[k], n); y[a] += (__float128)t[k] * x[a];
        }
        i = field(p, "oo"); t = field(p, "tout");
        for (mwSize k = 0; k < count(p, "tout"); ++k) {
            mwIndex a = index(i[k], n); y[a] += (__float128)t[k] * x[a];
        }
        plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
        double* result = mxGetPr(plhs[0]);
        for (mwSize k = 0; k < n; ++k) result[k] = (double)(y[k] * (__float128)scale[k]);
        return;
    }
    if (!strcmp(action, "add")) {
        const mxArray* x = prhs[1]; const mxArray* dx = prhs[2];
        mwSize n = mxGetM(x);
        if (nlhs != 1 || mxGetN(x) != 2 || mxGetNumberOfElements(dx) != n) mexErrMsgIdAndTxt("Study:Input", "Invalid pressure update");
        const double* a = mxGetPr(x); const double* d = mxGetPr(dx);
        plhs[0] = mxCreateDoubleMatrix(n, 2, mxREAL); double* y = mxGetPr(plhs[0]);
        for (mwSize k = 0; k < n; ++k) {
            __float128 v = (__float128)a[k] + (__float128)a[k+n] + (__float128)d[k];
            y[k] = (double)v; y[k+n] = (double)(v - (__float128)y[k]);
        }
        return;
    }
    if (strcmp(action, "audit") || nlhs != 2 || !mxIsStruct(prhs[1]) || mxGetN(prhs[2]) != 2)
        mexErrMsgIdAndTxt("Study:Input", "Invalid audit call");
    const mxArray* p = prhs[1]; mwSize n = mxGetM(prhs[2]); const double* v = mxGetPr(prhs[2]);
    std::vector<__float128> x(n), residual(n, 0), rhs(n, 0);
    __float128 pmin = 2, pmax = -1, energy = 0, qin = 0, qout = 0;
    for (mwSize k = 0; k < n; ++k) {
        x[k] = (__float128)v[k] + (__float128)v[k+n];
        pmin = fminq(pmin, x[k]); pmax = fmaxq(pmax, x[k]);
    }
    const double* i = field(p,"i"); const double* j = field(p,"j"); const double* t = field(p,"tt");
    for (mwSize k = 0; k < count(p,"tt"); ++k) {
        mwIndex a = index(i[k],n), b = index(j[k],n);
        __float128 d = x[a]-x[b], f = (__float128)t[k]*d;
        residual[a] -= f; residual[b] += f; energy += f*d;
    }
    i = field(p,"ii"); t = field(p,"tin");
    for (mwSize k = 0; k < count(p,"tin"); ++k) {
        mwIndex a = index(i[k],n); __float128 d = 1-x[a], f = (__float128)t[k]*d;
        residual[a] += f; rhs[a] += (__float128)t[k]; qin += f; energy += f*d;
    }
    i = field(p,"oo"); t = field(p,"tout");
    for (mwSize k = 0; k < count(p,"tout"); ++k) {
        mwIndex a = index(i[k],n); __float128 f = (__float128)t[k]*x[a];
        residual[a] -= f; qout += f; energy += f*x[a];
    }
    __float128 rr=0, bb=0;
    plhs[0] = mxCreateDoubleMatrix(n,1,mxREAL); double* r = mxGetPr(plhs[0]);
    for (mwSize k=0; k<n; ++k) { rr+=residual[k]*residual[k]; bb+=rhs[k]*rhs[k]; r[k]=(double)residual[k]; }
    plhs[1] = mxCreateDoubleMatrix(1,8,mxREAL); double* a = mxGetPr(plhs[1]);
    a[0]=(double)sqrtq(rr/bb); a[1]=(double)qin; a[2]=(double)qout; a[3]=(double)energy;
    a[4]=(double)(fabsq(qin-qout)/fmaxq(fabsq(qin),fabsq(qout)));
    a[5]=(double)(fabsq(energy-qout)/fabsq(qout)); a[6]=(double)pmin; a[7]=(double)pmax;
}
