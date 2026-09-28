#pragma once
#include <cstddef>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

// Minimal ABI adapter for the unchanged, sealed numerical MEX source files.
using mwSize = std::size_t;
using mwIndex = std::size_t;
constexpr int mxREAL = 0;
struct mxArray {
    enum Kind { Double, Struct, String } kind = Double;
    mwSize m = 0, n = 0;
    double* data = nullptr;
    std::vector<double> owned;
    std::map<std::string, const mxArray*> fields;
    std::string text;
};
inline bool mxIsDouble(const mxArray* a) { return a && a->kind == mxArray::Double; }
inline bool mxIsComplex(const mxArray*) { return false; }
inline bool mxIsStruct(const mxArray* a) { return a && a->kind == mxArray::Struct; }
inline bool mxIsChar(const mxArray* a) { return a && a->kind == mxArray::String; }
inline mwSize mxGetM(const mxArray* a) { return a->m; }
inline mwSize mxGetN(const mxArray* a) { return a->n; }
inline mwSize mxGetNumberOfElements(const mxArray* a) { return a->m*a->n; }
inline double* mxGetPr(const mxArray* a) { return a->data; }
inline const mxArray* mxGetField(const mxArray* a, mwIndex, const char* s) {
    auto it=a->fields.find(s);return it==a->fields.end() ? nullptr : it->second;
}
inline void mxGetString(const mxArray* a,char* s,mwSize n) {
    if(n) { std::strncpy(s,a->text.c_str(),n-1);s[n-1]=0; }
}
inline mxArray* mxCreateDoubleMatrix(mwSize m,mwSize n,int) {
    mxArray* a=new mxArray;a->m=m;a->n=n;a->owned.resize(m*n,0.0);a->data=a->owned.data();return a;
}
[[noreturn]] inline void mexErrMsgIdAndTxt(const char* id,const char* format,...) {
    char msg[1024];va_list args;va_start(args,format);vsnprintf(msg,sizeof(msg),format,args);va_end(args);
    throw std::runtime_error(std::string(id)+": "+msg);
}
