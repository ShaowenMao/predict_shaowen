#include "mex.h"
#include <array>
#include <cstdint>

void legacy_mex(int,mxArray**,int,const mxArray**);
void repair_mex(int,mxArray**,int,const mxArray**);
static thread_local std::string last_error;
static mxArray view(const double* p,std::size_t m,std::size_t n=1) {
    mxArray a;a.data=const_cast<double*>(p);a.m=m;a.n=n;return a;
}
static mxArray action(const char* s) { mxArray a;a.kind=mxArray::String;a.text=s;return a; }
extern "C" const char* phase_error() { return last_error.c_str(); }

extern "C" int phase_eval(int mode,std::int64_t n,std::int64_t ne,std::int64_t ni,std::int64_t no,
    const double* i,const double* j,const double* t,const double* ii,const double* oo,
    const double* tin,const double* tout,const double* scale,const double* x,double* y,double* metrics) {
    mxArray* outputs[2]={nullptr,nullptr};
    try {
        if(n<=0||ne<0||ni<0||no<0||(mode!=0&&mode!=1)) throw std::runtime_error("Invalid dimensions or action");
        std::array<mxArray,8> f={view(i,ne),view(j,ne),view(t,ne),view(ii,ni),view(oo,no),
                                view(tin,ni),view(tout,no),view(scale,n)};
        const char* names[]={"i","j","tt","ii","oo","tin","tout","scale"};
        mxArray parts;parts.kind=mxArray::Struct;
        for(int k=0;k<8;++k) parts.fields[names[k]]=&f[k];
        mxArray a=action(mode==0?"audit":"apply"),p=view(x,n,mode==0?2:1);
        const mxArray* inputs[]={&a,&parts,&p};
        if(mode==0) legacy_mex(2,outputs,3,inputs);else repair_mex(1,outputs,3,inputs);
        std::memcpy(y,outputs[0]->data,n*sizeof(double));
        if(mode==0) std::memcpy(metrics,outputs[1]->data,8*sizeof(double));
        delete outputs[0];delete outputs[1];last_error.clear();return 0;
    } catch(const std::exception& e) {
        delete outputs[0];delete outputs[1];last_error=e.what();return 1;
    }
}
extern "C" int phase_add(std::int64_t n,const double* x,const double* dx,double* y) {
    mxArray* outputs[1]={nullptr};
    try {
        if(n<=0) throw std::runtime_error("Invalid pressure dimension");
        mxArray a=action("add"),p=view(x,n,2),d=view(dx,n);
        const mxArray* inputs[]={&a,&p,&d};legacy_mex(1,outputs,3,inputs);
        std::memcpy(y,outputs[0]->data,2*n*sizeof(double));delete outputs[0];last_error.clear();return 0;
    } catch(const std::exception& e) {delete outputs[0];last_error=e.what();return 1;}
}
