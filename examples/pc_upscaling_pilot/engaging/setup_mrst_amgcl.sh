#!/usr/bin/env bash
# Install and compile MRST's AMGCL MEX gateways on Engaging.
#
# This script follows MRST's own pinned AMGCL dependency:
#   https://github.com/ddemidov/amgcl
#   revision 4f260881c7158bc5aede881f5f0ed272df2ab580
#
# It is safe to rerun. Existing dependency folders are reused, and the MEX
# gateways are rebuilt to match the active MATLAB/compiler environment.

set -euo pipefail

source /etc/profile.d/modules.sh 2>/dev/null || true
module load matlab/matlab-2025b

MRST_ROOT="${MRST_ROOT:-/home/shaowen/orcd/pool/predict_shaowen/software/mrst-current}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/home/shaowen/orcd/scratch/predict_shaowen}"
AMGCL_REV="${AMGCL_REV:-4f260881c7158bc5aede881f5f0ed272df2ab580}"

DEP_DIR="${MRST_ROOT}/solvers/linearsolvers/amgcl/dependencies"
AMGCL_DIR="${DEP_DIR}/amgcl-${AMGCL_REV}"
BOOST_DIR="${DEP_DIR}/boost-1_65_1_subset"
WORK_DIR="${SCRATCH_ROOT}/tmp/amgcl_build"

mkdir -p "${DEP_DIR}" "${WORK_DIR}"

if [[ ! -f "${AMGCL_DIR}/amgcl/amg.hpp" ]]; then
    echo "Installing AMGCL ${AMGCL_REV}"
    rm -rf "${AMGCL_DIR}"
    tmp="$(mktemp -d "${SCRATCH_ROOT}/tmp/amgcl-clone-XXXXXX")"
    git clone https://github.com/ddemidov/amgcl.git "${tmp}/amgcl-src"
    git -C "${tmp}/amgcl-src" checkout --detach "${AMGCL_REV}"
    mkdir -p "${AMGCL_DIR}"
    cp -a "${tmp}/amgcl-src"/. "${AMGCL_DIR}/"
    rm -rf "${tmp}"
else
    echo "AMGCL already present: ${AMGCL_DIR}"
fi

if [[ ! -d "${BOOST_DIR}/boost" ]]; then
    echo "Installing MRST Boost header subset"
    rm -rf "${BOOST_DIR}"
    tmpzip="$(mktemp "${SCRATCH_ROOT}/tmp/boost_subset-XXXXXX.zip")"
    curl -L -o "${tmpzip}" \
        "https://www.sintef.no/contentassets/124f261f170947a6bc51dd76aea66129/boost-1_65_1_subset.zip"
    unzip -q "${tmpzip}" -d "${DEP_DIR}"
    rm -f "${tmpzip}"
else
    echo "Boost subset already present: ${BOOST_DIR}"
fi

cat > "${WORK_DIR}/build_amgcl_mrst.m" <<MATLAB
mrstRoot = '${MRST_ROOT}';
cd(mrstRoot);
startup;
mrstModule add linearsolvers ad-core;
buildLinearSolvers(true);
disp('AMGCL_BUILD_DONE');
MATLAB

matlab -batch "cd ${WORK_DIR}; build_amgcl_mrst; exit" \
    | tee "${WORK_DIR}/build_amgcl_mrst.log"

cat > "${WORK_DIR}/test_amgcl_mrst.m" <<MATLAB
mrstRoot = '${MRST_ROOT}';
cd(mrstRoot);
startup;
mrstModule add linearsolvers ad-core;
A = sparse(gallery('poisson', 35));
b = rand(size(A, 1), 1);
x_ref = A\\b;
[x_amg, res_amg, its_amg] = callAMGCL(A, b, ...
    'preconditioner', 'amg', ...
    'coarsening', 'aggregation', ...
    'relaxation', 'spai0', ...
    'solver', 'bicgstab', ...
    'tolerance', 1e-8, ...
    'maxIterations', 1000, ...
    'verbose', false);
rel_direct = norm(A*x_amg - b)/norm(b);
err_direct = norm(x_amg - x_ref)/norm(x_ref);
fprintf('direct_call rel_res=%0.3e err_vs_backslash=%0.3e reported_res=%0.3e iterations=%d\\n', ...
    rel_direct, err_direct, res_amg, its_amg);
lsolve = AMGCLSolverAD('tolerance', 1e-8, 'maxIterations', 1000);
lsolve.setCoarsening('aggregation');
lsolve.setRelaxation('spai0');
lsolve.setSolver('bicgstab');
[x_ad, report_ad] = lsolve.solveLinearSystem(A, b);
rel_ad = norm(A*x_ad - b)/norm(b);
err_ad = norm(x_ad - x_ref)/norm(x_ref);
fprintf('AMGCLSolverAD rel_res=%0.3e err_vs_backslash=%0.3e report_res=%0.3e iterations=%d converged=%d\\n', ...
    rel_ad, err_ad, report_ad.Residual, report_ad.Iterations, report_ad.Converged);
if rel_direct > 1e-6 || rel_ad > 1e-6
    error('AMGCL smoke test failed residual check');
end
disp('AMGCL_SMOKE_TEST_PASSED');
MATLAB

matlab -batch "cd ${WORK_DIR}; test_amgcl_mrst; exit" \
    | tee "${WORK_DIR}/test_amgcl_mrst.log"

echo "AMGCL setup complete."
find "${MRST_ROOT}/solvers/linearsolvers/amgcl/utils" -maxdepth 1 \
    -name "amgcl_matlab*.mexa64" -print -ls
