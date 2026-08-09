function tests = test_fault_local_permeability_tensor_contract
%TEST_FAULT_LOCAL_PERMEABILITY_TENSOR_CONTRACT Coordinate-contract tests.
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
testDirectory = fileparts(mfilename('fullpath'));
implementationDirectory = fileparts(testDirectory);
addpath(implementationDirectory);
testCase.TestData.implementationDirectory = implementationDirectory;
end

function teardownOnce(testCase)
rmpath(testCase.TestData.implementationDirectory);
end

function testZeroRotation(testCase)
local = [2, 7, 11];
[tensor, compact] = fault_local_permeability_tensor_contract(local, 0);
verifyEqual(testCase, compact, [7, 0, 0, 2, 0, 11], ...
    'AbsTol', 1e-14);
verifyEqual(testCase, squeeze(tensor(1, :, :)), diag([7, 2, 11]), ...
    'AbsTol', 1e-14);
end

function testSignedOffDiagonal(testCase)
local = repmat([2, 7, 11], 2, 1);
[tensor, compact] = fault_local_permeability_tensor_contract( ...
    local, [30; -30]);
verifyEqual(testCase, compact(1, 5), -compact(2, 5), ...
    'AbsTol', 1e-14);
verifyGreaterThan(testCase, compact(2, 5), 0);
for row = 1:2
    values = eig(squeeze(tensor(row, :, :)));
    verifyEqual(testCase, sort(values), sort(local(row, :))', ...
        'AbsTol', 1e-12);
    verifyGreaterThan(testCase, min(values), 0);
end
end

function testMatrixFormula(testCase)
local = [3, 5, 13];
theta = -22.5;
[tensor, ~] = fault_local_permeability_tensor_contract(local, theta);
c = cosd(theta);
s = sind(theta);
rotation = [0, 1, 0; c, 0, -s; s, 0, c];
expected = rotation*diag(local)*rotation';
verifyEqual(testCase, squeeze(tensor(1, :, :)), expected, ...
    'AbsTol', 1e-12);
end
