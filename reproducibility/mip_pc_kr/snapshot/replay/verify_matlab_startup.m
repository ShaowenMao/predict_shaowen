function verify_matlab_startup(receipt)
% Observe startup configuration only; do not load replay/scientific helpers.
    expected_tmp = getenv('TMPDIR');
    actual_tmp = tempdir;
    actual_pref = prefdir;
    assert(strcmp(strip_slash(actual_tmp), strip_slash(expected_tmp)));
    assert(strcmp(strip_slash(actual_pref), strip_slash(getenv('MATLAB_PREFDIR'))));
    assert(strcmp(getenv('TMP'), expected_tmp));
    assert(strcmp(getenv('TEMP'), expected_tmp));
    assert(strcmp(getenv('TEMPDIR'), expected_tmp));
    assert(isfolder(getenv('MCR_CACHE_ROOT')));
    probe = fullfile(actual_tmp, 'matlab_startup_write_probe.txt');
    assert(~isfile(probe) && ~isfile(receipt));
    f = fopen(probe, 'w'); assert(f >= 0);
    fprintf(f, 'MATLAB project temporary directory is writable\n'); fclose(f);
    record = struct('passed', true, 'matlab_version', version, ...
        'tempdir', actual_tmp, 'prefdir', actual_pref, ...
        'mcr_cache_root', getenv('MCR_CACHE_ROOT'), 'write_probe', probe);
    f = fopen(receipt, 'w'); assert(f >= 0);
    fprintf(f, '%s\n', jsonencode(record)); fclose(f);
    fprintf('MATLAB_STARTUP_PASS: %s\n', actual_tmp);
end

function value = strip_slash(value)
    value = regexprep(value, '/+$', '');
end
