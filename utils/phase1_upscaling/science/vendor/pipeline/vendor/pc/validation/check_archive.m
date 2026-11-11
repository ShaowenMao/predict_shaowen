function check_archive(numeric,model,output,expected_hash,receipt)
addpath(fullfile(numeric,'vendor'));m=read_reference_model(model);
rows=readmatrix(fullfile(output,'pc_curve.csv'));seen=0;
meta=read_pc_archive(fullfile(output,'states.bin'),@check,expected_hash);
assert(seen==size(rows,1)&&seen==meta.points);
save_json(receipt,struct('all_local_kr_bitwise_equal',true,'rows',seen,'streaming_matlab_consumer_passed',true));
    function check(j,s,kw,kg)
        assert(j==seen+1&&s.pressure==rows(j,1));[a,b]=phase_values(m,s.sg);
        assert(isequal(kw,a)&&isequal(kg,b),'JuliaPc:Interchange','Local Kr field changed on archive read');
        pv=sum(m.pv);actual=[sum(m.pv.*s.sg),sum(m.pv(s.trapped)),sum(m.pv(kw==0)), ...
            double(isfinite(s.breakthrough))*pv,sum(m.pv(s.invaded))]/pv;
        assert(max(abs(actual-rows(j,2:6)))<=2e-12);seen=j;
    end
end
