############
# dspev!
############

if isdefined(Prox, :dspev!)

println("testing dspev!")

a = [1.0,2.0,3.0,5.0,6.0,9.0]
W_ref = [0.0,0.6992647456322766,14.300735254367698]
Z_ref = [0.9486832980505137	0.17781910596911388	-0.26149639682478454;
  0.0	-0.8269242138935418	-0.5623133863572413;
  -0.3162277660168381	0.5334573179073402	-0.7844891904743537]
A_ref = [1.0 2.0 3.0; 2.0 5.0 6.0; 3.0 6.0 9.0]

W, Z = Prox.dspev!('V','L',a)

A = Z*diagm(W)*Z'

@test all((W-W_ref)./(1+abs(W_ref)) .<= 1e-8)
@test all((A-A_ref)./(1+abs(A_ref)) .<= 1e-8)

end