// Benchmark created by MQT Bench on 2026-03-17
// For more info: https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: 2.1.0
// Qiskit version: 2.1.1
// Output format: qasm3

OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
gate rccx _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  t _gate_q_2;
  cx _gate_q_1, _gate_q_2;
  tdg _gate_q_2;
  cx _gate_q_0, _gate_q_2;
  t _gate_q_2;
  cx _gate_q_1, _gate_q_2;
  tdg _gate_q_2;
  h _gate_q_2;
}
gate c3sx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3 {
  h _gate_q_3;
  cp(pi/8) _gate_q_0, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_0, _gate_q_1;
  h _gate_q_3;
  cp(-pi/8) _gate_q_1, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_0, _gate_q_1;
  h _gate_q_3;
  cp(pi/8) _gate_q_1, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_1, _gate_q_2;
  h _gate_q_3;
  cp(-pi/8) _gate_q_2, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_0, _gate_q_2;
  h _gate_q_3;
  cp(pi/8) _gate_q_2, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_1, _gate_q_2;
  h _gate_q_3;
  cp(-pi/8) _gate_q_2, _gate_q_3;
  h _gate_q_3;
  cx _gate_q_0, _gate_q_2;
  h _gate_q_3;
  cp(pi/8) _gate_q_2, _gate_q_3;
  h _gate_q_3;
}
gate dcx _gate_q_0, _gate_q_1 {
  cx _gate_q_0, _gate_q_1;
  cx _gate_q_1, _gate_q_0;
}
gate rcccx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3 {
  h _gate_q_3;
  t _gate_q_3;
  cx _gate_q_2, _gate_q_3;
  tdg _gate_q_3;
  h _gate_q_3;
  cx _gate_q_0, _gate_q_3;
  t _gate_q_3;
  cx _gate_q_1, _gate_q_3;
  tdg _gate_q_3;
  cx _gate_q_0, _gate_q_3;
  t _gate_q_3;
  cx _gate_q_1, _gate_q_3;
  tdg _gate_q_3;
  h _gate_q_3;
  t _gate_q_3;
  cx _gate_q_2, _gate_q_3;
  tdg _gate_q_3;
  h _gate_q_3;
}
gate csdg _gate_q_0, _gate_q_1 {
  tdg _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  t _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  tdg _gate_q_1;
}
gate sxdg _gate_q_0 {
  s _gate_q_0;
  h _gate_q_0;
  s _gate_q_0;
}
gate xx_plus_yy(p0, p1) _gate_q_0, _gate_q_1 {
  rz(p1) _gate_q_0;
  sdg _gate_q_1;
  sx _gate_q_1;
  s _gate_q_1;
  s _gate_q_0;
  cx _gate_q_1, _gate_q_0;
  ry((-0.5)*p0) _gate_q_1;
  ry((-0.5)*p0) _gate_q_0;
  cx _gate_q_1, _gate_q_0;
  sdg _gate_q_0;
  sdg _gate_q_1;
  sxdg _gate_q_1;
  s _gate_q_1;
  rz(-p1) _gate_q_0;
}
gate cs _gate_q_0, _gate_q_1 {
  t _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  tdg _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  t _gate_q_1;
}
gate csx _gate_q_0, _gate_q_1 {
  h _gate_q_1;
  cs _gate_q_0, _gate_q_1;
  h _gate_q_1;
}
gate r(p0, p1) _gate_q_0 {
  U(p0, -pi/2 + p1, pi/2 - p1) _gate_q_0;
}
bit[10] meas;
qubit[10] q;
ccz q[0], q[3], q[1];
x q[1];
t q[1];
t q[4];
rccx q[2], q[7], q[6];
cswap q[5], q[9], q[8];
rccx q[5], q[4], q[3];
c3sx q[6], q[2], q[0], q[8];
tdg q[0];
cry(2.7511828987678966) q[3], q[2];
dcx q[7], q[9];
ccx q[5], q[9], q[4];
rccx q[4], q[9], q[0];
ccz q[7], q[8], q[6];
rccx q[5], q[1], q[7];
u3(3.195812521645201, 2.039548883385027, 1.0844822111232935) q[1];
cswap q[5], q[0], q[7];
rcccx q[6], q[8], q[3], q[2];
u3(3.3401519228349494, 0.08191660178282069, 0.12487864535773079) q[2];
csdg q[3], q[4];
cswap q[6], q[9], q[8];
xx_plus_yy(5.625348551233995, 1.454894982862097) q[0], q[9];
cswap q[5], q[3], q[8];
csx q[2], q[3];
r(1.7859367726051267, 2.867615720945729) q[5];
ry(2.453068833014572) q[6];
ccx q[7], q[4], q[1];
ccx q[4], q[6], q[1];
ccx q[8], q[5], q[6];
rccx q[9], q[0], q[7];
sdg q[0];
rccx q[4], q[3], q[7];
cswap q[0], q[7], q[5];
rccx q[3], q[8], q[6];
sx q[7];
sdg q[8];
cswap q[9], q[1], q[2];
c3sx q[9], q[4], q[1], q[2];
ch q[1], q[9];
ccz q[5], q[3], q[4];
s q[4];
rx(3.464488948976134) q[5];
ccx q[6], q[0], q[2];
rcccx q[1], q[6], q[9], q[3];
tdg q[1];
rccx q[8], q[7], q[2];
rcccx q[2], q[6], q[7], q[5];
ccx q[3], q[8], q[0];
ccx q[4], q[6], q[7];
U(2.077793364252715, 5.936603552194878, 2.4951327774590375) q[4];
rcccx q[8], q[0], q[3], q[2];
cswap q[2], q[0], q[3];
cswap q[8], q[6], q[7];
ry(3.709708394692678) q[8];
s q[9];
rccx q[1], q[9], q[5];
crx(4.622701549534623) q[1], q[4];
s q[5];
rccx q[5], q[2], q[7];
ccz q[8], q[7], q[4];
rx(0.4851534132028912) q[8];
u3(3.2145609627525853, 1.6556110685025274, 0.18571571838936715) q[9];
rccx q[3], q[9], q[6];
rccx q[0], q[9], q[2];
rcccx q[6], q[5], q[3], q[1];
ccx q[1], q[7], q[5];
u1(3.141693083085852) q[3];
rcccx q[4], q[6], q[9], q[0];
ccz q[2], q[0], q[5];
rccx q[2], q[5], q[3];
ccz q[8], q[1], q[6];
ccz q[9], q[4], q[7];
ccz q[4], q[1], q[6];
rccx q[1], q[6], q[9];
cswap q[8], q[0], q[7];
cswap q[0], q[2], q[5];
ccx q[8], q[3], q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
meas[0] = measure q[0];
meas[1] = measure q[1];
meas[2] = measure q[2];
meas[3] = measure q[3];
meas[4] = measure q[4];
meas[5] = measure q[5];
meas[6] = measure q[6];
meas[7] = measure q[7];
meas[8] = measure q[8];
meas[9] = measure q[9];
