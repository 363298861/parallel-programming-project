// Advection Coefficient class
// written by Peter Strazdins, Oct 18

  inline void N2Coeff(double v, double &cm1, double &c0, double &cp1) {
    double v2 = v/2.0;
    cm1 = v2*(v+1.0);
    c0  = 1.0 - v*v;
    cp1 = v2*(v-1.0);    
  }; 

  inline void W2Coeff(double v, double &cm2, double &c0, double &cp2) {
    v /= 2.0;
    double v2 = v/2.0;
    cm2 = v2*(v+1.0);
    c0  = 1.0 - v*v;
    cp2 = v2*(v-1.0);    
  }

  inline void F2Coeff(double v, double &cm3, double &cm1, double &cp1, double &cp3) {
    double v6 = v/6.0, vv16 = v*v/16.0,
           vvm1_16 = vv16 - 1.0/16.0;
    cm3 = vvm1_16 + v6;
    cm1 = cp1 = 9.0/16 - vv16;
    cp3 = vvm1_16 - v6;
  }

  inline void N4Coeff(double v, double &cm2, double &cm1, double &c0,
		      double &cp1, double &cp2) {
    double vm1 = v-1.0, vp1 = v+1.0, vm2 = v-2.0, vp2 = v+2.0, v2 = v*v;
    double vvm1vp1_24 = v*vm1*vp1/24.0,
           vvm2vp2_6  = v*vm2*vp2/-6.0;
    cm2 = vvm1vp1_24 * vp2;
    cm1 = vvm2vp2_6  * vp1;
    c0  = 1.0 + v2/4.0*(v2 - 5.0);
    cp1 = vvm2vp2_6  * vm1; 
    cp2 = vvm1vp1_24 * vm2;
  } 


