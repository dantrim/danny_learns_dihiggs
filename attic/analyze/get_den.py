#!/usr/bin/env python

import sys
reco_eff = float(sys.argv[1])
truth_eff = float(sys.argv[2])
k = 0.051
BR = 2 * 0.57 * 0.21
#k = 0.01244
acc = truth_eff / 21298.8
e = reco_eff / truth_eff
den = acc * e * k * BR

print( "acceptance = {}, reco eff = {}, acc x e x k = {}".format(acc, e, acc*e*k))
print( "DEN = {}".format(den))
xsecUL = float(sys.argv[3])
print( "UL  = {}".format(xsecUL / den))
