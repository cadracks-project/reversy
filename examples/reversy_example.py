#!/usr/bin/env python
#coding: utf-8

r"""reversy example use"""


import logging

from reversy.reversy import reverse
import reversy.pointcloud as pc

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s :: %(levelname)6s :: '
                           '%(module)20s :: %(lineno)3d :: %(message)s')

# filename = "../step/0_tabby2.stp"  # OCC compound
filename = "../step/ASM0001_ASM_1_ASM.stp"  # OCC compound
# filename = "../step/arduino.stp"  # OCC compound
# filename = "../step/MOTORIDUTTORE_ASM.stp" # OCC compound
# filename = "../step/aube_pleine.stp"  # OCC Solid
a1 = reverse(filename, view=False)
#
#A = a1.merge_nodes([1, 7, 9, 3, 5])
# B = a1.merge_nodes([0, 6, 8, 2, 4])
# a1.save_json()
# a1 = Assembly()
# basename = os.path.basename(filename)
# rep = os.path.join(os.path.dirname(filename),os.path.splitext(basename)[0])
# filename = os.path.join(rep,os.path.splitext(basename)[0]+'.json')
# a1.from_json(filename)
# ls = []
# a1.view([1,7,9,3,5])
# s0 = a1.node[0]['shape']
# s1 = a1.node[1]['shape']
# s0.to_html('s0.html')
# s1.to_html('s1.html')
# for k in range(10):
#     ls.append(a1.get_node_solid(k))
# cd.view(a1)
