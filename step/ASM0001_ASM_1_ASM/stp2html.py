from OCC.Core import STEPControl
#from OCC.Display.WebGl import x3dom_renderer_osv
from OCC.Display.WebGl import x3dom_renderer_osv
import pdb
import sys
import random
#
# usage stp2html.py filename level
#
filename = sys.argv[1]
level = sys.argv[2]
fileout = filename.replace('.stp','')
filehtml = filename.replace('.stp','.html')
step_reader = STEPControl.STEPControl_Reader()
step_reader.ReadFile(filename)
step_reader.TransferRoots()
shape = step_reader.Shape()
#my_renderer = x3dom_renderer.X3DomRenderer(path='.')
my_renderer = x3dom_renderer_osv.X3DomRenderer(path='.',filehtml=filehtml)
my_renderer.DisplayShape(shape)
my_renderer.GenerateHTMLFile(filename=fileout,level=level)
#my_renderer.render()
#my_renderer.render()
