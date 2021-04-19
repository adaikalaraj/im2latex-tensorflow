import numpy as np
import re
from IPython.display import display, Math, Latex, Image


import numpy as np
imgs = np.load('pred_imgs.npy')
preds = np.load('pred_latex.npy')
properties = np.load('properties.npy', allow_pickle=True).tolist()
displayPreds = lambda Y: display(Math(Y.split('#END')[0]))
idx_to_chars = lambda Y: ' '.join(map(lambda x: properties['idx_to_char'][x],Y))
#displayIdxs = lambda Y: display(Math(''.join(map(lambda x: properties['idx_to_char'][x],Y))))



import PIL.Image
from cStringIO import StringIO
import IPython.display
import numpy as np
def showarray(a, fmt='png'):
    a = np.uint8(a)
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    IPython.display.display(IPython.display.Image(data=f.getvalue()))


 
batch_size=1
from PIL import Image as Img
for i in xrange(batch_size):
    preds_chars = idx_to_chars(preds[i,1:]).replace('$','')
    print "Original (Input) Image: %d"%(i+1)
    showarray(imgs[i][0])
    print "Predicted Latex"
    print preds_chars.split('#END')[0]
    print "\nRendering the predicted latex"
    displayPreds(preds_chars)
    print "\n"  