import sklearn.datasets as datasets


```python
import pandas as pd
```


```python
iris=datasets.load_iris()

```


```python
df=pd.DataFrame(iris.data,columns=iris.feature_names)
```


```python
y=iris.target
```


```python
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()

dtree.fit(df,y)

```




    DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                           max_depth=None, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort='deprecated',
                           random_state=None, splitter='best')




```python

from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
dot_data=StringIO()
export_graphviz(dtree,out_file=dot_data,filled=True,rounded=True,special_characters=True)
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
```




![png](output_6_0.png)




```python
import numpy as np
import scipy
import pandas as pd
import sklearn
import keras.backend as k
import tensorflow as tf

```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    <ipython-input-26-50f849aa46b3> in <module>
          3 import pandas as pd
          4 import sklearn
    ----> 5 import keras.backend as k
          6 import tensorflow as tf
    

    ModuleNotFoundError: No module named 'keras'



```python

import numpy as np
A=np.array([1,2,3])
B=np.array([4,3,0])
print("A+B=",np.add(A,B))
print("A-B=",np.subtract(A,B))
```

    A+B= [5 5 3]
    A-B= [-3 -1  3]
    


```python

A@B
```




    10




```python







import numpy as np
A=np.array([1,2,3])
B=np.array([4,3,0])
print("A+B=",np.add(A,B))
print("A-B=",np.subtract(A,B))
 

A@B

import tensorflow as tfA
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    <ipython-input-25-971738cbc9da> in <module>
          3 import pandas as pd
          4 import sklearn
    ----> 5 import keras.backend as k
          6 import tensorflow as tf
          7 
    

    ModuleNotFoundError: No module named 'keras'



```python
import tensorflow as tf
```


    ---------------------------------------------------------------------------

    ImportError                               Traceback (most recent call last)

    ~\anaconda3\lib\site-packages\tensorflow\python\pywrap_tensorflow.py in <module>
         57 
    ---> 58   from tensorflow.python.pywrap_tensorflow_internal import *
         59 
    

    ~\anaconda3\lib\site-packages\tensorflow\python\pywrap_tensorflow_internal.py in <module>
         27             return _mod
    ---> 28     _pywrap_tensorflow_internal = swig_import_helper()
         29     del swig_import_helper
    

    ~\anaconda3\lib\site-packages\tensorflow\python\pywrap_tensorflow_internal.py in swig_import_helper()
         23             try:
    ---> 24                 _mod = imp.load_module('_pywrap_tensorflow_internal', fp, pathname, description)
         25             finally:
    

    ~\anaconda3\lib\imp.py in load_module(name, file, filename, details)
        241         else:
    --> 242             return load_dynamic(name, filename, file)
        243     elif type_ == PKG_DIRECTORY:
    

    ~\anaconda3\lib\imp.py in load_dynamic(name, path, file)
        341             name=name, loader=loader, origin=path)
    --> 342         return _load(spec)
        343 
    

    ImportError: DLL load failed: 找不到指定的模块。

    
    During handling of the above exception, another exception occurred:
    

    ImportError                               Traceback (most recent call last)

    <ipython-input-1-64156d691fe5> in <module>
    ----> 1 import tensorflow as tf
    

    ~\anaconda3\lib\site-packages\tensorflow\__init__.py in <module>
         39 import sys as _sys
         40 
    ---> 41 from tensorflow.python.tools import module_util as _module_util
         42 from tensorflow.python.util.lazy_loader import LazyLoader as _LazyLoader
         43 
    

    ~\anaconda3\lib\site-packages\tensorflow\python\__init__.py in <module>
         48 import numpy as np
         49 
    ---> 50 from tensorflow.python import pywrap_tensorflow
         51 
         52 # Protocol buffers
    

    ~\anaconda3\lib\site-packages\tensorflow\python\pywrap_tensorflow.py in <module>
         67 for some common reasons and solutions.  Include the entire stack trace
         68 above this error message when asking for help.""" % traceback.format_exc()
    ---> 69   raise ImportError(msg)
         70 
         71 # pylint: enable=wildcard-import,g-import-not-at-top,unused-import,line-too-long
    

    ImportError: Traceback (most recent call last):
      File "C:\Users\yi\anaconda3\lib\site-packages\tensorflow\python\pywrap_tensorflow.py", line 58, in <module>
        from tensorflow.python.pywrap_tensorflow_internal import *
      File "C:\Users\yi\anaconda3\lib\site-packages\tensorflow\python\pywrap_tensorflow_internal.py", line 28, in <module>
        _pywrap_tensorflow_internal = swig_import_helper()
      File "C:\Users\yi\anaconda3\lib\site-packages\tensorflow\python\pywrap_tensorflow_internal.py", line 24, in swig_import_helper
        _mod = imp.load_module('_pywrap_tensorflow_internal', fp, pathname, description)
      File "C:\Users\yi\anaconda3\lib\imp.py", line 242, in load_module
        return load_dynamic(name, filename, file)
      File "C:\Users\yi\anaconda3\lib\imp.py", line 342, in load_dynamic
        return _load(spec)
    ImportError: DLL load failed: 找不到指定的模块。
    
    
    Failed to load the native TensorFlow runtime.
    
    See https://www.tensorflow.org/install/errors
    
    for some common reasons and solutions.  Include the entire stack trace
    above this error message when asking for help.



```python

```
