# This file was automatically generated by SWIG (http://www.swig.org).
# Version 2.0.11
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.





from sys import version_info
if version_info >= (2,6,0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_DMLLCpp', [dirname(__file__)])
        except ImportError:
            import _DMLLCpp
            return _DMLLCpp
        if fp is not None:
            try:
                _mod = imp.load_module('_DMLLCpp', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    _DMLLCpp = swig_import_helper()
    del swig_import_helper
else:
    import _DMLLCpp
del version_info
try:
    _swig_property = property
except NameError:
    pass # Python < 2.2 doesn't have 'property'.
def _swig_setattr_nondynamic(self,class_type,name,value,static=1):
    if (name == "thisown"): return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name,None)
    if method: return method(self,value)
    if (not static):
        self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)

def _swig_setattr(self,class_type,name,value):
    return _swig_setattr_nondynamic(self,class_type,name,value,0)

def _swig_getattr(self,class_type,name):
    if (name == "thisown"): return self.this.own()
    method = class_type.__swig_getmethods__.get(name,None)
    if method: return method(self)
    raise AttributeError(name)

def _swig_repr(self):
    try: strthis = "proxy of " + self.this.__repr__()
    except: strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except AttributeError:
    class _object : pass
    _newclass = 0


class SayHelloCpp(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, SayHelloCpp, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, SayHelloCpp, name)
    __repr__ = _swig_repr
    def __init__(self): 
        this = _DMLLCpp.new_SayHelloCpp()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DMLLCpp.delete_SayHelloCpp
    __del__ = lambda self : None;
    def hello(self, *args): return _DMLLCpp.SayHelloCpp_hello(self, *args)
SayHelloCpp_swigregister = _DMLLCpp.SayHelloCpp_swigregister
SayHelloCpp_swigregister(SayHelloCpp)


def BcastCpp(*args):
  return _DMLLCpp.BcastCpp(*args)
BcastCpp = _DMLLCpp.BcastCpp

def CalcLocalICpp(*args):
  return _DMLLCpp.CalcLocalICpp(*args)
CalcLocalICpp = _DMLLCpp.CalcLocalICpp

def Scatter1dCpp(*args):
  return _DMLLCpp.Scatter1dCpp(*args)
Scatter1dCpp = _DMLLCpp.Scatter1dCpp

def ScatterCpp(*args):
  return _DMLLCpp.ScatterCpp(*args)
ScatterCpp = _DMLLCpp.ScatterCpp

def SendDoubleCpp(*args):
  return _DMLLCpp.SendDoubleCpp(*args)
SendDoubleCpp = _DMLLCpp.SendDoubleCpp

def SendIntCpp(*args):
  return _DMLLCpp.SendIntCpp(*args)
SendIntCpp = _DMLLCpp.SendIntCpp

def RecvDoubleCpp(*args):
  return _DMLLCpp.RecvDoubleCpp(*args)
RecvDoubleCpp = _DMLLCpp.RecvDoubleCpp

def RecvIntCpp(*args):
  return _DMLLCpp.RecvIntCpp(*args)
RecvIntCpp = _DMLLCpp.RecvIntCpp
class NumericallyOptimisedMLAlgorithmCpp(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, NumericallyOptimisedMLAlgorithmCpp, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, NumericallyOptimisedMLAlgorithmCpp, name)
    __repr__ = _swig_repr
    def __init__(self): 
        this = _DMLLCpp.new_NumericallyOptimisedMLAlgorithmCpp()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DMLLCpp.delete_NumericallyOptimisedMLAlgorithmCpp
    __del__ = lambda self : None;
    def GetIterationsNeeded(self): return _DMLLCpp.NumericallyOptimisedMLAlgorithmCpp_GetIterationsNeeded(self)
    def GetSumGradients(self, *args): return _DMLLCpp.NumericallyOptimisedMLAlgorithmCpp_GetSumGradients(self, *args)
    def GetParams(self, *args): return _DMLLCpp.NumericallyOptimisedMLAlgorithmCpp_GetParams(self, *args)
    def SetParams(self, *args): return _DMLLCpp.NumericallyOptimisedMLAlgorithmCpp_SetParams(self, *args)
    def GetLengthW(self): return _DMLLCpp.NumericallyOptimisedMLAlgorithmCpp_GetLengthW(self)
NumericallyOptimisedMLAlgorithmCpp_swigregister = _DMLLCpp.NumericallyOptimisedMLAlgorithmCpp_swigregister
NumericallyOptimisedMLAlgorithmCpp_swigregister(NumericallyOptimisedMLAlgorithmCpp)

class OptimiserCpp(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, OptimiserCpp, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, OptimiserCpp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _DMLLCpp.new_OptimiserCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DMLLCpp.delete_OptimiserCpp
    __del__ = lambda self : None;
OptimiserCpp_swigregister = _DMLLCpp.OptimiserCpp_swigregister
OptimiserCpp_swigregister(OptimiserCpp)

class GradientDescentCpp(OptimiserCpp):
    __swig_setmethods__ = {}
    for _s in [OptimiserCpp]: __swig_setmethods__.update(getattr(_s,'__swig_setmethods__',{}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, GradientDescentCpp, name, value)
    __swig_getmethods__ = {}
    for _s in [OptimiserCpp]: __swig_getmethods__.update(getattr(_s,'__swig_getmethods__',{}))
    __getattr__ = lambda self, name: _swig_getattr(self, GradientDescentCpp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _DMLLCpp.new_GradientDescentCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DMLLCpp.delete_GradientDescentCpp
    __del__ = lambda self : None;
GradientDescentCpp_swigregister = _DMLLCpp.GradientDescentCpp_swigregister
GradientDescentCpp_swigregister(GradientDescentCpp)

class BacktrackingLineSearchCpp(OptimiserCpp):
    __swig_setmethods__ = {}
    for _s in [OptimiserCpp]: __swig_setmethods__.update(getattr(_s,'__swig_setmethods__',{}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, BacktrackingLineSearchCpp, name, value)
    __swig_getmethods__ = {}
    for _s in [OptimiserCpp]: __swig_getmethods__.update(getattr(_s,'__swig_getmethods__',{}))
    __getattr__ = lambda self, name: _swig_getattr(self, BacktrackingLineSearchCpp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _DMLLCpp.new_BacktrackingLineSearchCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DMLLCpp.delete_BacktrackingLineSearchCpp
    __del__ = lambda self : None;
BacktrackingLineSearchCpp_swigregister = _DMLLCpp.BacktrackingLineSearchCpp_swigregister
BacktrackingLineSearchCpp_swigregister(BacktrackingLineSearchCpp)

class GradientDescentWithMomentumCpp(OptimiserCpp):
    __swig_setmethods__ = {}
    for _s in [OptimiserCpp]: __swig_setmethods__.update(getattr(_s,'__swig_setmethods__',{}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, GradientDescentWithMomentumCpp, name, value)
    __swig_getmethods__ = {}
    for _s in [OptimiserCpp]: __swig_getmethods__.update(getattr(_s,'__swig_getmethods__',{}))
    __getattr__ = lambda self, name: _swig_getattr(self, GradientDescentWithMomentumCpp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _DMLLCpp.new_GradientDescentWithMomentumCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DMLLCpp.delete_GradientDescentWithMomentumCpp
    __del__ = lambda self : None;
GradientDescentWithMomentumCpp_swigregister = _DMLLCpp.GradientDescentWithMomentumCpp_swigregister
GradientDescentWithMomentumCpp_swigregister(GradientDescentWithMomentumCpp)

class LinearRegressionCpp(NumericallyOptimisedMLAlgorithmCpp):
    __swig_setmethods__ = {}
    for _s in [NumericallyOptimisedMLAlgorithmCpp]: __swig_setmethods__.update(getattr(_s,'__swig_setmethods__',{}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, LinearRegressionCpp, name, value)
    __swig_getmethods__ = {}
    for _s in [NumericallyOptimisedMLAlgorithmCpp]: __swig_getmethods__.update(getattr(_s,'__swig_getmethods__',{}))
    __getattr__ = lambda self, name: _swig_getattr(self, LinearRegressionCpp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _DMLLCpp.new_LinearRegressionCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DMLLCpp.delete_LinearRegressionCpp
    __del__ = lambda self : None;
    def fit(self, *args): return _DMLLCpp.LinearRegressionCpp_fit(self, *args)
    def predict(self, *args): return _DMLLCpp.LinearRegressionCpp_predict(self, *args)
LinearRegressionCpp_swigregister = _DMLLCpp.LinearRegressionCpp_swigregister
LinearRegressionCpp_swigregister(LinearRegressionCpp)

class LinearMahaFeatExtSparseCpp(NumericallyOptimisedMLAlgorithmCpp):
    __swig_setmethods__ = {}
    for _s in [NumericallyOptimisedMLAlgorithmCpp]: __swig_setmethods__.update(getattr(_s,'__swig_setmethods__',{}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, LinearMahaFeatExtSparseCpp, name, value)
    __swig_getmethods__ = {}
    for _s in [NumericallyOptimisedMLAlgorithmCpp]: __swig_getmethods__.update(getattr(_s,'__swig_getmethods__',{}))
    __getattr__ = lambda self, name: _swig_getattr(self, LinearMahaFeatExtSparseCpp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _DMLLCpp.new_LinearMahaFeatExtSparseCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DMLLCpp.delete_LinearMahaFeatExtSparseCpp
    __del__ = lambda self : None;
    def fit(self, *args): return _DMLLCpp.LinearMahaFeatExtSparseCpp_fit(self, *args)
    def transform(self, *args): return _DMLLCpp.LinearMahaFeatExtSparseCpp_transform(self, *args)
LinearMahaFeatExtSparseCpp_swigregister = _DMLLCpp.LinearMahaFeatExtSparseCpp_swigregister
LinearMahaFeatExtSparseCpp_swigregister(LinearMahaFeatExtSparseCpp)

class RBFMahaFeatExtSparseCpp(NumericallyOptimisedMLAlgorithmCpp):
    __swig_setmethods__ = {}
    for _s in [NumericallyOptimisedMLAlgorithmCpp]: __swig_setmethods__.update(getattr(_s,'__swig_setmethods__',{}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, RBFMahaFeatExtSparseCpp, name, value)
    __swig_getmethods__ = {}
    for _s in [NumericallyOptimisedMLAlgorithmCpp]: __swig_getmethods__.update(getattr(_s,'__swig_getmethods__',{}))
    __getattr__ = lambda self, name: _swig_getattr(self, RBFMahaFeatExtSparseCpp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _DMLLCpp.new_RBFMahaFeatExtSparseCpp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _DMLLCpp.delete_RBFMahaFeatExtSparseCpp
    __del__ = lambda self : None;
    def fit(self, *args): return _DMLLCpp.RBFMahaFeatExtSparseCpp_fit(self, *args)
    def transform(self, *args): return _DMLLCpp.RBFMahaFeatExtSparseCpp_transform(self, *args)
RBFMahaFeatExtSparseCpp_swigregister = _DMLLCpp.RBFMahaFeatExtSparseCpp_swigregister
RBFMahaFeatExtSparseCpp_swigregister(RBFMahaFeatExtSparseCpp)

# This file is compatible with both classic and new-style classes.


