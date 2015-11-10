import theano
import theano.tensor as T
import numpy
import time


# can not run as tets because there is no gpu on travis
def demo_theano_gpu():
    theano.config.floatX = 'float32'
    vlen = 20 * 30 * 768  # multiplier x #cores x # threads per core
    iters = 1000

    rng = numpy.random.RandomState(22)
    x = theano.shared(numpy.asarray(rng.rand(vlen), dtype=theano.config.floatX))
    f = theano.function([], theano.sandbox.cuda.basic_ops.gpu_from_host(T.exp(x)))
    print(f.maker.fgraph.toposort())
    t0 = time.time()
    for i in xrange(iters):
        r = f()
    t1 = time.time()
    print("Looping %d times took %f seconds" % (iters, t1 - t0))
    print("Result is %s" % (r,))
    if numpy.any([isinstance(x.op, theano.tensor.Elemwise) and
                  ('Gpu' not in type(x.op).__name__)
                  for x in f.maker.fgraph.toposort()]):
        print('Used the cpu')
    else:
        print('Used the gpu')


if __name__ == "__main__":
    # to run on gpu, run from terminal THEANO_FLAGS='device=gpu' python learn_theano/basics_tutorials/test_6_gpu.py
    demo_theano_gpu()
    '''
    Results on mac book pro:
    cpu, float32 - 9.411071 seconds
    cpu, float64 - 3.120860 seconds
    gpu, store in numpy, float32 - 1.184434 seconds
    gpu, store in numpy, float64 - 3.011664 seconds
    gpu, store on device, float32 - 0.591813 seconds
    gpu, store on device, float64 - not supported
    '''
