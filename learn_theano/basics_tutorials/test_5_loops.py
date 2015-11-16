import theano.tensor as T
import theano
from theano.ifelse import ifelse
import numpy as np


'''
scan is the theano construct that allows you to do recursions and loops.
function that is passed to scan has the following order of arguments:
sequences (if any), prior result(s) (if needed), non-sequences (if any)
Prior results are specified in outputs_info. If some are None, then they are not fed to the function,
but are accumulated from function output.
In this case function returns more values than takes, only part of the return is necessary for the next computation.
'''


def test_5_loops_accumulator():
    '''
    result = 1
    for i in xrange(k):
      result = result * A
    '''
    k = T.iscalar('k')
    A = T.ivector('A')

    result, updates = theano.scan(
        # order of parameters: result from the previous step then non_sequence
        fn=lambda prior_result, A: prior_result*A,
        non_sequences=A,
        n_steps=k,
        # initial value for the accumulator
        outputs_info=T.ones_like(A)
    )

    final_result = result[-1]
    power = theano.function([A, k], final_result, updates=updates)

    np.testing.assert_array_almost_equal(
        power(np.asarray([1, 2, 3], dtype=np.int32), 3),
        [1, 8, 27]
    )


def test_5_loops_forloop_coefficients():
    coefficients = T.vector('coefficients')
    x = T.scalar('x')

    max_coefficients = 100000
    output, updates = theano.scan(
        outputs_info=None,
        # it is ok to pass very long sequence here because theano will truncate result to the shortest one (coeffs)
        sequences=[coefficients, T.arange(max_coefficients)],
        non_sequences=x,
        # since outputs_info is None, there is no first argument - result - passed to the function
        # order is sequences as in "x in list", then non_sequeence
        fn=lambda coefficient, power, variable: coefficient*(variable**power)
    )
    polynomial = theano.function([coefficients, x], output.sum())

    np.testing.assert_array_almost_equal(
        polynomial([1, 2, 3], 2),
        17
    )


def test_5_loops_counter_using_shared():
    a = theano.shared(0)
    k = T.iscalar('k')
    values, updates = theano.scan(lambda: {a: a+1}, n_steps=k)
    # return the last values of a in "updates"
    counter = theano.function([k], updates[a], updates=updates)

    np.testing.assert_array_equal(counter(5), 5)


def test_5_loops_counter_using_shared_with_termination():
    a = theano.shared(0)
    k = T.iscalar('k')
    values, updates = theano.scan(
        fn=lambda: ({a: a+1}, theano.scan_module.until(1)),
        n_steps=k)
    # return the last values of a in "updates"
    counter = theano.function([k], updates[a], updates=updates)

    # theano.scan_module.until(1) will break the iterations right away
    np.testing.assert_array_equal(counter(5), 1)


if __name__ == '__main__':
    test_5_loops_counter_using_shared_with_termination()
