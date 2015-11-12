import cProfile
import pstats
import inspect


def profiler_run(command,
                 how_many_lines_to_print=10,
                 print_callers_of=None,
                 print_callees_of=None,
                 use_walltime=False):
    '''
    Run a string statement under profiler with nice defaults
    command - string statement that goes to exec
    how_many_lines_to_print - how_many_lines_to_print
    print_callers_of - print callers of a function (by string name)
    print_callees_of - print callees of a function (by string name)
    '''
    frame = inspect.currentframe()
    global_vars = frame.f_back.f_globals
    local_vars = frame.f_back.f_locals
    del frame

    profiler = cProfile.Profile()
    try:
        profiler.runctx(command, global_vars, local_vars)
    finally:
        profiler.dump_stats('prof')

    p = pstats.Stats("prof")
    p.strip_dirs().sort_stats('time').print_stats(how_many_lines_to_print)
    if print_callers_of is not None:
        p.print_callers(print_callers_of)
    if print_callees_of is not None:
        p.print_callees(print_callees_of)
