cdef extern from "utilities.cpp":
    pass

cdef extern from "aggregation_two_areas.cpp":
    pass

# Declare the class with cdef
cdef extern from "aggregation_two_areas.h":
    cdef cppclass Problem:
        Problem() except +
        int ninputs
        int noutputs
        void seed(int s)
        void reset()
        double step()
        void close()
        void render()
        double isDone()
        double renderScale()
        void copyObs(float* observation)
        void copyAct(float* action)
        void copyDone(int* done)
        void copyDobj(double* objs)

