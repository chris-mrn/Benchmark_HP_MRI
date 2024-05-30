
My Benchopt HP_MRI Benchmark


Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/chris-mrn/benchmark_HP_MRI
   $ benchopt run benchmark_HP_MRI

Apart from the problem, options can be passed to ``benchopt run``, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run benchmark_HP_MRI -s solver1 -d dataset2 --max-runs 10 --n-repetitions 10


Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

